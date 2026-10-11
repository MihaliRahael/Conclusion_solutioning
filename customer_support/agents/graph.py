"""LangGraph workflow wiring the five agents with conditional routing and a checkpointer.

    supervisor --(route_next)--> customer | order_product | policy | refund | respond
    every specialist --(route_next)--> next planned specialist | respond --> END

Conversation state (`messages`, `memory`) is persisted per `thread_id` (= session ID) by a LangGraph checkpointer,
so a follow-up turn starts from the previous turn's state. Everything else in the state is per-turn and is reset
by `run_chat` at the start of each turn.

If `langgraph` cannot be imported (it needs the `ormsgpack` wheel, which does not exist for every Python
version), a minimal runner with the same add_node / add_conditional_edges / reducer / thread API is used so the
identical graph definition still runs, with in-memory threads. `GRAPH_BACKEND` reports which one is active.
"""
import copy
import logging
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, TypedDict, get_origin, get_type_hints

from agents.customer import customer_agent
from agents.memory import append_messages
from agents.order_product import order_product_agent
from agents.policy import policy_agent
from agents.refund import refund_agent
from agents.supervisor import analyze, respond, route_next

log = logging.getLogger("support")

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    GRAPH_BACKEND = "langgraph"
except Exception as exc:  # ImportError, or a transitive dependency failing at import time
    log.warning("langgraph unavailable (%s); using the built-in graph runner", exc)
    END = "__end__"
    GRAPH_BACKEND = "builtin"
    MemorySaver = None

    class StateGraph:
        def __init__(self, schema=None):
            self.nodes, self.cond, self.edges, self.entry, self.threads = {}, {}, {}, None, {}
            hints = get_type_hints(schema, include_extras=True) if schema else {}
            self.reducers = {k: h.__metadata__[0] for k, h in hints.items() if get_origin(h) is Annotated}

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def set_entry_point(self, name):
            self.entry = name

        def add_edge(self, src, dst):
            self.edges[src] = dst

        def add_conditional_edges(self, src, fn, mapping):
            self.cond[src] = (fn, mapping)

        def compile(self, checkpointer=None):
            self.checkpointer = checkpointer
            return self

        def _merge(self, state: dict, update: dict) -> None:
            for k, v in (update or {}).items():
                state[k] = self.reducers[k](state.get(k), v) if k in self.reducers else v

        @staticmethod
        def _thread(config):
            return ((config or {}).get("configurable") or {}).get("thread_id")

        def get_state(self, config):
            return SimpleNamespace(values=copy.deepcopy(self.threads.get(self._thread(config), {})))

        def invoke(self, inp: dict, config=None) -> dict:
            tid = self._thread(config)
            state = copy.deepcopy(self.threads.get(tid, {})) if tid is not None else {}
            self._merge(state, inp)
            cur = self.entry
            for _ in range(20):  # hard stop against routing loops
                self._merge(state, self.nodes[cur](state))
                if cur in self.cond:
                    fn, mapping = self.cond[cur]
                    cur = mapping[fn(state)]
                else:
                    cur = self.edges.get(cur, END)
                if cur == END:
                    if tid is not None:
                        self.threads[tid] = copy.deepcopy(state)
                    return state
            raise RuntimeError("graph did not terminate")


class ChatState(TypedDict, total=False):
    # persisted across turns by the checkpointer
    messages: Annotated[list, append_messages]  # {"role", "content"} for user and assistant, capped
    memory: dict                                # focus entities, last intents, options, last facts (agents/memory.py)
    # per-turn (reset by run_chat)
    question: str
    analysis: dict
    plan: list
    route: list
    customer: dict
    order: dict
    orders: list
    product: dict
    products: list
    policy: list
    assessment: dict
    sources: list
    warnings: list
    not_found: list
    answer: str
    llm_used: bool


def make_checkpointer(path: str | None = None):
    """SQLite-backed LangGraph checkpointer: conversations survive restarts. ':memory:' is allowed.

    Returns None if LangGraph or its sqlite checkpointer package is unavailable (build_graph then uses memory).
    """
    if GRAPH_BACKEND != "langgraph":
        return None
    path = path or os.getenv("CHECKPOINT_DB") or str(Path(__file__).resolve().parent.parent / ".state" / "checkpoints.sqlite")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    except Exception as exc:
        log.warning("SQLite checkpointer unavailable (%s); conversations will not survive a restart", exc)
        return None


def checkpointer_name(graph) -> str:
    cp = getattr(graph, "checkpointer", None)
    return type(cp).__name__ if cp is not None else "in-memory (built-in runner)"


def build_graph(svc, checkpointer=None):
    g = StateGraph(ChatState)
    g.add_node("supervisor", lambda s: analyze(s, svc))
    g.add_node("customer", lambda s: customer_agent(s, svc))
    g.add_node("order_product", lambda s: order_product_agent(s, svc))
    g.add_node("policy", lambda s: policy_agent(s, svc))
    g.add_node("refund", lambda s: refund_agent(s, svc))
    g.add_node("respond", lambda s: respond(s, svc))
    g.set_entry_point("supervisor")
    targets = {n: n for n in ("customer", "order_product", "policy", "refund", "respond")}
    for src in ("supervisor", "customer", "order_product", "policy", "refund"):
        g.add_conditional_edges(src, route_next, targets)
    g.add_edge("respond", END)
    if GRAPH_BACKEND == "langgraph":
        return g.compile(checkpointer=checkpointer or MemorySaver())
    return g.compile(checkpointer=checkpointer)


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def run_chat(graph, message: str, session_id: str = "default") -> dict:
    """Run one turn of a session and return the final state. Conversation state is loaded from / saved to the checkpointer."""
    message = message.strip()
    turn = {  # `messages` goes through its reducer (appended); `memory` is not passed, so it persists untouched
        "messages": [{"role": "user", "content": message}], "question": message,
        "analysis": {}, "plan": [], "route": [], "warnings": [], "sources": [], "not_found": [],
        "customer": {}, "order": {}, "orders": [], "product": {}, "products": [], "policy": [],
        "assessment": {}, "answer": "", "llm_used": False,
    }
    return graph.invoke(turn, _config(session_id))


def get_session(graph, session_id: str) -> dict:
    """The persisted conversation state of a session ({} if it does not exist)."""
    return dict(graph.get_state(_config(session_id)).values or {})
