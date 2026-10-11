"""Conversation memory: follow-ups resolved against LangGraph-persisted state, per session."""
import sqlite3
from dataclasses import replace

import pytest

from agents import GRAPH_BACKEND, build_graph, get_session, make_checkpointer, run_chat
from agents.memory import MAX_MESSAGES, append_messages
from tests.conftest import DownLLM


def say(graph, q, sid="s1"):
    return run_chat(graph, q, sid)


# ----------------------------------------------------------------- relational follow-ups
def test_what_about_the_product(graph):
    say(graph, "Tell me about order O005")
    s = say(graph, "What about the product?")
    assert s["route"] == ["supervisor", "order_product", "respond"]
    assert s["analysis"]["product_id"] == "P001" and s["analysis"]["resolved"] == {"product_id": "P001"}
    assert "Product 1" in s["answer"] and "unit price 100" in s["answer"]
    assert "Using earlier context: product P001" in s["answer"]


def test_who_ordered_it_and_their_orders(graph):
    say(graph, "Tell me about order O005")
    s = say(graph, "Who ordered it?")
    assert s["analysis"]["customer_id"] == "C005" and "Customer 5" in s["answer"]
    s = say(graph, "And their orders?")
    assert s["route"] == ["supervisor", "order_product", "respond"] and "O005" in s["answer"]


def test_is_it_eligible_uses_the_order_under_discussion(graph):
    say(graph, "Tell me about order O005")
    s = say(graph, "Is it eligible for a refund?")
    assert s["analysis"]["order_id"] == "O005" and s["assessment"]["status"] == "eligible"


def test_focus_survives_a_detour(graph):
    say(graph, "Is order O005 eligible for a refund?")
    say(graph, "What about the product?")        # a product-only turn must not drop the order
    say(graph, "How long does a refund take?")   # nor must a policy question
    s = say(graph, "Is that order refundable?")
    assert s["analysis"]["order_id"] == "O005" and s["assessment"]["status"] == "eligible"


def test_status_and_price_questions_with_pronouns(graph):
    say(graph, "Tell me about order O003")
    assert "Delivered" in say(graph, "What is its status?")["answer"]
    assert "400" in say(graph, "What is the price of the product?")["answer"]


# ------------------------------------------------------------------ carried-over requests
def test_bare_entity_repeats_the_previous_kind_of_request(graph):
    say(graph, "Is order O005 eligible for a refund?")
    s = say(graph, "What about O003?")
    assert s["analysis"]["intents"] == ["refund_eligibility"] and s["assessment"]["order_id"] == "O003"
    assert "same kind of request as before" in s["answer"]
    s = say(graph, "And Customer 7?")  # a customer id is compatible with a refund request too
    assert s["assessment"]["order_id"] == "O007"


def test_bare_entity_after_a_profile_question_stays_a_profile_question(graph):
    say(graph, "Show customer C001")
    s = say(graph, "What about C002?")
    assert s["analysis"]["intents"] == ["customer_info"] and "Customer 2" in s["answer"]


def test_retry_after_a_wrong_id(graph):
    assert say(graph, "Is order O999 eligible for a refund?")["assessment"]["status"] == "insufficient_evidence"
    s = say(graph, "Sorry, I meant O005")
    assert s["assessment"]["status"] == "eligible"


def test_damage_follow_up_reruns_the_refund_for_the_same_order(graph):
    say(graph, "Is order O005 eligible for a refund?")
    s = say(graph, "What if it was damaged?")
    assert s["analysis"]["intents"] == ["refund_eligibility"] and s["analysis"]["issue"] == "damaged"
    assert any("within 7 days" in c for c in s["assessment"]["caveats"])


# ------------------------------------------------------------------------------ "why?"
def test_why_explains_the_previous_assessment_without_rerunning_agents(graph):
    first = say(graph, "Is order O005 eligible for a refund?")
    s = say(graph, "Why?")
    assert s["route"] == ["supervisor", "respond"]                      # no specialist ran
    assert s["assessment"]["status"] == first["assessment"]["status"]   # facts restored from memory
    assert "reasoning behind my refund assessment" in s["answer"] and "Return window" in s["answer"]


def test_why_after_a_detour_still_explains_the_refund_verdict(graph):
    say(graph, "Is order O005 eligible for a refund?")
    say(graph, "Who ordered it?")
    s = say(graph, "Why?")
    assert s["assessment"]["status"] == "eligible" and "Return window" in s["answer"]
    assert s["order"]["OrderID"] == "O005"


def test_why_with_nothing_to_explain(graph):
    s = say(graph, "Why?")
    assert s["route"] == ["supervisor", "respond"] and "customer profiles" in s["answer"]


# ---------------------------------------------------------------- choices and ordinals
@pytest.fixture
def two_orders(svc):
    class TwoOrders(type(svc.store)):
        def orders_for_customer(self, cid):
            return [self.order("O003"), self.order("O005")]

    store = TwoOrders.__new__(TwoOrders)
    store.__dict__.update(svc.store.__dict__)
    return build_graph(replace(svc, store=store))


@pytest.mark.parametrize("reply,order", [("the second one", "O005"), ("the first one", "O003"), ("the latest", "O005"), ("O003", "O003")])
def test_picking_one_of_the_listed_orders(two_orders, reply, order):
    s = say(two_orders, "Can customer C003 get a refund?")
    assert s["assessment"]["status"] == "requires_clarification" and "O003, O005" in s["assessment"]["questions"][0]
    s = say(two_orders, reply)
    assert s["assessment"]["order_id"] == order and s["assessment"]["status"] == "eligible"


# ------------------------------------------------------- unresolvable references, isolation
def test_reference_with_no_context_asks_instead_of_guessing(graph):
    s = say(graph, "What about the product?")
    assert s["analysis"]["unresolved"] == ["product"] and not s["products"] and not s["product"]
    assert "not sure which product" in s["answer"]


def test_sessions_are_isolated(graph):
    say(graph, "Tell me about order O005", sid="alice123")
    s = say(graph, "Is it eligible for a refund?", sid="bob45678")
    assert s["assessment"]["status"] == "requires_clarification"
    assert say(graph, "Is it eligible for a refund?", sid="alice123")["assessment"]["status"] == "eligible"


def test_per_turn_fields_do_not_leak_into_the_next_turn(graph):
    say(graph, "Is order O005 eligible for a refund?")
    s = say(graph, "Show customer C001")
    assert not s["assessment"] and not s["order"] and not s["policy"]
    assert "refund" not in s["route"] and "Refund assessment" not in s["answer"]


# -------------------------------------------------------------------- state & checkpointer
def test_messages_are_kept_in_graph_state(graph):
    say(graph, "Show customer C001")
    say(graph, "Is order O005 eligible for a refund?")
    msgs = get_session(graph, "s1")["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[2]["content"] == "Is order O005 eligible for a refund?" and "ELIGIBLE" in msgs[3]["content"]
    mem = get_session(graph, "s1")["memory"]
    assert mem["focus"] == {"order_id": "O005", "customer_id": "C005", "product_id": "P001"} and mem["turn"] == 2
    assert get_session(graph, "nobody") == {}


def test_message_history_is_capped():
    msgs = []
    for i in range(MAX_MESSAGES + 10):
        msgs = append_messages(msgs, [{"role": "user", "content": str(i)}])
    assert len(msgs) == MAX_MESSAGES and msgs[-1]["content"] == str(MAX_MESSAGES + 9)


@pytest.mark.skipif(GRAPH_BACKEND != "langgraph", reason="needs the real LangGraph checkpointer")
def test_sqlite_checkpointer_persists_across_graph_rebuilds(svc, tmp_path):
    db = str(tmp_path / "cp.sqlite")
    cp = make_checkpointer(db)
    if cp is None:
        pytest.skip("langgraph-checkpoint-sqlite not installed")
    say(build_graph(svc, cp), "Is order O005 eligible for a refund?", sid="persist01")
    cp.conn.close()
    # a brand-new graph and a brand-new connection to the same file == an application restart
    g2 = build_graph(svc, make_checkpointer(db))
    s = say(g2, "What about the product?", sid="persist01")
    assert s["analysis"]["product_id"] == "P001" and "Product 1" in s["answer"]
    assert len(get_session(g2, "persist01")["messages"]) == 4
    assert sqlite3.connect(db).execute("select count(*) from checkpoints").fetchone()[0] > 0


# -------------------------------------------------------------------------- with an LLM
class ContextLLM:
    """Records what the classifier is shown; always classifies the follow-up as product_info."""

    model = "ctx"

    def __init__(self):
        self.prompts = []

    def chat(self, messages, json_mode=False, **k):
        if json_mode:
            self.prompts.append(messages[-1]["content"])
            return '{"intents": ["product_info"]}'
        return "Here are the product details."


def test_llm_classifier_sees_focus_and_history_but_ids_still_come_from_memory(svc):
    llm = ContextLLM()
    g = build_graph(replace(svc, llm=llm))
    say(g, "Tell me about order O005")
    s = say(g, "and the thing they bought?")  # no reference phrase: only the LLM can read this as a product question
    p = llm.prompts[-1]
    assert "Conversation focus: order O005, customer C005, product P001" in p and "Tell me about order O005" in p
    assert s["analysis"]["intents"] == ["product_info"] and s["analysis"]["product_id"] == "P001"


def test_llm_down_follow_ups_still_work_with_keyword_routing(svc):
    g = build_graph(replace(svc, llm=DownLLM()))
    say(g, "Tell me about order O005")
    s = say(g, "What about the product?")
    assert s["analysis"]["product_id"] == "P001" and "Product 1" in s["answer"]
    assert any("language model was unavailable" in w for w in s["warnings"])


# ------------------------------------------------------------------------------- the API
def test_api_sessions_and_follow_up(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "api.sqlite"))
    from fastapi.testclient import TestClient

    from app import app

    with TestClient(app) as c:
        r1 = c.post("/chat", json={"message": "Is order O005 eligible for a refund?"}).json()
        sid = r1["session_id"]
        assert len(sid) == 32 and r1["context"]["focus"]["order_id"] == "O005"
        r2 = c.post("/chat", json={"message": "What about the product?", "session_id": sid}).json()
        assert r2["session_id"] == sid and r2["context"]["resolved"] == {"product_id": "P001"} and "Product 1" in r2["answer"]
        r3 = c.post("/chat", json={"message": "Why?", "session_id": sid}).json()
        assert "reasoning behind my refund assessment" in r3["answer"]
        data = c.get(f"/sessions/{sid}").json()
        assert len(data["messages"]) == 6 and data["memory"]["focus"]["product_id"] == "P001" and data["memory"]["turns"] == 3
        assert c.get("/sessions/doesnotexist1").status_code == 404
        assert c.get("/sessions/bad id!").status_code in (404, 422)
        assert c.post("/chat", json={"message": "hi", "session_id": "x"}).status_code == 422
        assert c.get("/health").json()["checkpointer"]
