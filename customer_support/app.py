"""FastAPI backend: GET /health, POST /chat."""
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents import GRAPH_BACKEND, build_graph, checkpointer_name, get_session, make_checkpointer, run_chat
from services import DataError, build_services, load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("support")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.svc = app.state.graph = app.state.init_error = None
    try:  # load workbook + policy index once; reused by every request
        app.state.svc = build_services()
        app.state.graph = build_graph(app.state.svc, make_checkpointer())  # conversation state persists per session
        log.info("Ready: %d policy sections, retrieval=%s, graph=%s, checkpointer=%s", len(app.state.svc.policy.chunks),
                 app.state.svc.policy.backend, GRAPH_BACKEND, checkpointer_name(app.state.graph))
    except DataError as exc:
        app.state.init_error = str(exc)
        log.error("Startup failed: %s", exc)
    yield


app = FastAPI(title="Customer Support Assistant", version="1.0", lifespan=lifespan)


SESSION_ID = r"^[A-Za-z0-9_-]{8,64}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=SESSION_ID)  # omit to start a new conversation


class Source(BaseModel):
    type: Literal["excel", "pdf"]
    ref: str
    snippet: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source] = []
    refund_status: str | None = None
    assessment: dict | None = None
    route: list[str] = []
    warnings: list[str] = []
    llm_used: bool = False
    context: dict = {}  # what was inferred from earlier turns, and the current conversation focus


@app.get("/health")
def health():
    svc = app.state.svc
    if svc is None:
        return {"status": "degraded", "error": app.state.init_error}  # e.g. "run python ingest.py"
    return {
        "status": "ok",
        "workbook": {"customers": len(svc.store.customers), "orders": len(svc.store.orders), "products": len(svc.store.products)},
        "policy": {"sections": len(svc.policy.chunks), "retrieval": svc.policy.backend, "rules_loaded": svc.rules is not None},
        "llm": {"model": svc.llm.model, "reachable": svc.llm.available()} if svc.llm else None,
        "graph_backend": GRAPH_BACKEND,
        "checkpointer": checkpointer_name(app.state.graph),
        "ingested_at": load_manifest().get("ingested_at"),
        "reference_date": svc.today.isoformat(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if app.state.graph is None:
        raise HTTPException(503, f"Service is not ready: {app.state.init_error}")
    message = req.message.strip()
    if not message:
        raise HTTPException(422, "Message must not be blank.")
    sid = req.session_id or uuid.uuid4().hex
    try:
        state = run_chat(app.state.graph, message, sid)
    except Exception:  # never leak a stack trace to the client
        log.exception("Chat turn failed")
        return ChatResponse(session_id=sid, answer="Sorry, something went wrong while processing that. Please try again.",
                            warnings=["Internal error; see server logs."])
    assessment = state.get("assessment") or None
    a = state["analysis"]
    return ChatResponse(
        session_id=sid, answer=state["answer"], sources=state["sources"], assessment=assessment,
        refund_status=assessment["status"] if assessment else None,
        route=state["route"], warnings=list(dict.fromkeys(state["warnings"])), llm_used=state.get("llm_used", False),
        context={"interpreted": a.get("interpreted"), "resolved": a.get("resolved", {}),
                 "focus": (state.get("memory") or {}).get("focus", {})},
    )


@app.get("/sessions/{session_id}")
def session(session_id: str):
    """The persisted conversation (messages + memory) of a session, as held by the LangGraph checkpointer."""
    if app.state.graph is None:
        raise HTTPException(503, f"Service is not ready: {app.state.init_error}")
    if not re.match(SESSION_ID, session_id):
        raise HTTPException(422, "Invalid session id.")
    values = get_session(app.state.graph, session_id)
    if not values.get("messages"):
        raise HTTPException(404, "No such session.")
    mem = values.get("memory") or {}
    return {"session_id": session_id, "messages": values["messages"],
            "memory": {"focus": mem.get("focus", {}), "last_intents": mem.get("last_intents", []), "turns": mem.get("turn", 0)}}
