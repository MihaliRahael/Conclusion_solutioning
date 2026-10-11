"""Streamlit chat UI. Talks to the FastAPI backend over HTTP."""
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
EXAMPLES = [
    "Show customer C001",
    "What orders did C002 place?",
    "Tell me about product P003",
    "How long does a refund take?",
    "Is order O005 eligible for a refund?",
    "What is the status of order O999?",
]
STATUS_ICON = {"eligible": "✅", "ineligible": "❌", "insufficient_evidence": "⚠️", "requires_clarification": "❓"}

st.set_page_config(page_title="ABC Retail Support", page_icon="💬", layout="centered")
st.title("💬 ABC Retail Support Assistant")
st.session_state.setdefault("messages", [])

# The conversation lives on the server (LangGraph checkpointer), keyed by a session id kept in the URL, so a page
# refresh reopens the same conversation. The UI only sends the new message and the session id.
if "session_id" not in st.session_state:
    st.session_state.session_id = st.query_params.get("sid") or uuid.uuid4().hex
    st.query_params["sid"] = st.session_state.session_id
    try:  # restore the visible transcript if the server already knows this session
        saved = requests.get(f"{API_URL}/sessions/{st.session_state.session_id}", timeout=5)
        if saved.ok:
            st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in saved.json()["messages"]]
    except Exception:
        pass


@st.cache_data(ttl=5)
def get_health(url: str) -> dict:
    try:
        return requests.get(f"{url}/health", timeout=5).json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


with st.sidebar:
    st.header("Status")
    health = get_health(API_URL)
    if health["status"] == "ok":
        st.success("Backend ready")
        llm = health.get("llm") or {}
        (st.success if llm.get("reachable") else st.warning)(
            f"LLM {llm.get('model', '?')}: " + ("reachable" if llm.get("reachable") else "not reachable - template answers"))
        st.caption(f"Retrieval: {health['policy']['retrieval']} · Graph: {health['graph_backend']} · Ref. date {health['reference_date']}")
    else:
        st.error(f"Backend {health['status']}: {health.get('error', '')}")
    st.header("Try asking")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    focus = st.session_state.get("focus") or {}
    if focus:
        st.header("Conversation focus")
        st.caption(" · ".join(f"{k.replace('_id', '')} **{v}**" for k, v in focus.items()))
    if st.button("New conversation"):  # a new session id = a fresh server-side memory
        st.session_state.messages, st.session_state.focus = [], {}
        st.session_state.session_id = uuid.uuid4().hex
        st.query_params["sid"] = st.session_state.session_id
        st.rerun()


def render_evidence(msg: dict) -> None:
    if a := msg.get("assessment"):
        with st.expander(f"{STATUS_ICON.get(a['status'], '')} Refund assessment: {a['status']}"):
            for c in a["checks"]:
                icon = {"pass": "✅", "fail": "❌", "unknown": "❔"}[c["result"]]
                st.markdown(f"{icon} **{c['name']}** - {c['detail']}" + (f" _{c['policy_ref']}_" if c.get("policy_ref") else ""))
            for c in a["caveats"]:
                st.caption(f"• {c}")
    if msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for s in msg["sources"]:
                st.markdown(f"**{'📄 PDF' if s['type'] == 'pdf' else '📊 Excel'}** - {s['ref']}")
                if s.get("snippet"):
                    st.caption(s["snippet"])
    for w in msg.get("warnings", []):
        st.caption(f"⚠️ {w}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_evidence(m)

prompt = st.chat_input("Ask about a customer, order, product or the refund policy") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(f"{API_URL}/chat", json={"message": prompt, "session_id": st.session_state.session_id}, timeout=300)
                r.raise_for_status()
                reply = {"role": "assistant", **r.json()}
                reply["content"] = reply.pop("answer")
                st.session_state.focus = reply["context"].get("focus", {})
            except requests.HTTPError as exc:
                detail = exc.response.json().get("detail", exc.response.text) if exc.response.content else str(exc)
                reply = {"role": "assistant", "content": f"The service returned an error: {detail}"}
            except Exception as exc:
                reply = {"role": "assistant", "content": f"I couldn't reach the backend at {API_URL}. Is it running? ({exc})"}
        st.markdown(reply["content"])
        render_evidence(reply)
    st.session_state.messages.append(reply)
