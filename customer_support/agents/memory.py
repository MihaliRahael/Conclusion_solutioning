"""Conversation memory: what the user is talking about, and how follow-up questions resolve against it.

The memory dict lives in the LangGraph state next to `messages`, so the checkpointer persists both.

    memory = {
      "focus":        {"customer_id", "order_id", "product_id"},  # entities in play, incl. ones derived by joins
      "last_intents": [...],                                       # what the previous turn was about
      "options":      {"kind": "order", "ids": [...], "dates": [...]} | None,   # choices just listed to the user
      "last":         {assessment, order, customer, product, orders, products, policy, sources},  # for "why?"
      "last_refund":  {assessment, order, policy, sources},        # the most recent verdict, so "why?" still works after a detour
      "turn":         n,
    }

Entity binding is deterministic (IDs come from this memory or from the user's text, never from an LLM),
so a follow-up like "What about the product?" cannot be pointed at an invented ID.
"""
import re

MAX_MESSAGES = 40
LAST_KEYS = ("assessment", "order", "customer", "product", "orders", "products", "policy", "sources")

REF_RE = {
    "product": re.compile(r"\b(the|its|that|this|their|same)\s+(product|item|unit)s?\b", re.I),
    "customer": re.compile(r"\b(the|that|this|same)\s+(customer|buyer|person|client)\b|\b(their|his|her)\b", re.I),
    "order": re.compile(r"\b(the|that|this|same|last|previous)\s+(order|purchase)\b", re.I),
}
WHO_RE = re.compile(r"\bwho\s+(ordered|bought|placed|made|purchased)\b|\bwhose\b", re.I)
PRONOUN_RE = re.compile(r"\b(it|its|this|that|they|them|those)\b", re.I)
EXPLAIN_RE = re.compile(r"^\W*(why|how come|explain|what\s+do\s+you\s+mean|can\s+you\s+explain|on\s+what\s+basis|what.s\s+the\s+reason)\b", re.I)
ORDERS_WORD = re.compile(r"\border|purchase|bought|history", re.I)
ORDER_FACT = re.compile(r"status|deliver|placed|amount|total|when|date", re.I)
PRODUCT_FACT = re.compile(r"price|cost|categor|unit|name", re.I)
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "oldest": "oldest",
            "last": "latest", "latest": "latest", "newest": "latest", "recent": "latest"}
KIND_COMPAT = {"order_id": {"order_info", "refund_eligibility"},
               "customer_id": {"customer_info", "order_info", "refund_eligibility"},
               "product_id": {"product_info"}}
KIND_DEFAULT = {"order_id": "order_info", "customer_id": "customer_info", "product_id": "product_info"}
LABEL = {"order_id": "order", "customer_id": "customer", "product_id": "product"}


def append_messages(old, new):
    """LangGraph reducer for the persisted conversation: append, keep the most recent MAX_MESSAGES."""
    return ((old or []) + list(new or []))[-MAX_MESSAGES:]


def _ordinal(q: str, options: dict) -> int | None:
    """'the second one' / 'the latest' -> index into the options just offered to the user."""
    ids = (options or {}).get("ids") or []
    if not ids:
        return None
    for word, idx in ORDINALS.items():
        if re.search(rf"\b{word}\b", q, re.I):
            if idx == "latest":
                dates = options.get("dates") or []
                return max(range(len(ids)), key=lambda i: dates[i] or "") if len(dates) == len(ids) else len(ids) - 1
            if idx == "oldest":
                dates = options.get("dates") or []
                return min(range(len(ids)), key=lambda i: dates[i] or "~") if len(dates) == len(ids) else 0
            return idx if idx < len(ids) else None
    return None


def interpret(q: str, ids: dict, intents: list[str], memory: dict) -> dict:
    """Resolve a (possibly elliptical) question against the conversation memory.

    Returns {intents, ids, resolved, unresolved, carried, explain}. `resolved` maps id-kind -> the ID that was
    filled in from memory; `unresolved` lists kinds the user referred to that memory cannot supply.
    """
    focus, last = memory.get("focus") or {}, memory.get("last_intents") or []
    options = memory.get("options") or {}
    out = {"intents": [i for i in intents if i != "out_of_scope"], "ids": dict(ids), "resolved": {}, "unresolved": [],
           "carried": False, "explain": False}

    if EXPLAIN_RE.search(q) and memory.get("last"):  # "why?" -> explain the previous answer, run nothing new
        out.update(intents=["explain_previous"], explain=True)
        return out

    refs = {k: bool(rx.search(q)) for k, rx in REF_RE.items()}
    pronoun = bool(PRONOUN_RE.search(q))
    ordinal = _ordinal(q, options)
    given = next((k for k in ("order_id", "customer_id", "product_id") if out["ids"][k]), None)

    # 1. Intent when the question itself carries no topic keywords
    if not out["intents"]:
        if WHO_RE.search(q):
            out["intents"] = ["customer_info"]
        elif given:  # bare entity: "And O004?" / "What about Customer 3?" -> repeat the previous kind of request
            carry = [i for i in last if i in KIND_COMPAT[given]]
            out["intents"], out["carried"] = carry or [KIND_DEFAULT[given]], bool(carry)
        elif ordinal is not None:  # "the second one"
            carry = [i for i in last if i in KIND_COMPAT["order_id"]]
            out["intents"], out["carried"] = carry or ["order_info"], bool(carry)
        elif refs["product"] or (pronoun and PRODUCT_FACT.search(q)):
            out["intents"] = ["product_info"]
        elif refs["customer"]:
            out["intents"] = ["order_info" if ORDERS_WORD.search(q) else "customer_info"]
        elif refs["order"] or (pronoun and ORDER_FACT.search(q)):
            out["intents"] = ["order_info"]
    # "What if it was damaged?" is a refund question about the order under discussion, not a generic policy question
    if out["intents"] == ["policy_question"] and focus.get("order_id") and (refs["order"] or pronoun) \
            and re.search(r"damag|defect|broken|faulty", q, re.I):
        out["intents"] = ["refund_eligibility"]

    # 2. Bind entities the user referred to but did not name
    want = set(out["intents"])
    ctx = any(refs.values()) or pronoun or out["carried"] or ordinal is not None
    ids_ = out["ids"]

    def bind(kind: str, report: bool = True) -> None:
        if focus.get(kind):
            ids_[kind] = out["resolved"][kind] = focus[kind]
        elif report:
            out["unresolved"].append(LABEL[kind])

    if want & {"refund_eligibility", "order_info"} and not ids_["order_id"]:
        if ordinal is not None:
            ids_["order_id"] = out["resolved"]["order_id"] = options["ids"][ordinal]
        elif refs["customer"] and "order_info" in want:  # "their orders"
            if not ids_["customer_id"]:
                bind("customer_id")
        elif ctx and not ids_["customer_id"]:
            bind("order_id", report="refund_eligibility" not in want)  # the refund agent asks for the order itself
    if "customer_info" in want and not ids_["customer_id"] and ctx:
        bind("customer_id")
    if "product_info" in want and not ids_["product_id"] and not ids_["order_id"] and (refs["product"] or pronoun or out["carried"]):
        bind("product_id")
    return out


def describe(resolved: dict, carried: bool, intents: list[str]) -> str | None:
    """Human-readable note about what was inferred from earlier turns (shown to the user)."""
    parts = [f"{LABEL[k]} {v}" for k, v in resolved.items()]
    if carried:
        parts.append("same kind of request as before (" + ", ".join(i.replace("_", " ") for i in intents) + ")")
    return "Using earlier context: " + "; ".join(parts) if parts else None


def update_memory(state: dict) -> dict:
    """Return the memory to persist after this turn."""
    old = state.get("memory") or {}
    new = {"focus": dict(old.get("focus") or {}), "last_intents": list(old.get("last_intents") or []),
           "options": None, "last": old.get("last") or {}, "last_refund": old.get("last_refund") or {},
           "turn": old.get("turn", 0) + 1}
    a = state["analysis"]
    if "explain_previous" in a["intents"]:
        return {**new, "options": old.get("options")}

    order, orders = state.get("order"), state.get("orders") or []
    if order:  # an order pins down its customer and product too
        new["focus"] = {"order_id": order["OrderID"], "customer_id": order["CustomerID"], "product_id": order["ProductID"]}
    else:
        cid = (state.get("customer") or {}).get("CustomerID") or (a.get("customer_id") if orders else None)
        if cid:
            if cid != new["focus"].get("customer_id"):
                new["focus"] = {"customer_id": cid}  # new subject: stale order/product no longer apply
            if len(orders) == 1:
                new["focus"].update(order_id=orders[0]["OrderID"], product_id=orders[0]["ProductID"])
        if state.get("product"):
            new["focus"]["product_id"] = state["product"]["ProductID"]
    if len(orders) > 1:
        new["options"] = {"kind": "order", "ids": [o["OrderID"] for o in orders], "dates": [o["OrderDate"] for o in orders]}
    real = [i for i in a["intents"] if i != "out_of_scope"]
    if real:
        new["last_intents"] = real
        new["last"] = {k: state[k] for k in LAST_KEYS if state.get(k)}
    if state.get("assessment"):
        new["last_refund"] = {k: state[k] for k in ("assessment", "order", "policy", "sources") if state.get(k)}
    return new
