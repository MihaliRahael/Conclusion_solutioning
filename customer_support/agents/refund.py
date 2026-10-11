"""Refund Assessment Agent: order facts + policy evidence -> deterministic verdict."""
from services import assess_refund


def refund_agent(state: dict, svc) -> dict:
    a = state["analysis"]
    order_id = a.get("order_id") or (state.get("order") or {}).get("OrderID")  # explicit, or the customer's only order
    assessment = assess_refund(state.get("order"), order_id, state.get("policy") or [], svc.rules, svc.today, a.get("issue"))
    update = {"route": state["route"] + ["refund"], "assessment": assessment.model_dump()}
    if not order_id and any("several order IDs" in w for w in state["warnings"]):  # not "none given" but "too many given"
        update["assessment"]["summary"] = "The question mentions more than one order, so I cannot tell which to assess."
        update["assessment"]["questions"] = ["Which one order should I check? Please ask about them one at a time."]
    # A customer ID without an order ID: list their orders so the user can pick one.
    if not order_id and state.get("orders"):
        ids = ", ".join(o["OrderID"] for o in state["orders"])
        update["assessment"]["questions"] = [f"Which order should I check? Orders on record: {ids}."]
    return update
