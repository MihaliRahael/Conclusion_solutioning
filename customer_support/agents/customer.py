"""Customer Agent: retrieves a customer profile (PII masked by the data layer)."""


def customer_agent(state: dict, svc) -> dict:
    a = state["analysis"]
    cid = a.get("customer_id")
    update = {"route": state["route"] + ["customer"]}
    if not cid:
        update["warnings"] = state["warnings"] + ["No customer ID or name was found in the question."]
        update["not_found"] = state["not_found"] + [{"kind": "customer", "id": None}]
        return update
    profile = svc.store.customer(cid)
    if profile is None:
        update["warnings"] = state["warnings"] + [f"Customer {cid} was not found."]
        update["not_found"] = state["not_found"] + [{"kind": "customer", "id": cid}]
        return update
    update["customer"] = profile
    update["sources"] = state["sources"] + [{"type": "excel", "ref": f"Customers sheet, {cid}", "snippet": None}]
    return update
