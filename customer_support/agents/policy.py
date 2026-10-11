"""Policy Agent: retrieves refund-policy passages with page references.

Retrieved text is untrusted data. It is length-capped here and wrapped in delimiters when shown
to the LLM; no business rule is ever derived from it by the model.
"""

MAX_CHARS = 1200
MANDATORY_FOR_REFUND = (1, 2, 8)  # eligibility, non-refundable items, special business rules


def policy_agent(state: dict, svc) -> dict:
    refund = "refund_eligibility" in state["analysis"]["intents"]
    query = state["question"] + (" refund eligibility non-refundable special business rules" if refund else "")
    hits = svc.policy.retrieve(query, k=4)
    if refund:
        have = {h["section"] for h in hits}
        hits += [dict(c, score=0.0) for n in MANDATORY_FOR_REFUND if n not in have and (c := svc.policy.section(n))]
    passages = [dict(h, text=h["text"][:MAX_CHARS]) for h in hits]
    update = {"route": state["route"] + ["policy"], "policy": passages}
    if passages:
        update["sources"] = state["sources"] + [
            {"type": "pdf", "ref": f"Refund policy {p['ref']}", "snippet": p["text"][:300]} for p in passages
        ]
    else:
        update["warnings"] = state["warnings"] + ["No relevant passage was found in the refund policy."]
    return update
