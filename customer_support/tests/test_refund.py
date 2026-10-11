from datetime import date

from services import assess_refund

TODAY = date(2026, 7, 20)


def run(store, policy, rules, oid, today=TODAY, order=..., issue=None, pol=...):
    order = store.order(oid) if order is ... else order
    pol = policy.retrieve("refund eligibility", k=4) if pol is ... else pol
    return assess_refund(order, oid, pol, rules, today, issue)


def test_o005_eligible_within_window_with_caveats(store, policy, rules):
    a = run(store, policy, rules, "O005")
    assert a.status == "eligible"
    assert {c.name: c.result for c in a.checks} == {"Delivery status": "pass", "Return window": "pass", "Product restrictions": "pass"}
    text = " ".join(a.caveats)
    assert "delivery date" in text and "Not verifiable" in text and "Data quality" in text  # honest about gaps


def test_o005_ineligible_on_real_today(store, policy, rules):
    a = run(store, policy, rules, "O005", today=date(2026, 10, 11))
    assert a.status == "ineligible"
    fail = next(c for c in a.checks if c.result == "fail")
    assert "older than 30 days" in fail.detail and "§8" in fail.policy_ref


def test_boundary_day_30_is_still_eligible_day_31_is_not(store, policy, rules):
    assert run(store, policy, rules, "O005", today=date(2026, 8, 4)).status == "eligible"    # 30 days
    assert run(store, policy, rules, "O005", today=date(2026, 8, 5)).status == "ineligible"  # 31 days


def test_unknown_order_and_missing_order_id(store, policy, rules):
    assert run(store, policy, rules, "O999").status == "insufficient_evidence"
    a = assess_refund(None, None, [], rules, TODAY)
    assert a.status == "requires_clarification" and a.questions


def test_empty_policy_or_unreadable_rules_never_yield_eligible(store, policy, rules):
    assert run(store, policy, rules, "O005", pol=[]).status == "insufficient_evidence"
    assert run(store, policy, None, "O005").status == "insufficient_evidence"


def test_order_not_yet_delivered_is_insufficient(store, policy, rules):
    o = dict(store.order("O005"), Status="Shipped")
    assert run(store, policy, rules, "O005", order=o).status == "insufficient_evidence"


def test_cancelled_order_is_ineligible(store, policy, rules):
    o = dict(store.order("O005"), Status="Cancelled")
    assert run(store, policy, rules, "O005", order=o).status == "ineligible"


def test_missing_or_future_date_is_insufficient(store, policy, rules):
    assert run(store, policy, rules, "O005", order=dict(store.order("O005"), OrderDate=None)).status == "insufficient_evidence"
    assert run(store, policy, rules, "O005", today=date(2026, 6, 1)).status == "insufficient_evidence"


def test_high_value_order_needs_manager_approval(store, policy, rules):
    o = dict(store.order("O005"), Amount=2500)
    a = run(store, policy, rules, "O005", order=o)
    assert a.status == "eligible" and a.needs_manager_approval


def test_product_restrictions(store, policy, rules):
    base = store.order("O005")
    digital = dict(base, product=dict(base["product"], Category="Digital Software"))
    a = run(store, policy, rules, "O005", order=digital)
    assert a.status == "requires_clarification" and "activated" in a.questions[0]
    gift = dict(base, product=dict(base["product"], ProductName="Gift Card"))
    assert run(store, policy, rules, "O005", order=gift).status == "ineligible"


def test_damage_claim_adds_7_day_caveat(store, policy, rules):
    a = run(store, policy, rules, "O005", issue="damaged")  # 15 days old
    assert any("within 7 days" in c for c in a.caveats)
