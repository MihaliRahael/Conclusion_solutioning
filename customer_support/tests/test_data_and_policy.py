import os

import pandas as pd
import pytest

from ingest import ingest
from services import POLICY_PDF, STORE_DIR, WORKBOOK, DataError, DataStore, PolicyIndex, RefundRules, read_workbook


# ------------------------------------------------------------------ Excel lookups
def test_customer_lookup_is_case_insensitive_and_masks_pii(store):
    c = store.customer("c001")
    assert c["CustomerID"] == "C001" and c["City"] == "City 2"
    assert c["Email"] == "c***@example.com" and c["Phone"].endswith("0001") and "*" in c["Phone"]


def test_order_joins_product_and_flags_amount_mismatch(store):
    o = store.order("O005")
    assert o["CustomerID"] == "C005" and o["product"]["ProductName"] == "Product 1"
    assert o["Quantity"] == 3 and o["Amount"] == 600 and o["expected_amount"] == 300
    assert any("differs from Quantity x UnitPrice" in i for i in o["issues"])


def test_orders_for_customer_and_unknown_ids(store):
    assert [o["OrderID"] for o in store.orders_for_customer("C002")] == ["O002"]
    assert store.customer("C999") is None and store.order("O999") is None and store.product("P999") is None
    assert store.orders_for_customer("C999") == [] and store.customer(None) is None


def test_name_lookup_does_not_confuse_1_and_10(store):
    assert store.find_customer_by_name("tell me about customer 10") == "C010"
    assert store.find_customer_by_name("tell me about Customer 1 please") == "C001"


def _workbook(tmp_path, sheets):
    path = tmp_path / "bad.xlsx"
    with pd.ExcelWriter(path) as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    return path


def test_invalid_workbook_schema_is_reported_at_ingest(tmp_path):
    path = _workbook(tmp_path, {"Customers": pd.DataFrame({"CustomerID": ["C1"], "Name": ["x"]})})
    with pytest.raises(DataError) as e:
        read_workbook(path)
    msg = str(e.value)
    assert "missing sheet 'Orders'" in msg and "missing sheet 'Products'" in msg and "missing columns" in msg
    with pytest.raises(DataError):
        read_workbook(tmp_path / "nope.xlsx")


def test_missing_cells_survive_ingest_and_lookup(tmp_path):
    wb = _workbook(tmp_path, {
        "Customers": pd.DataFrame({"CustomerID": ["C1"], "Name": ["A"], "Email": [None], "Phone": [None], "City": [None]}),
        "Orders": pd.DataFrame({"OrderID": ["O1"], "CustomerID": ["C1"], "ProductID": ["P9"], "OrderDate": ["garbage"],
                                "Quantity": [None], "Amount": [None], "Status": [None]}),
        "Products": pd.DataFrame({"ProductID": ["P1"], "ProductName": ["x"], "Category": ["y"], "UnitPrice": [1]}),
    })
    out = tmp_path / "store"
    ingest(wb, POLICY_PDF, out, tmp_path / "chroma", embeddings=False)
    s = DataStore(out)
    o = s.order("O1")
    assert o["OrderDate"] is None and o["Amount"] is None and o["product"] is None
    assert len(o["issues"]) >= 4
    assert s.customer("C1")["Email"] is None  # missing cell stays None, no crash


def test_missing_or_corrupt_store_tells_you_to_run_ingest(tmp_path):
    with pytest.raises(DataError, match="ingest.py"):
        DataStore(tmp_path)
    for name in ("customers", "orders", "products"):
        (tmp_path / f"{name}.json").write_text("not json")
    with pytest.raises(DataError, match="not valid JSON"):
        DataStore(tmp_path)
    (tmp_path / "customers.json").write_text('[{"CustomerID": "C1"}]')  # wrong columns
    (tmp_path / "orders.json").write_text("[]")
    (tmp_path / "products.json").write_text("[]")
    with pytest.raises(DataError, match="expected columns"):
        DataStore(tmp_path)


@pytest.mark.skipif(not (WORKBOOK.exists() and POLICY_PDF.exists()), reason="source files not present")
def test_committed_store_matches_the_source_files(tmp_path):
    """data/ must be regenerated (python ingest.py) whenever the Excel or PDF changes."""
    ingest(WORKBOOK, POLICY_PDF, tmp_path, tmp_path / "chroma", embeddings=False)
    for name in ("customers", "orders", "products", "policy"):
        assert (tmp_path / f"{name}.json").read_text("utf-8") == (STORE_DIR / f"{name}.json").read_text("utf-8"),             f"{name}.json is stale: run `python ingest.py`"


# ---------------------------------------------------------------- policy retrieval
def test_policy_sections_and_pages(policy):
    assert [c["section"] for c in policy.chunks] == list(range(9))
    assert policy.section(8)["pages"] == [1, 2]  # heading on page 1, bullets on page 2
    assert "Example Questions" not in " ".join(c["title"] for c in policy.chunks)
    assert '"' not in policy.section(7)["text"]


@pytest.mark.parametrize("question,section", [
    ("Can a digital product be refunded?", 2),
    ("How long does a refund take?", 5),
    ("Who pays for return shipping?", 4),
    ("Can a customer receive a partial refund?", 6),
    ("Does a premium customer receive faster processing?", 8),
    ("What if my item arrived damaged?", 3),
])
def test_retrieval_finds_the_right_section(policy, question, section):
    hits = policy.retrieve(question, k=3)
    assert section in [h["section"] for h in hits[:2]], [(h["section"], h["score"]) for h in hits]


def test_empty_retrieval_for_unrelated_query(policy):
    assert policy.retrieve("what is the weather in paris", k=3) == []
    assert policy.retrieve("", k=3) == []


def test_missing_or_corrupt_policy_store_raises(tmp_path):
    with pytest.raises(DataError, match="ingest.py"):
        PolicyIndex(tmp_path, embeddings=False)
    (tmp_path / "policy.json").write_text('{"hash": "x", "chunks": []}')
    with pytest.raises(DataError, match="invalid"):
        PolicyIndex(tmp_path, embeddings=False)


def test_runtime_does_not_need_the_source_files(tmp_path, monkeypatch):
    """The store alone (no xlsx/pdf) is enough to answer."""
    monkeypatch.setenv("WORKBOOK_PATH", str(tmp_path / "missing.xlsx"))
    assert DataStore().order("O005") and PolicyIndex(embeddings=False).retrieve("refund processing time")


@pytest.mark.skipif(os.getenv("RUN_EMBEDDING_TESTS") != "1", reason="set RUN_EMBEDDING_TESTS=1 (needs chromadb + sentence-transformers + model download)")
def test_chroma_index_roundtrip(tmp_path):
    ingest(WORKBOOK, POLICY_PDF, tmp_path / "store", tmp_path / "chroma", embeddings=True)
    idx = PolicyIndex(tmp_path / "store", embeddings=True, chroma_dir=tmp_path / "chroma")
    assert idx.backend == "chroma"
    assert idx.retrieve("Who pays for return shipping?", k=2)[0]["section"] == 4
    assert idx.retrieve("How long until I get my money back?", k=2)[0]["section"] == 5


def test_rules_are_read_from_the_policy_text(rules):
    assert (rules.window_days, rules.auto_reject_days, rules.damage_report_days, rules.manager_approval_above) == (30, 30, 7, 2000)
    assert RefundRules.from_policy("nothing useful here") is None
