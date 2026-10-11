"""Static data store access, policy retrieval, deterministic refund rules and the local LLM client.

Source files (Excel, PDF) are converted once by `ingest.py` into a static store:
  data/*.json      Excel sheets as JSON records, loaded by DataStore
  data/policy.json policy sections + page numbers + refund thresholds, loaded by PolicyIndex
  chroma_db/       persisted Chroma collection with the policy-section embeddings
At runtime only that store is read; the Excel and PDF are not needed.

Python does exact lookups, joins, validation and business rules; the LLM only interprets and phrases.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).with_name(".env"))
log = logging.getLogger("support")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR") or ROOT)
WORKBOOK = Path(os.getenv("WORKBOOK_PATH") or DATA_DIR / "Sample_Customer_Data.xlsx")
POLICY_PDF = Path(os.getenv("POLICY_PDF_PATH") or DATA_DIR / "Detailed_Sample_Refund_Policy.pdf")
STORE_DIR = Path(os.getenv("STORE_DIR") or Path(__file__).with_name("data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR") or Path(__file__).with_name("chroma_db"))
COLLECTION = "refund_policy"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INGEST_HINT = "Run `python ingest.py` to (re)build the data store."

SCHEMA = {
    "Customers": ["CustomerID", "Name", "Email", "Phone", "City"],
    "Orders": ["OrderID", "CustomerID", "ProductID", "OrderDate", "Quantity", "Amount", "Status"],
    "Products": ["ProductID", "ProductName", "Category", "UnitPrice"],
}


class DataError(Exception):
    """A source file or the static data store is missing or does not have the expected shape."""


# --------------------------------------------------------------------------- Excel


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else (int(f) if f.is_integer() else f)


def _date(v) -> date | None:
    ts = pd.to_datetime(v, errors="coerce")
    return None if pd.isna(ts) else ts.date()


def mask_email(e: str | None) -> str | None:
    if not e:
        return e
    local, _, domain = e.partition("@")
    return f"{local[:1]}***@{domain}" if domain else "***"


def mask_phone(p: str | None) -> str | None:
    return p if not p else "*" * max(len(p) - 4, 0) + p[-4:]


def read_workbook(path: Path | str) -> dict[str, list[dict]]:
    """Read and validate the Excel workbook into plain records (used by ingest.py)."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"Workbook not found: {path}")
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str, engine="openpyxl")
    except Exception as exc:  # corrupt file, wrong format, locked file
        raise DataError(f"Could not read workbook {path.name}: {exc}") from exc
    problems = []
    for sheet, cols in SCHEMA.items():
        if sheet not in sheets:
            problems.append(f"missing sheet '{sheet}'")
        elif missing := [c for c in cols if c not in sheets[sheet].columns]:
            problems.append(f"sheet '{sheet}' is missing columns {missing}")
    if problems:
        raise DataError("Invalid workbook schema: " + "; ".join(problems))
    return {s: [{c: (None if pd.isna(v) else str(v).strip()) for c, v in row.items()}
                for row in sheets[s][cols].to_dict("records")] for s, cols in SCHEMA.items()}


class DataStore:
    """Reads the static JSON store (see ingest.py) once; lookups are exact key matches on the cached frames."""

    def __init__(self, store_dir: Path | str = STORE_DIR, mask_pii: bool | None = None):
        frames, problems = {}, []
        for sheet, cols in SCHEMA.items():
            f = Path(store_dir) / f"{sheet.lower()}.json"
            if not f.exists():
                problems.append(f"{f.name} not found")
                continue
            try:
                records = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                problems.append(f"{f.name} is not valid JSON")
                continue
            if not isinstance(records, list) or any(not isinstance(r, dict) or not set(cols) <= r.keys() for r in records):
                problems.append(f"{f.name} does not match the expected columns {cols}")
                continue
            frames[sheet] = pd.DataFrame(records, columns=cols, dtype=str)
        if problems:
            raise DataError(f"Data store problem in {store_dir}: " + "; ".join(problems) + ". " + INGEST_HINT)
        self.customers = self._prep(frames["Customers"], "CustomerID")
        self.orders = self._prep(frames["Orders"], "OrderID")
        self.products = self._prep(frames["Products"], "ProductID")
        self.mask_pii = (os.getenv("MASK_PII", "true").lower() != "false") if mask_pii is None else mask_pii

    @staticmethod
    def _prep(df: pd.DataFrame, key: str) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v).strip())
        df = df[df[key].notna() & (df[key] != "")]
        df[key] = df[key].str.upper()
        df = df.drop_duplicates(key).set_index(key, drop=False)
        df.index.name = None
        return df

    @staticmethod
    def _row(df: pd.DataFrame, key: str | None) -> dict | None:
        key = (key or "").strip().upper()
        if key not in df.index:
            return None
        return {k: (None if pd.isna(v) else v) for k, v in df.loc[key].items()}

    # -- customers
    def customer(self, cid: str | None) -> dict | None:
        row = self._row(self.customers, cid)
        if row and self.mask_pii:
            row["Email"], row["Phone"] = mask_email(row["Email"]), mask_phone(row["Phone"])
        return row

    def find_customer_by_name(self, text: str) -> str | None:
        for cid, name in self.customers["Name"].items():
            if name and re.search(rf"\b{re.escape(name)}\b", text, re.I):
                return cid
        return None

    # -- products
    def product(self, pid: str | None) -> dict | None:
        row = self._row(self.products, pid)
        if row:
            row["UnitPrice"] = _num(row["UnitPrice"])
        return row

    def find_product_by_name(self, text: str) -> str | None:
        for pid, name in self.products["ProductName"].items():
            if name and re.search(rf"\b{re.escape(name)}\b", text, re.I):
                return pid
        return None

    def list_products(self) -> list[dict]:
        return [self.product(pid) for pid in self.products.index]

    # -- orders (joined with product, with data-quality notes)
    def order(self, oid: str | None) -> dict | None:
        row = self._row(self.orders, oid)
        if not row:
            return None
        qty, amount, odate = _num(row["Quantity"]), _num(row["Amount"]), _date(row["OrderDate"])
        product = self.product(row["ProductID"])
        issues = []
        if odate is None:
            issues.append("OrderDate is missing or invalid")
        if qty is None:
            issues.append("Quantity is missing or invalid")
        if amount is None:
            issues.append("Amount is missing or invalid")
        if not row["Status"]:
            issues.append("Status is missing")
        if row["ProductID"] and product is None:
            issues.append(f"ProductID {row['ProductID']} was not found in Products")
        if (row["CustomerID"] or "").upper() not in self.customers.index:
            issues.append(f"CustomerID {row['CustomerID']} was not found in Customers")
        expected = None
        if product and qty is not None and product["UnitPrice"] is not None:
            expected = qty * product["UnitPrice"]
            if amount is not None and abs(expected - amount) > 0.01:
                issues.append(
                    f"Amount {amount:g} differs from Quantity x UnitPrice "
                    f"({qty:g} x {product['UnitPrice']:g} = {expected:g})"
                )
        return {
            "OrderID": row["OrderID"], "CustomerID": row["CustomerID"], "ProductID": row["ProductID"],
            "OrderDate": odate.isoformat() if odate else None, "Quantity": qty, "Amount": amount,
            "Status": row["Status"], "product": product, "expected_amount": expected, "issues": issues,
        }

    def orders_for_customer(self, cid: str | None) -> list[dict]:
        cid = (cid or "").strip().upper()
        ids = self.orders.index[self.orders["CustomerID"].str.upper() == cid]
        return [self.order(o) for o in ids]


# ----------------------------------------------------------------------- PDF policy

SECTION_RE = re.compile(r"^(\d+)\.\s+(.{3,60})$")
STOP = set(
    "a an the is are be been to of for in on and or can i my it this that do does how what who when which "
    "with at by from if as will would should could there their its you your me we get".split()
)
# Stemmed query term -> policy vocabulary it should also match (the PDF never says "how long").
EXPAND = {
    "long": ["day", "process", "business"], "take": ["process", "day"], "much": ["fee"],
    "fast": ["priority", "process"], "faster": ["priority", "process"], "quick": ["priority", "process"],
    "quicker": ["priority", "process"], "speed": ["priority", "process"], "premium": ["priority"],
    "pay": ["responsible", "shipping"], "cost": ["shipping", "fee"], "fee": ["shipping"],
    "eligible": ["eligibility", "qualify", "day"], "eligibility": ["eligible", "qualify"],
    "time": ["day", "calendar"], "window": ["day", "calendar"], "deadline": ["day", "calendar"],
    "late": ["day", "older"], "old": ["day", "older"], "expire": ["day"],
    "broken": ["damage", "defective"], "faulty": ["damage", "defective"], "defective": ["damage"],
    "wrong": ["incorrect", "defective"], "incorrect": ["defective"],
    "expensive": ["above", "manager"], "large": ["above", "manager"], "big": ["above", "manager"],
    "business": ["corporate", "bulk"], "company": ["corporate", "bulk"], "swap": ["exchange", "replacement"],
    "replace": ["replacement", "exchange"], "partial": ["partial"], "fraud": ["fraudulent", "suspension"],
}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


def _pages(pages: list[int]) -> str:
    return f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"


def parse_policy_pdf(path: Path | str) -> list[dict]:
    """Split the policy PDF into numbered sections with page numbers (used by ingest.py)."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"Policy PDF not found: {path}")
    try:
        import pymupdf

        doc = pymupdf.open(path)
    except Exception as exc:
        raise DataError(f"Could not read policy PDF {path.name}: {exc}") from exc
    lines: list[tuple[int, str]] = []
    for page_no, page in enumerate(doc, start=1):
        for raw in page.get_text().splitlines():
            text = raw.strip()
            if raw.count('"') == 1:  # stray quote artifacts in the source PDF
                text = text.replace('"', "").strip()
            if text:
                lines.append((page_no, text))
    chunks: list[dict] = []
    cur = None
    for page_no, text in lines:
        m = SECTION_RE.match(text)
        if m or text == "Purpose":
            cur = {"section": int(m.group(1)) if m else 0, "title": m.group(2) if m else text,
                   "body": [], "pages": {page_no}}
            chunks.append(cur)
        elif cur is not None:
            cur["body"].append(text)
            cur["pages"].add(page_no)
    out = []
    for c in chunks:
        if "example question" in c["title"].lower():  # not policy content
            continue
        pages = sorted(c["pages"])
        out.append({"section": c["section"], "title": c["title"], "text": "\n".join(c["body"]),
                    "pages": pages, "ref": f"§{c['section']} {c['title']} ({_pages(pages)})"})
    if not out:
        raise DataError(f"No policy sections could be extracted from {path.name}")
    return out


def policy_digest(chunks: list[dict]) -> str:
    return hashlib.sha256(json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_policy_embeddings(chunks: list[dict], digest: str, chroma_dir: Path | str = CHROMA_DIR,
                            model: str = EMBED_MODEL) -> None:
    """Embed each section and persist it to Chroma (used by ingest.py). Replaces any previous collection."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first run: nothing to delete
    col = client.create_collection(COLLECTION, metadata={"policy_hash": digest, "embed_model": model, "hnsw:space": "cosine"})
    docs = [f"{c['title']}\n{c['text']}" for c in chunks]
    col.add(ids=[f"s{c['section']}" for c in chunks], documents=docs,
            embeddings=SentenceTransformer(model).encode(docs).tolist(),
            metadatas=[{"section": c["section"], "title": c["title"], "pages": ",".join(map(str, c["pages"]))} for c in chunks])


class PolicyIndex:
    """Reads the ingested policy (data/policy.json) and retrieves sections for a question.

    Retrieval queries the persisted Chroma collection (built by ingest.py) when available, otherwise a
    built-in BM25-style keyword ranker over the same sections, so the app and tests work without a model.
    """

    def __init__(self, store_dir: Path | str = STORE_DIR, embeddings: bool | None = None, chroma_dir: Path | str = CHROMA_DIR):
        f = Path(store_dir) / "policy.json"
        if not f.exists():
            raise DataError(f"{f} not found. {INGEST_HINT}")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            self.chunks = data["chunks"]
            self.digest, self.source = data["hash"], data.get("source", {})
            self.rules = RefundRules(**data["rules"]) if data.get("rules") else None
            if not self.chunks or any(not {"section", "title", "text", "pages", "ref"} <= c.keys() for c in self.chunks):
                raise ValueError("empty or malformed chunks")
        except (ValueError, KeyError, TypeError) as exc:
            raise DataError(f"{f.name} is invalid ({exc}). {INGEST_HINT}") from exc
        self.full_text = "\n".join(c["text"] for c in self.chunks)
        self._toks = [_tokens(c["title"]) * 3 + _tokens(c["text"]) for c in self.chunks]
        n = len(self._toks)
        df: dict[str, int] = {}
        for toks in self._toks:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._avg = sum(len(t) for t in self._toks) / n
        self.backend = "keyword"
        mode = (os.getenv("USE_EMBEDDINGS", "auto") if embeddings is None else str(embeddings)).lower()
        if mode != "false":
            self._open_chroma(chroma_dir, strict=mode == "true")

    def _open_chroma(self, chroma_dir: Path | str, strict: bool) -> None:
        """Open the persisted collection (never builds it) and check it matches data/policy.json."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            col = chromadb.PersistentClient(path=str(chroma_dir)).get_collection(COLLECTION)
            meta = col.metadata or {}
            if meta.get("policy_hash") != self.digest:
                raise DataError("the Chroma index does not match data/policy.json (stale). " + INGEST_HINT)
            self._embedder = SentenceTransformer(meta.get("embed_model", EMBED_MODEL))  # same model as ingest
            self._col, self.backend = col, "chroma"
        except Exception as exc:
            if strict:
                raise
            log.warning("Chroma index unavailable (%s); using keyword retrieval. %s", exc, INGEST_HINT)

    # -- retrieval
    def section(self, n: int) -> dict | None:
        return next((c for c in self.chunks if c["section"] == n), None)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        if self.backend == "chroma":
            try:
                emb = self._embedder.encode([query]).tolist()
                res = self._col.query(query_embeddings=emb, n_results=min(k, len(self.chunks)))
                by_id = {f"s{c['section']}": c for c in self.chunks}
                hits = [dict(by_id[i], score=round(1 - d, 3)) for i, d in zip(res["ids"][0], res["distances"][0]) if i in by_id]
                return [h for h in hits if h["score"] > 0.15]
            except Exception as exc:
                log.warning("Chroma query failed (%s); falling back to keyword retrieval", exc)
        return self._keyword(query, k)

    def _keyword(self, query: str, k: int) -> list[dict]:
        weights: dict[str, float] = {}
        for t in _tokens(query):
            weights[t] = max(weights.get(t, 0), 1.0)
            for extra in EXPAND.get(t, []):
                weights[_stem(extra)] = max(weights.get(_stem(extra), 0), 0.5)
        scored = []
        for chunk, toks in zip(self.chunks, self._toks):
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = sum(
                w * self._idf.get(t, 0) * (tf[t] * 2.2) / (tf[t] + 1.2 * (0.25 + 0.75 * len(toks) / self._avg))
                for t, w in weights.items() if t in tf
            )
            if score > 0:
                scored.append(dict(chunk, score=round(score, 3)))
        return sorted(scored, key=lambda c: -c["score"])[:k]


# ----------------------------------------------------------------------- refund rules


class RefundRules(BaseModel):
    """Thresholds read from the policy text itself, so they cannot drift from the PDF."""

    window_days: int
    auto_reject_days: int
    damage_report_days: int
    manager_approval_above: float

    @classmethod
    def from_policy(cls, text: str) -> "RefundRules | None":
        found = [
            re.search(r"within\s+(\d+)\s+calendar\s+days", text, re.I),
            re.search(r"older\s+than\s+(\d+)\s+days", text, re.I),
            re.search(r"within\s+(\d+)\s+days\s+of\s+delivery", text, re.I),
            re.search(r"above\s+\$?([\d,]+)\s+require\s+manager\s+approval", text, re.I),
        ]
        if not all(found):
            return None
        a, b, c, d = (m.group(1).replace(",", "") for m in found)
        return cls(window_days=int(a), auto_reject_days=int(b), damage_report_days=int(c), manager_approval_above=float(d))


Status = Literal["eligible", "ineligible", "insufficient_evidence", "requires_clarification"]


class Check(BaseModel):
    name: str
    result: Literal["pass", "fail", "unknown"]
    detail: str
    policy_ref: str | None = None


class RefundAssessment(BaseModel):
    status: Status
    order_id: str | None = None
    summary: str
    checks: list[Check] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    needs_manager_approval: bool = False


# Section 2 categories. Keyword in product name/category -> (status impact, explanation).
RESTRICTED = {
    "digital": ("clarify", "Digital products are non-refundable after activation or download."),
    "gift card": ("fail", "Gift cards and store credits are non-refundable."),
    "store credit": ("fail", "Gift cards and store credits are non-refundable."),
    "personalized": ("fail", "Personalized or custom-made products are non-refundable."),
    "custom": ("fail", "Personalized or custom-made products are non-refundable."),
    "hygiene": ("clarify", "Hygiene products are non-refundable after opening."),
}
DEAD_STATUSES = {"cancelled", "canceled", "returned", "refunded"}


def assess_refund(order: dict | None, order_id: str | None, policy: list[dict], rules: RefundRules | None,
                  today: date, issue: str | None = None) -> RefundAssessment:
    """Deterministic refund assessment. Never returns `eligible` without passing every verifiable check."""
    if not order_id:
        return RefundAssessment(status="requires_clarification", summary="No order ID was given, so I cannot assess a refund.",
                                questions=["Which order ID (e.g. O005) do you want checked?"])
    if not order:
        return RefundAssessment(status="insufficient_evidence", order_id=order_id,
                                summary=f"Order {order_id} does not exist in the order data, so no assessment is possible.",
                                questions=["Please double-check the order ID."])
    if not policy or rules is None:
        return RefundAssessment(status="insufficient_evidence", order_id=order_id,
                                summary="The refund policy could not be retrieved or its thresholds could not be read, so no assessment is possible.")

    ref = lambda n: (c["ref"] if (c := next((c for c in policy if c["section"] == n), None)) else f"§{n}")
    checks: list[Check] = []
    caveats: list[str] = []
    questions: list[str] = []
    missing = False

    # 1. Delivery status
    status = (order["Status"] or "").strip()
    if status.lower() == "delivered":
        checks.append(Check(name="Delivery status", result="pass", detail="Order is marked Delivered.", policy_ref=ref(1)))
    elif status.lower() in DEAD_STATUSES:
        checks.append(Check(name="Delivery status", result="fail", policy_ref=ref(1),
                            detail=f"Order status is '{status}', so there is no delivered purchase to refund."))
    else:
        missing = True
        checks.append(Check(name="Delivery status", result="unknown", policy_ref=ref(1),
                            detail=f"Order status is '{status or 'missing'}', not Delivered; the {rules.window_days}-day window starts at delivery."))

    # 2. Order age
    odate = date.fromisoformat(order["OrderDate"]) if order["OrderDate"] else None
    if odate is None:
        missing = True
        checks.append(Check(name="Return window", result="unknown", detail="Order date is missing or invalid.", policy_ref=ref(1)))
    elif (age := (today - odate).days) < 0:
        missing = True
        checks.append(Check(name="Return window", result="unknown", policy_ref=ref(1),
                            detail=f"Order date {odate} is after the reference date {today}; the data looks invalid."))
    elif age > rules.auto_reject_days:
        checks.append(Check(name="Return window", result="fail", policy_ref=ref(8),
                            detail=f"Order is {age} days old (placed {odate}); orders older than {rules.auto_reject_days} days are automatically rejected."))
    else:
        checks.append(Check(name="Return window", result="pass", policy_ref=ref(1),
                            detail=f"Order is {age} days old, within the {rules.window_days}-day limit."))
        caveats.append("The workbook has no delivery date, so the order date was used; the policy counts from delivery.")
        if issue == "damaged" and age > rules.damage_report_days:
            caveats.append(f"Damaged/incorrect items must be reported within {rules.damage_report_days} days of delivery ({ref(3)}); "
                           "this may already have passed.")

    # 3. Product restrictions (section 2)
    product = order.get("product")
    text = f"{(product or {}).get('ProductName', '')} {(product or {}).get('Category', '')}".lower()
    hit = next(((kind, why) for key, (kind, why) in RESTRICTED.items() if key in text), None)
    if product is None:
        missing = True
        checks.append(Check(name="Product restrictions", result="unknown", detail="Product details are missing, so section 2 cannot be checked.", policy_ref=ref(2)))
    elif hit and hit[0] == "fail":
        checks.append(Check(name="Product restrictions", result="fail", detail=hit[1], policy_ref=ref(2)))
    elif hit:
        questions.append(f"{hit[1]} Has the product been activated/opened?")
        checks.append(Check(name="Product restrictions", result="unknown", detail=hit[1], policy_ref=ref(2)))
    else:
        checks.append(Check(name="Product restrictions", result="pass", policy_ref=ref(2),
                            detail=f"Category '{product['Category']}' is not a non-refundable category."))

    # 4. Manager approval and data quality
    needs_manager = order["Amount"] is not None and order["Amount"] > rules.manager_approval_above
    if needs_manager:
        caveats.append(f"Order amount {order['Amount']:g} exceeds {rules.manager_approval_above:,.0f}: manager approval is required ({ref(8)}).")
    for msg in order["issues"]:
        caveats.append(f"Data quality: {msg}.")
    caveats.append("Not verifiable from the data: item unused/original condition, packaging, proof of purchase and Final Sale status "
                   f"({ref(1)}). These must be confirmed when the return is submitted.")

    if any(c.result == "fail" for c in checks):
        why = "; ".join(c.detail for c in checks if c.result == "fail")
        return RefundAssessment(status="ineligible", order_id=order_id, summary=why, checks=checks, caveats=caveats)
    if questions:
        return RefundAssessment(status="requires_clarification", order_id=order_id, checks=checks, caveats=caveats, questions=questions,
                                summary="More information is needed before eligibility can be decided.")
    if missing:
        return RefundAssessment(status="insufficient_evidence", order_id=order_id, checks=checks, caveats=caveats,
                                summary="The order data is incomplete, so eligibility cannot be decided.")
    return RefundAssessment(status="eligible", order_id=order_id, checks=checks, caveats=caveats, needs_manager_approval=needs_manager,
                            summary="Eligible on every check the order data can verify; item-condition requirements still apply.")


# ----------------------------------------------------------------------------- LLM


class LLMError(Exception):
    pass


class LLM:
    """Thin Ollama chat client (local, no API key)."""

    def __init__(self, host: str | None = None, model: str | None = None, timeout: float | None = None):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT", "120"))

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            return r.ok and any(m.get("name", "").split(":")[0] == self.model.split(":")[0] and
                                (":" not in self.model or m["name"] == self.model) for m in r.json().get("models", []))
        except Exception:
            return False

    def chat(self, messages: list[dict], json_mode: bool = False, temperature: float = 0.1) -> str:
        body = {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=(3, self.timeout))
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc


# ------------------------------------------------------------------------- container


@dataclass
class Services:
    store: DataStore
    policy: PolicyIndex
    rules: RefundRules | None
    llm: LLM | None
    today: date


def load_manifest(store_dir: Path | str = STORE_DIR) -> dict:
    try:
        return json.loads((Path(store_dir) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_services(use_llm: bool = True) -> Services:
    """Open the static store (JSON + Chroma). Raises DataError if it is missing or invalid."""
    store, policy = DataStore(), PolicyIndex()
    if policy.rules is None:
        log.warning("The ingested policy has no refund thresholds; refund assessments will be 'insufficient_evidence'")
    ref = os.getenv("REFERENCE_DATE", "").strip()
    try:
        today = datetime.strptime(ref, "%Y-%m-%d").date() if ref else date.today()
    except ValueError:
        log.warning("Invalid REFERENCE_DATE %r (expected YYYY-MM-DD); using today", ref)
        today = date.today()
    return Services(store, policy, policy.rules, LLM() if use_llm else None, today)
