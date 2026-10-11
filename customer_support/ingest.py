"""Build the static data store the app reads at runtime. Run once, and again whenever the sources change.

    python ingest.py                  # Excel -> data/*.json, PDF -> data/policy.json + Chroma (chroma_db/)
    python ingest.py --no-embeddings  # skip Chroma (keyword retrieval will be used)
"""
import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from services import (CHROMA_DIR, EMBED_MODEL, POLICY_PDF, SCHEMA, STORE_DIR, WORKBOOK, DataError, RefundRules,
                      build_policy_embeddings, parse_policy_pdf, policy_digest, read_workbook)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ingest(workbook=WORKBOOK, pdf=POLICY_PDF, out=STORE_DIR, chroma=CHROMA_DIR, embeddings: bool = True) -> dict:
    """Validate the sources, write the static store, and return a summary. Raises DataError on bad input."""
    workbook, pdf, out = Path(workbook), Path(pdf), Path(out)
    sheets = read_workbook(workbook)          # validates sheets/columns before anything is written
    chunks = parse_policy_pdf(pdf)
    rules = RefundRules.from_policy("\n".join(c["text"] for c in chunks))
    digest = policy_digest(chunks)

    out.mkdir(parents=True, exist_ok=True)
    for sheet in SCHEMA:
        _write(out / f"{sheet.lower()}.json", sheets[sheet])
    _write(out / "policy.json", {"source": {"file": pdf.name, "sha256": _sha(pdf)}, "hash": digest,
                                 "rules": rules.model_dump() if rules else None, "chunks": chunks})
    embedded = None
    if embeddings:
        try:
            build_policy_embeddings(chunks, digest, chroma, EMBED_MODEL)
            embedded = EMBED_MODEL
        except Exception as exc:  # packages missing (e.g. Python 3.15) or model download failed
            logging.getLogger("support").warning("Chroma index not built (%s). Keyword retrieval will be used.", exc)
    manifest = {"ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "workbook": {"file": workbook.name, "sha256": _sha(workbook), "rows": {s: len(r) for s, r in sheets.items()}},
                "policy": {"file": pdf.name, "sha256": _sha(pdf), "sections": len(chunks), "rules": rules.model_dump() if rules else None},
                "embeddings": embedded}
    _write(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workbook", default=WORKBOOK)
    ap.add_argument("--pdf", default=POLICY_PDF)
    ap.add_argument("--out", default=STORE_DIR, help="JSON store directory")
    ap.add_argument("--chroma", default=CHROMA_DIR, help="Chroma persist directory")
    ap.add_argument("--no-embeddings", action="store_true")
    a = ap.parse_args()
    try:
        m = ingest(a.workbook, a.pdf, a.out, a.chroma, not a.no_embeddings)
    except DataError as exc:
        raise SystemExit(f"Ingest failed: {exc}")
    print(f"Excel  -> {a.out}: " + ", ".join(f"{s} {n} rows" for s, n in m["workbook"]["rows"].items()))
    print(f"PDF    -> {a.out}/policy.json: {m['policy']['sections']} sections, rules {m['policy']['rules']}")
    print(f"Chroma -> {a.chroma}: " + (f"built with {m['embeddings']}" if m["embeddings"] else "NOT built (keyword retrieval will be used)"))
