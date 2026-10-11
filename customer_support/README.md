# ABC Retail Customer Support Assistant

A local-first, multi-agent chatbot that answers questions from an Excel workbook (Customers, Orders, Products)
and a refund-policy PDF. Python does the exact lookups, joins and refund rules; a local LLM (Ollama, optional)
interprets the question and phrases the answer. No paid API keys.

What it does:
- Answers customer, order and product questions from the Excel data.
- Answers refund-policy questions from the PDF, with section and page references.
- Combines both to assess refund eligibility (`eligible`, `ineligible`, `insufficient_evidence`, `requires_clarification`).
- Remembers the conversation, so follow-ups like "What about the product?" or "Why?" work, and survive a restart.

## Contents

[Architecture](#architecture) · [How a question is answered](#how-a-question-is-answered) · [Refund assessment](#refund-assessment) ·
[Conversation memory](#conversation-memory) · [Design decisions](#design-decisions) · [Project layout](#project-layout) ·
[Setup](#setup-windows-powershell) · [Ingestion](#ingestion-one-time) · [Run](#run) · [Example questions](#example-questions) ·
[Tests](#tests) · [Configuration](#configuration) · [Known limitations](#known-limitations)

## Architecture

### System components

```mermaid
flowchart TB
    User([User]) --> UI

    subgraph Client
        UI["Streamlit chat UI<br/>session id kept in the URL"]
    end

    subgraph Backend["FastAPI backend"]
        API["GET /health<br/>POST /chat<br/>GET /sessions/ID"]
    end

    subgraph Graph["LangGraph workflow - one run per turn"]
        SUP["Supervisor<br/>intent, follow-ups, routing, answer"]
        CUST["Customer Agent"]
        ORD["Order/Product Agent"]
        POL["Policy Agent"]
        REF["Refund Assessment Agent<br/>deterministic rules"]
    end

    CP[("Checkpointer: SQLite<br/>.state/checkpoints.sqlite<br/>messages + memory per session")]

    subgraph Store["Static data store - built once by ingest.py"]
        JSON[("data/*.json<br/>customers, orders, products")]
        PJSON[("data/policy.json<br/>sections, pages, thresholds")]
        CHROMA[("chroma_db/<br/>policy embeddings")]
    end

    LLM["Ollama LLM - optional<br/>classify intent, phrase answer"]

    UI -->|HTTP| API
    API -->|"run_chat with thread_id = session id"| SUP
    SUP <-->|"load / save state"| CP
    SUP -.-> LLM
    CUST --> JSON
    ORD --> JSON
    POL --> CHROMA
    POL --> PJSON
    REF --> PJSON
```

| Layer | Technology | Role |
|---|---|---|
| UI | Streamlit | Chat, expandable evidence, conversation focus |
| API | FastAPI + Uvicorn | `/health`, `/chat`, `/sessions/{id}` |
| Orchestration | LangGraph + SQLite checkpointer | Multi-agent routing and per-session persistent state |
| Structured data | JSON files loaded with pandas | Exact ID lookups and joins |
| Policy retrieval | PyMuPDF (at ingest) + Chroma + sentence-transformers, keyword fallback | Section retrieval with page references |
| LLM | Ollama (default `qwen2.5:3b`) | Intent hints and answer wording only |
| Validation | Pydantic | Request/response and analysis schemas |

### Ingestion pipeline (offline, run once)

```mermaid
flowchart LR
    XLSX["Sample_Customer_Data.xlsx<br/>Customers, Orders, Products"] --> ING
    PDF["Detailed_Sample_Refund_Policy.pdf"] --> ING
    ING["python ingest.py<br/>validate schema<br/>parse sections and pages<br/>read refund thresholds<br/>embed sections"]
    ING --> J["data/customers.json<br/>data/orders.json<br/>data/products.json"]
    ING --> P["data/policy.json"]
    ING --> C["chroma_db/"]
    ING --> M["data/manifest.json<br/>source hashes"]
```

At runtime the agents read only this store. The source Excel and PDF are not needed and nothing is re-parsed per run.

## How a question is answered

```mermaid
flowchart TD
    IN(["User message + session id"]) --> LOAD["Checkpointer loads saved messages and memory"]
    LOAD --> SUP["Supervisor analyze<br/>1. extract IDs<br/>2. classify intent: LLM or keywords<br/>3. resolve follow-ups from memory<br/>4. build the plan"]
    SUP --> NEXT{"route_next<br/>planned agent left?"}
    NEXT -->|customer| CUST["Customer Agent"]
    NEXT -->|order_product| ORD["Order/Product Agent"]
    NEXT -->|policy| POL["Policy Agent"]
    NEXT -->|refund| REF["Refund Assessment Agent"]
    CUST --> NEXT
    ORD --> NEXT
    POL --> NEXT
    REF --> NEXT
    NEXT -->|none left| RESP["Supervisor respond<br/>LLM or template answer<br/>update memory"]
    RESP --> SAVE["Checkpointer saves messages and memory"]
    SAVE --> OUT(["Answer + sources + refund status"])
```

`route_next` is a LangGraph conditional edge that runs after the Supervisor and after every specialist. Only the
planned specialists run, always in the order customer → order/product → policy → refund:

| Question type | Agents that run |
|---|---|
| Customer profile | Customer |
| Orders / product details | Order/Product |
| Policy question | Policy |
| Refund eligibility | Order/Product → Policy → Refund Assessment |
| "Why?" / "explain" after an answer | none: the saved facts are restored from memory |
| Anything else | none (help text) |

| Agent | Responsibility |
|---|---|
| Supervisor | Extracts IDs with regex (the LLM can never invent one), classifies intent, resolves follow-ups, plans, then writes the final answer and updates memory |
| Customer | Profile lookup (email and phone masked) |
| Order/Product | Order history and product details, joined through the Orders → Products/Customers keys; a customer with exactly one order is resolved to it for refund questions |
| Policy | Retrieves the relevant policy sections with page references (sections 1, 2 and 8 are always included for refund questions) |
| Refund Assessment | Combines order facts and policy evidence with plain Python rules |

## Refund assessment

`assess_refund` in `services.py` is deterministic. Its thresholds (30 days, 7 days, $2,000) are read from the policy
text at ingest time and stored in `data/policy.json`, so they cannot drift from the PDF.

| Check | Result |
|---|---|
| Delivery status | `Delivered` passes. `Cancelled`, `Returned` or `Refunded` fails. Any other status (for example `Shipped`) is unknown. |
| Return window (§1, §8) | An order older than 30 days fails (automatic rejection). A missing or future date is unknown. |
| Product restrictions (§2) | Gift cards and personalized items fail. Digital and hygiene products raise a question ("activated / opened?"). |
| Manager approval (§8) | Amount above $2,000 sets `needs_manager_approval` and adds a note. It does not change the status. |
| Data quality | Mismatched amount, missing fields and so on are listed as notes. |

Status precedence: any failed check gives `ineligible`; otherwise any open question gives `requires_clarification`;
otherwise any unknown gives `insufficient_evidence`; otherwise `eligible`. An unknown order, a missing policy or
unreadable thresholds also give `insufficient_evidence`. `eligible` means "every check the data can verify passed":
item condition, packaging, proof of purchase and Final Sale status are not in the data and are always listed as
unverified. The LLM only writes prose around the verdict; the verdict header is added by code.

## Conversation memory

Two things are kept in the LangGraph state and persisted by the checkpointer under `thread_id = session_id`:

| State key | Contents |
|---|---|
| `messages` | The user/assistant transcript (a reducer appends; capped at the last 40) |
| `memory` | `focus` (customer/order/product in play, including ones derived by joins: order O005 pins customer C005 and product P001), `last_intents`, `options` (choices just listed), `last` and `last_refund` (facts of the previous answer and of the last refund verdict, for "why?") |

Everything else in the state (plan, route, retrieved rows, assessment, answer) is per-turn and reset by `run_chat`.

**How a follow-up is understood** (`agents/memory.py`, called by the Supervisor):

| User says (after discussing O005) | Resolved as |
|---|---|
| "What about the product?" / "What is its price?" | Product info for **P001** (the order's product) |
| "Who ordered it?" / "And their orders?" | Customer **C005** / C005's orders |
| "Is it eligible?" / "Is that order refundable?" | Refund assessment for **O005** |
| "What if it was damaged?" | Refund assessment for O005 with a damaged-item claim |
| "What about O003?" / "And Customer 7?" | The **same kind of request** as before, for the new entity |
| "the second one" / "the latest" (after a list of orders) | That order from the options just shown |
| "Why?" | The last refund verdict is re-explained from saved facts; no agent re-runs |

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Streamlit
    participant API as FastAPI
    participant G as LangGraph
    participant CP as Checkpointer

    U->>UI: What about the product?
    UI->>API: POST /chat with message and session_id
    API->>G: run_chat with thread_id = session_id
    G->>CP: load state - focus is order O005, customer C005, product P001
    CP-->>G: messages and memory
    G->>G: Supervisor binds "the product" to P001
    G->>G: Order/Product Agent reads data/products.json
    G->>G: respond writes the answer and updates memory
    G->>CP: save messages and memory
    G-->>API: answer, sources, context.resolved = product P001
    API-->>UI: JSON response
    UI-->>U: Product 1 details, marked "Using earlier context"
```

Design rules:
- **IDs never come from the LLM.** They come from the user's text or from the saved focus. With Ollama running, the model
  only classifies intent, and it is shown the focus and the last turns so it can read elliptical questions.
  Without Ollama, keyword rules do the same job (tests cover both).
- **No guessing.** "What about the product?" in a brand-new conversation asks which product, instead of listing everything
  or picking one. A product-only or policy-only turn does not drop the order focus; a different customer does.
- **Transparency.** Answers that used memory start with *"Using earlier context: product P001"*, and `/chat` returns
  `context.interpreted`, `context.resolved` and `context.focus`.
- **Sessions are isolated** by session ID. The UI keeps the ID in the URL (`?sid=...`), so a refresh restores the
  transcript from `GET /sessions/{id}`; "New conversation" starts a new ID.
- **Checkpointer.** `SqliteSaver` at `.state/checkpoints.sqlite` (`CHECKPOINT_DB` to move it, `:memory:` for none). On a Python
  without LangGraph the built-in runner keeps sessions in memory only. Conversations are not deleted automatically;
  delete the file to reset everything.

## Design decisions

- *Order date stands in for delivery date.* The workbook has no delivery column. Policy §8 rejects orders older
  than 30 days outright, so that rule is applied to the order date; inside the window the answer says the delivery date is unknown.
- *Sample amounts are inconsistent* (for example O005: 3 × 100 ≠ 600). This is reported as a data-quality note, never silently "fixed".
- *No membership field exists*, so "is C001 premium?" gets an honest "can't verify".
- *PII*: email and phone are masked in answers (`MASK_PII=false` to disable). PDF text is capped, wrapped in
  `<policy_text>` tags for the LLM, and treated as untrusted; business rules never come from model output.
- *Ambiguity*: two different order IDs in one question, a missing ID, or an unknown ID produce a clarification or
  "not found", not a guess. Follow-ups are resolved from the saved conversation focus (see above), and a reference with
  nothing to resolve to is asked about, not guessed.
- *Graceful degradation*: if Ollama is down, routing falls back to keywords and answers come from templates
  (flagged in `warnings`); a missing or invalid data store makes `/health` report `degraded` and `/chat` return 503.
- *Static store*: the Excel and PDF are converted once and committed as JSON plus a Chroma index, so runtime has no
  parsing step and the agents read exactly what you can inspect in `data/`.

## Project layout

```
customer_support/
  ingest.py           one-time: Excel -> data/*.json, PDF -> data/policy.json + chroma_db/
  data/               static store (committed): customers/orders/products.json, policy.json, manifest.json
  chroma_db/          persisted Chroma collection of policy-section embeddings (committed)
  app.py              FastAPI backend (/health, /chat, /sessions/{id})
  streamlit_app.py    Streamlit chat UI (calls the API over HTTP)
  services.py         DataStore (JSON), PolicyIndex (Chroma/keyword), refund rules, Ollama client
  agents/             supervisor.py, customer.py, order_product.py, policy.py, refund.py,
                      memory.py (follow-up resolution + memory update), graph.py (LangGraph wiring + checkpointer)
  .state/             SQLite checkpoint database (created at runtime, git-ignored)
  tests/              pytest (no LLM or network needed)
  requirements.txt, requirements-embeddings.txt, .env.example
```
The source Excel and PDF stay in the parent folder and are only read by `ingest.py`.

## Setup (Windows PowerShell)

Python **3.12 or 3.13** gives the full stack (LangGraph, Chroma, sentence-transformers). From the workspace root:

```powershell
uv venv --python 3.12 .venv312        # or: py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
cd customer_support
uv pip install -r requirements.txt -r requirements-embeddings.txt   # or: pip install -r ... -r ...
Copy-Item .env.example .env           # optional; defaults work
```

> **Python 3.14+** cannot install `chromadb`/`sentence-transformers`/`torch`, and `langgraph` fails to import (no
> `ormsgpack` wheel); `requirements.txt` skips those packages there via environment markers. The app still runs:
> `ingest.py` skips the Chroma build, retrieval uses a built-in keyword ranker over `data/policy.json`, and a built-in runner
> executes the same graph definition with in-memory sessions. `GET /health` shows what is active (`graph_backend`,
> `checkpointer`, `policy.retrieval`).

### Ollama (local LLM, optional)

Ollama is a separate program, not a Python package.

```powershell
winget install Ollama.Ollama          # or download from https://ollama.com
# open a NEW terminal so the PATH update is picked up, then:
ollama pull qwen2.5:3b                # ~2 GB; use qwen2.5:7b or llama3.2:3b if you prefer
ollama list                           # confirm the model is present
```
Set `OLLAMA_MODEL` in `.env` to the model you pulled. Ollama runs on `http://127.0.0.1:11434` by default.
The app works without it, with template answers; the sidebar and `/health` show whether the LLM is reachable.

## Ingestion (one time)

```powershell
python ingest.py                  # Excel -> data/*.json; PDF -> data/policy.json + chroma_db/
python ingest.py --no-embeddings  # skip Chroma (keyword retrieval)
```
This validates the workbook schema, parses the PDF into sections with page numbers, reads the refund thresholds
(30 days, 7 days, $2,000) from the policy text, embeds each section with `all-MiniLM-L6-v2` (first run downloads
~90 MB) and writes `data/manifest.json` with source-file hashes. The generated `data/` and `chroma_db/` are meant to
be committed. **Re-run it only when the Excel or PDF changes**, then restart the backend. If the store is missing or
invalid, `/health` reports `degraded` with a "run `python ingest.py`" message, and `/chat` returns 503.

## Run

Two terminals, both from `customer_support/` with the venv activated:

```powershell
# 1. Backend
uvicorn app:app --port 8000

# 2. UI
streamlit run streamlit_app.py
```
Open http://localhost:8501. API docs are at http://127.0.0.1:8000/docs. Endpoints: `GET /health`,
`POST /chat` (`{"message": "...", "session_id": "optional"}`; the reply includes the `session_id` to reuse),
`GET /sessions/{session_id}` (persisted messages + memory).

**Refund dates.** The sample orders are dated 2026-07-01 to 2026-07-10, so on a later date every order is past
the 30-day window (correctly *ineligible*). To demo an eligible refund, set a reference date before starting the backend:
```powershell
$env:REFERENCE_DATE = "2026-07-20"; uvicorn app:app --port 8000
```

## Example questions

Each sample customer C00N has exactly one order, O00N. Examples below assume `REFERENCE_DATE=2026-07-20`.

| Try | Expect |
|---|---|
| `Show customer C001` | Profile, masked email/phone, source `Customers` |
| `What orders did C002 place?` | Order list joined to product names |
| `Tell me about product P003` / `What products do you have?` | Product details / list |
| `How long does a refund take?` | §5 passage with page reference |
| `Can a digital product be refunded?` | §2 passage |
| `Who pays for return shipping?` | §4 passage |
| `Does a premium customer receive faster processing?` | §8 passage + note that the workbook has no tier field |
| `Is order O005 eligible for a refund?` | `eligible` (with the reference date) or `ineligible` (after 30 days), with checks and caveats |
| `Is order O999 eligible for a refund?` | `insufficient_evidence`: order not found |
| `Is my order eligible for a refund?` | `requires_clarification`: asks for an order ID |

A conversation that uses memory:

| Turn | You ask | Expect |
|---|---|---|
| 1 | `Does Customer 5 is eligible to get a refund?` | Customer 5 has one order, O005, so it is assessed directly |
| 2 | `What about the product?` | Product 1 / P001 details, marked "Using earlier context" |
| 3 | `Who ordered it?` | Customer C005's profile |
| 4 | `Why?` | The refund verdict re-explained with its checks |
| 5 | `What if it was damaged?` | O005 re-assessed with the 7-day damaged-item caveat |
| 6 | `What about O003?` | O003 (same kind of request as turn 5) |

Direct API calls (reuse the returned `session_id`):
```powershell
$r = Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType 'application/json' `
  -Body '{"message":"Is order O005 eligible for a refund?"}'
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType 'application/json' `
  -Body (@{message="What about the product?"; session_id=$r.session_id} | ConvertTo-Json)
```

## Tests

```powershell
python -m pytest -q
```
Expect about 83 passed and 1 skipped on Python 3.12. The tests cover:
- ingest and JSON store loading, including corrupt or missing files and a check that the committed `data/` matches the source files;
- policy retrieval, and refund rules (30-day boundary, manager approval, non-delivered or cancelled orders, restricted products, empty policy);
- agent routing and the end-to-end demo flows;
- conversation memory: follow-ups, "why?", ordinals, session isolation, and SQLite persistence across a simulated restart;
- LLM-failure fallback and the HTTP API.

No test needs Ollama or network access. The one skipped test builds and queries a real Chroma index; it runs only with
`RUN_EMBEDDING_TESTS=1` (needs the embeddings packages and a model download):
```powershell
$env:RUN_EMBEDDING_TESTS = "1"; python -m pytest -q
```

## Configuration

See `.env.example`: `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`, `STORE_DIR`, `CHROMA_DIR`, `WORKBOOK_PATH`, `POLICY_PDF_PATH` (ingest only),
`CHECKPOINT_DB`, `USE_EMBEDDINGS` (`auto`/`true`/`false`), `EMBED_MODEL`, `REFERENCE_DATE`, `MASK_PII`, `API_URL`.

## Known limitations

- The Ollama-backed path (LLM intent hints and phrased answers) is implemented but was **not exercised against a real model**
  in development: Ollama was not installed. The tests cover it with stand-in models; the template path is what the live checks cover.
- Chroma retrieval and the real LangGraph backend were verified on Python 3.12 (including a Chroma round-trip and SQLite
  persistence across a restart); on Python 3.14+ the app falls back as described in Setup.
- Follow-up understanding is rule-based plus an optional LLM intent hint: unusual phrasings with no reference word and no
  running LLM (for example "and the thing they bought?") are not understood. It does not resolve several entities at once or comparisons
  ("compare it with C003").
- There is no aggregation: totals, sums and comparisons ("how much did C010 spend?") are not supported.
- Item condition, packaging, proof of purchase and "Final Sale" are not in the data; they are reported as unverified, not assumed.
- Keyword retrieval (fallback only) uses a small synonym table; unusual phrasings may miss. Chroma embeddings handle paraphrases.
- `data/` holds the sample customers' emails/phones in plain JSON (committed). Fine for sample data; do not commit real PII.
