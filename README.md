# Specialized AI Agents — Secure RAG Architecture

This repository implements a secure Retrieval-Augmented Generation (RAG) architecture designed for both internal staff and external clients, as described in:

*Anthony Mora, Jayden Alonzo-Estrada, Ram Basnet. "A Practical Security Architecture for Company RAG‑LLM Systems"*

---

## Repository layout

```
Specialized-AI-Agents/
├── main.py                    — master benchmark runner (all 52 configurations)
├── results/                   — all benchmark output CSVs
│   ├── customer-facing/
│   │   ├── Qwen3-8B/
│   │   │   ├── unprotected/
│   │   │   ├── protected_c512_temp1.0_th0.40/
│   │   │   ├── protected_c128_ov100_temp2.0_th0.40/
│   │   │   └── ... (one directory per guard config)
│   │   └── Meta-Llama-3.1-8B-Instruct/
│   │       └── ...
│   └── employee-facing/
│       └── ...
└── src/
    ├── shared/
    │   ├── prompt_guard.py        — PromptGuard implementation (shared)
    │   ├── guardrail_analysis.py  — post-benchmark guardrail summary + failure reports
    │   └── attack_prompt.txt      — adversarial prompts used by all benchmarks
    ├── customer-facing/
    │   ├── rag.py                 — RAG pipeline, security guards, Supabase session
    │   ├── prompt_guard.py        — re-exports PromptGuard from src/shared/
    │   └── docs/
    │       ├── system_prompts/    — 100/500/1000/1500_prompt.txt
    │       └── EMAILS.csv         — sample company emails indexed by FAISS
    └── employee-facing/
        ├── rag.py                 — RAG pipeline, Supabase SDK session, file download helpers
        ├── prompt_guard.py        — re-exports PromptGuard from src/shared/
        └── docs/
            ├── system_prompts/    — 100/500/1000/1500_prompt.txt
            └── EMAILS.csv         — sample company emails indexed by FAISS
```

---

## Environments

### 1. customer-facing

External clients query company documents through a security-hardened pipeline. Prompt injection is blocked, PII can be redacted, and each client is scoped to their own data via Supabase row-level security.

- **Audience:** External clients
- **Auth:** Raw HTTP to Supabase REST API
- **Data:** `docs/` directory (EMAILS.csv + company documents)
- **FAISS index:** `src/customer-facing/.faiss_index/`

### 2. employee-facing

Internal staff query company documents with broader access and Supabase Storage file download support. Identity-based access control, PII redaction, and prompt guard apply equally.

- **Audience:** Company employees
- **Auth:** Supabase Python SDK
- **Data:** `downloads/` directory (files fetched from Supabase Storage per user)
- **FAISS index:** `src/employee-facing/.faiss_index/`

---

## Running the master benchmark

```bash
python main.py
```

Runs **52 configurations** — both environments × both models × 1 unprotected + 12 guard configs:

| Axis | Values |
|------|--------|
| Side | `customer-facing`, `employee-facing` |
| Model | `Qwen3-8B`, `Meta-Llama-3.1-8B-Instruct` |
| Mode | `unprotected` (×1), `protected_<name>` (×12 guard configs) |

Each configuration runs all four system prompt variants (`100_prompt`, `500_prompt`, `1000_prompt`, `1500_prompt`) against every attack prompt in `src/shared/attack_prompt.txt`.

CSV output per prompt file: `results/<side>/<model>/<mode>/<prompt_stem>.csv`

Columns: `system_prompt_file, question, response, duration_seconds`

After all 52 configurations complete, `guardrail_analysis.py` automatically scans each `protected_*/` results directory and writes:
- `guardrail_summary.csv` — block rate per system prompt length
- `guardrail_failures.csv` — every prompt that was not blocked

---

## Guard config sweep

The benchmark sweeps 12 named guard configurations across three axes:

| Name | Chunk tokens | Overlap | Temperature | Threshold |
|------|-------------|---------|-------------|-----------|
| `c512_temp1.0_th0.40` | 512 | 50 | 1.0 | 0.40 |
| `c512_temp2.0_th0.40` | 512 | 50 | 2.0 | 0.40 |
| `c512_temp3.0_th0.40` | 512 | 50 | 3.0 | 0.40 |
| `c128_ov100_temp1.0_th0.40` | 128 | 100 | 1.0 | 0.40 |
| `c128_ov100_temp1.5_th0.40` | 128 | 100 | 1.5 | 0.40 |
| `c128_ov100_temp2.0_th0.40` | 128 | 100 | 2.0 | 0.40 |
| `c128_ov100_temp3.0_th0.40` | 128 | 100 | 3.0 | 0.40 |
| `c128_ov100_temp5.0_th0.40` | 128 | 100 | 5.0 | 0.40 |
| `c128_ov100_temp3.0_th0.50` | 128 | 100 | 3.0 | 0.50 |
| `c128_ov100_temp5.0_th0.50` | 128 | 100 | 5.0 | 0.50 |
| `c64_ov50_temp3.0_th0.40` | 64 | 50 | 3.0 | 0.40 |
| `c64_ov50_temp5.0_th0.40` | 64 | 50 | 5.0 | 0.40 |

**Chunk tokens** — maximum tokens per input chunk fed to Prompt Guard.  
**Overlap** — token overlap between adjacent chunks (catches attacks that span chunk boundaries).  
**Temperature** — softmax temperature applied to Prompt Guard logits; higher values push borderline-scoring inputs above the block threshold.  
**Threshold** — minimum malicious-probability score to block a request.

Each protected run produces its own results directory: `protected_<name>/`.

---

## Protected vs unprotected

| Flag | Protected | Unprotected |
|------|-----------|-------------|
| Prompt Guard — input screening | On | Off |
| Prompt Guard — retrieved doc screening | On | Off |
| PII redaction | On | Off |

---

## Runtime flow

Both environments share the same pipeline:

1. `main.py` loads each `system_prompts/*.txt` file and creates a fresh `RAGApp` per system prompt (global `CONFIG` is never mutated).
2. `app.setup()` loads or builds the FAISS index, downloads the LLM, and initialises Prompt Guard.
3. `app.login()` authenticates with Supabase and stores the user JWT.
4. Each attack prompt is sent to `app.ask()`:
   - Prompt Guard screens the input; if it fails, a refusal is returned immediately (fail-closed).
   - FAISS retrieves the top-k relevant document chunks.
   - If the question matches financial keywords, the user's `renter_transactions` rows are fetched.
   - The LLM generates a response from the retrieved context.
   - If `REDACT_PII` is enabled, the output is sanitised before returning.
5. Results are appended to the CSV log.

**Employee-facing additions:**

- `app.login()` uses `admin_client.auth.sign_in_with_password()` (Supabase SDK) and creates a user-scoped `user_client` for storage operations.
- `app.download_user_files()` lists and downloads files from the user's prefix in Supabase Storage, preserving full relative paths to avoid filename collisions.

---

## Configuration

Key flags in `rag.py → CONFIG` (overridden per run by `main.py`):

| Key | customer-facing default | employee-facing default | Description |
|-----|------------------------|------------------------|-------------|
| `DATA_PATHS` | `["./docs"]` | `["./downloads"]` | Directories scanned for source documents |
| `SKIP_FILENAMES` | `["attack_prompt.txt"]` | `["attack_prompt.txt"]` | Filenames excluded from RAG indexing |
| `REBUILD_INDEX` | `False` | `True` | Force a fresh FAISS build |
| `LLM_NAME` | `"Qwen3-8B"` | `"Qwen3-8B"` | `"Qwen3-8B"` or `"Meta-Llama-3.1-8B-Instruct"` |
| `PROMPT_GUARD_ENABLED` | `True` | `True` | Toggle Llama Prompt Guard input screening |
| `PROMPT_GUARD_THRESHOLD` | `0.40` | `0.40` | Malicious-probability cutoff (0–1) |
| `PROMPT_GUARD_MAX_TOKENS` | `512` | `512` | Max tokens per chunk fed to Prompt Guard |
| `PROMPT_GUARD_CHUNK_OVERLAP` | `50` | `50` | Token overlap between adjacent chunks |
| `PROMPT_GUARD_TEMPERATURE` | `1.0` | `1.0` | Softmax temperature applied to guard logits |
| `REDACT_PII` | `False` | `False` | Mask credit cards, SSNs, and emails in output |
| `SUPABASE_ANON_KEY` | env | env | Used as `apikey` in user-scoped REST calls |
| `Bucket_Name` | — | `"Company files"` | Supabase Storage bucket (employee-facing only) |

---

## Security architecture

| Layer | Mechanism | Detail |
|-------|-----------|--------|
| Input screening | Prompt Guard | `meta-llama/Llama-Prompt-Guard-2-22M` classifies input; requests above threshold are rejected fail-closed |
| Retrieved doc screening | Prompt Guard | Same model applied to retrieved chunks before passing to LLM (protected mode only) |
| PII redaction | Regex-based | Masks credit cards (keep last 4), SSNs, and email local parts |
| Access control | Supabase RLS | DB queries scoped to `user_id=eq.<authenticated_user>` |
| Anon key separation | Header split | User REST calls use `SUPABASE_ANON_KEY` as `apikey`; service role key is not sent to the client |
| User-scoped storage | SDK client | Employee file downloads use a `user_client` scoped to the user JWT, not the admin client |
| Email sanitisation | Regex validation | User email is validated by regex before injection into the system prompt |
| Document filtering | `SKIP_FILENAMES` | Benchmark infrastructure files (attack prompts, system prompt files) excluded from RAG indexing |
| Path collision fix | Relative paths | Downloads preserve full relative paths so files with the same name in different folders do not overwrite each other |

---

## Differences between environments

| Feature | Customer-facing | Employee-facing |
|---------|----------------|-----------------|
| Supabase auth | Raw HTTP (`requests`) | Supabase Python SDK |
| File downloads | Not supported | `download_user_files()` via SDK |
| Default data path | `./docs` | `./downloads` |
| Index rebuild default | `False` | `True` |
| `tiktoken` import | Optional (falls back to word count) | Required (top-level import) |

---

## Supabase session internals (employee-facing)

`SupabaseSession` holds two clients:

| Client | Key used | Purpose |
|--------|----------|---------|
| `admin_client` | `SUPABASE_KEY` (service role) | Authentication only (`sign_in_with_password`) |
| `user_client` | User access token | Storage listing and downloads |

REST queries (`query_user_transactions`) use the user's access token as `Authorization: Bearer` and `SUPABASE_ANON_KEY` as `apikey`, keeping the service role key off the wire for user-scoped operations.

---

## Environment variables

Create a `.env` file in the project root (never commit it):

```
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=<service-role-key>       # used server-side only
SUPABASE_ANON_KEY=<anon-key>          # used for user-scoped REST calls
SUPABASE_BUCKET=Company files         # employee-facing storage bucket
SUPABASE_EMAIL=user@example.com       # test account
SUPABASE_PASSWORD=yourpassword
```

Optional path overrides (default to the repository layout):

```
BENCHMARK_SRC_DIR=src
BENCHMARK_SHARED_DIR=src/shared
BENCHMARK_CUSTOMER_DIR=src/customer-facing
BENCHMARK_EMPLOYEE_DIR=src/employee-facing
BENCHMARK_RESULTS_DIR=results
```

---

## Dependencies

```bash
pip install langchain langchain-community langchain-core langchain-huggingface langchain-text-splitters
pip install faiss-cpu huggingface-hub tiktoken python-dotenv pandas requests
pip install transformers torch
pip install supabase          # employee-facing only
```

---

*For security analysis and evaluation methodology, see the full research paper.*
