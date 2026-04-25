# Customer-Facing RAG App

External client-facing RAG pipeline described in:

*Anthony Mora, Jayden Alonzo-Estrada, Ram Basnet. "A Practical Security Architecture for Company RAG‑LLM Systems"*

Clients query company documents through a security-hardened pipeline. Prompt injection is blocked, PII can be redacted, and each client is scoped to their own data via Supabase row-level security.

---

## Directory layout

```
customer-facing/
├── main.py               — standalone benchmark (unprotected, Meta-Llama, all system prompts)
├── rag.py                — RAG pipeline, security guards, Supabase session
├── prompt_guard.py       — re-exports PromptGuard from src/shared/
├── testGuard.py          — quick smoke-test for the prompt guard
└── docs/
    ├── system_prompts/   — .txt files, one per token-length variant
    │   ├── 100_prompt.txt
    │   ├── 500_prompt.txt
    │   ├── 1000_prompt.txt
    │   └── 1500_prompt.txt
    └── EMAILS.csv        — sample company emails indexed by FAISS
```

Attack prompts are shared across both environments and live at:
```
src/shared/attack_prompt.txt
```

FAISS index is built automatically on first run and cached at:
```
src/customer-facing/.faiss_index/
```

Benchmark results are written to the project-root results tree:
```
results/customer-facing/<model>/<protected|unprotected>/<prompt_stem>.csv
```

---

## Running standalone

Run from this directory (or via the master `main.py` at the project root):

```bash
cd src/customer-facing
python main.py
```

This runs every system prompt in `docs/system_prompts/` against all attack prompts in `src/shared/attack_prompt.txt`, unprotected, using Meta-Llama-3.1-8B-Instruct. Results are written to:
```
results/customer-facing/Meta-Llama-3.1-8B-Instruct/unprotected/
```

To run all 8 configurations (both models, both guard modes) use the master benchmark:
```bash
cd <project-root>
python main.py
```

---

## Configuration

Key flags in `rag.py → CONFIG`:

| Key | Default | Description |
|-----|---------|-------------|
| `DATA_PATHS` | `["./docs"]` | Directories scanned for source documents |
| `SKIP_FILENAMES` | `["attack_prompt.txt"]` | Filenames excluded from RAG indexing |
| `REBUILD_INDEX` | `False` | Force a fresh FAISS build |
| `LLM_NAME` | `"Qwen3-8B"` | `"Qwen3-8B"` or `"Meta-Llama-3.1-8B-Instruct"` |
| `PROMPT_GUARD_ENABLED` | `True` | Toggle Llama Prompt Guard input screening |
| `PROMPT_GUARD_THRESHOLD` | `0.6` | Malicious-probability cutoff (0–1) |
| `REDACT_PII` | `False` | Mask credit cards, SSNs, and emails in output |
| `SUPABASE_ANON_KEY` | env | Used as `apikey` in user-scoped REST calls |

---

## Runtime flow

1. `main.py` iterates over each `system_prompts/*.txt` file.
2. For each system prompt a `RAGApp` is created with a fresh config copy (global `CONFIG` is never mutated).
3. `app.setup()` loads or builds the FAISS index, downloads the LLM, and initialises the Prompt Guard.
4. `app.login()` authenticates with Supabase via REST and stores the user JWT.
5. Each attack prompt is sent through `app.ask()`:
   - Prompt Guard screens the input; if it fails, a refusal is returned immediately (fail-closed).
   - FAISS retrieves the top-k relevant document chunks.
   - If the question matches financial keywords, the user's `renter_transactions` rows are fetched from Supabase.
   - The LLM generates a response from the retrieved context.
   - If `REDACT_PII` is enabled, the output is sanitised before returning.
6. Results are appended to the CSV log.

---

## Security features

| Feature | Detail |
|---------|--------|
| Prompt Guard | Classifies input with `meta-llama/Llama-Prompt-Guard-2-22M`; blocks requests above threshold |
| Fail-closed guard | Guard exceptions return an error to the user instead of falling through |
| PII redaction | Masks credit card numbers (last 4 shown), SSNs, and email local parts |
| Supabase RLS | `query_user_transactions` filters by `user_id=eq.<authenticated_user>` |
| Anon key separation | User REST calls use `SUPABASE_ANON_KEY` as `apikey`; service key is not sent to the client |
| Email sanitisation | User email is validated by regex before injection into the system prompt |
| Document filtering | `SKIP_FILENAMES` prevents benchmark infrastructure files from being indexed |

---

## Differences from employee-facing

| Feature | Customer-facing | Employee-facing |
|---------|----------------|-----------------|
| Supabase auth | Raw HTTP (`requests`) | Supabase Python SDK |
| File downloads | Not supported | `download_user_files()` via SDK |
| Default data path | `./docs` | `./downloads` |
| Index rebuild default | `False` | `True` |
| `tiktoken` import | Optional (falls back) | Required (imported at top) |

---

*See the top-level `README.md` for the master benchmark and full architecture overview.*
