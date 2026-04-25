# Employee-Facing RAG App

Internal staff RAG pipeline described in:

*Anthony Mora, Jayden Alonzo-Estrada, Ram Basnet. "A Practical Security Architecture for Company RAG‑LLM Systems"*

Employees query company documents retrieved from Supabase Storage. The pipeline supports file download, identity-based access control, PII redaction, and prompt-guard input screening.

---

## Directory layout

```
employee-facing/
├── main.py               — standalone benchmark (unprotected, Meta-Llama, all system prompts)
├── rag.py                — RAG pipeline, Supabase SDK session, file download helpers
├── prompt_guard.py       — re-exports PromptGuard from src/shared/
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
src/employee-facing/.faiss_index/
```

Downloaded user files land in:
```
src/employee-facing/downloads/
```

Benchmark results are written to the project-root results tree:
```
results/employee-facing/<model>/<protected|unprotected>/<prompt_stem>.csv
```

---

## Running standalone

```bash
cd src/employee-facing
python main.py
```

This runs every system prompt in `docs/system_prompts/` against all attack prompts in `src/shared/attack_prompt.txt`, unprotected, using Meta-Llama-3.1-8B-Instruct. Results are written to:
```
results/employee-facing/Meta-Llama-3.1-8B-Instruct/unprotected/
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
| `DATA_PATHS` | `["./downloads"]` | Directories scanned for source documents |
| `SKIP_FILENAMES` | `["attack_prompt.txt"]` | Filenames excluded from RAG indexing |
| `REBUILD_INDEX` | `True` | Force a fresh FAISS build (default True because downloads change) |
| `LLM_NAME` | `"Qwen3-8B"` | `"Qwen3-8B"` or `"Meta-Llama-3.1-8B-Instruct"` |
| `PROMPT_GUARD_ENABLED` | `True` | Toggle Llama Prompt Guard input screening |
| `PROMPT_GUARD_THRESHOLD` | `0.6` | Malicious-probability cutoff (0–1) |
| `REDACT_PII` | `False` | Mask credit cards, SSNs, and emails in output |
| `Bucket_Name` | `"Company files"` | Supabase Storage bucket for file downloads |
| `SUPABASE_ANON_KEY` | env | Used as `apikey` in user-scoped REST calls |

---

## Runtime flow

1. `app.login()` authenticates via `admin_client.auth.sign_in_with_password()` (Supabase SDK).
   - Stores `user_id` and `access_token`.
   - Creates a `user_client` scoped to the user's access token for storage operations.
2. `app.download_user_files()` lists and downloads files from the user's prefix in Supabase Storage.
   - Object paths are preserved as relative sub-paths to avoid filename collisions.
   - Files land in `downloads/`.
3. `app.setup()`:
   - Creates the `downloads/` directory if missing.
   - Loads or builds the FAISS index from `DATA_PATHS`.
   - Downloads the LLM and initialises the Prompt Guard.
4. Each question sent to `app.ask()`:
   - Prompt Guard screens the input; if it fails, a refusal is returned immediately (fail-closed).
   - FAISS retrieves the top-k relevant document chunks.
   - If the question matches financial keywords, the user's `renter_transactions` rows are fetched.
   - The LLM generates a response from the retrieved context.
   - If `REDACT_PII` is enabled, the output is sanitised before returning.
5. Results are appended to the CSV log.

---

## Security features

| Feature | Detail |
|---------|--------|
| Prompt Guard | Classifies input with `meta-llama/Llama-Prompt-Guard-2-22M`; blocks requests above threshold |
| Fail-closed guard | Guard exceptions return an error to the user instead of falling through |
| PII redaction | Masks credit card numbers (last 4 shown), SSNs, and email local parts |
| Supabase RLS | `query_user_transactions` filters by `user_id=eq.<authenticated_user>` |
| Anon key separation | User REST calls use `SUPABASE_ANON_KEY` as `apikey`; service key is not sent to the client |
| User-scoped storage | File downloads use `user_client` (scoped to user JWT), not the admin client |
| Email sanitisation | User email is validated by regex before injection into the system prompt |
| Document filtering | `SKIP_FILENAMES` prevents benchmark infrastructure files from being indexed |
| Path collision fix | Downloads preserve full relative paths so files with the same name in different folders do not overwrite each other |

---

## Supabase session internals

`SupabaseSession` holds two clients:

| Client | Key used | Purpose |
|--------|----------|---------|
| `admin_client` | `SUPABASE_KEY` (service role) | Authentication only (`sign_in_with_password`) |
| `user_client` | User access token | Storage listing and downloads |

REST queries (`query_user_transactions`) use the user's access token as `Authorization: Bearer` and `SUPABASE_ANON_KEY` as `apikey`, keeping the service role key off the wire for user-scoped operations.

---

## Differences from customer-facing

| Feature | Employee-facing | Customer-facing |
|---------|----------------|-----------------|
| Supabase auth | Supabase Python SDK | Raw HTTP (`requests`) |
| File downloads | `download_user_files()` via SDK | Not supported |
| Default data path | `./downloads` | `./docs` |
| Index rebuild default | `True` | `False` |
| `tiktoken` import | Required (top-level import) | Optional (falls back) |

---

*See the top-level `README.md` for the master benchmark and full architecture overview.*
