# LLM-SHIELD — A Practical Security Architecture for RAG-Based LLM Systems

This repository is the reference implementation and benchmark harness for the paper:

*Anthony Mora, Jayden Alonzo-Estrada, Ram Basnet. **LLM-SHIELD: A Practical Security Architecture for RAG-Based LLM Systems**.*

LLM-SHIELD is a deployable, defense-in-depth architecture for retrieval-augmented LLM assistants that combines a pre-generation input classifier, retrieval isolation, identity-based access control, and multi-stage PII redaction. It is designed to be practical for small and mid-sized organizations on consumer-grade hardware. The code here reproduces every number reported in the paper.

---

## Repository layout

```
Specialized-AI-Agents/
├── requirements.txt           — pinned dependencies
├── analysis/
│   ├── fp_sweep.py            — classifier-only false-positive sweep
│   └── analyze_results.py     — aggregated result tables
├── results/                   — all benchmark output CSVs
│   ├── run_manifest.json      — seed, token cap, versions, prompt-set digests
│   ├── customer-facing/
│   │   ├── Qwen3-8B/
│   │   │   ├── unprotected/
│   │   │   ├── protected_pg86m_c512_temp1.0_th0.40/
│   │   │   │   ├── 100_prompt.csv
│   │   │   │   ├── benign_100_prompt.csv   ← false positive results
│   │   │   │   ├── guardrail_summary.csv
│   │   │   │   └── guardrail_failures.csv
│   │   │   └── ... (one directory per variant × guard config)
│   │   └── Meta-Llama-3.1-8B-Instruct/
│   │       └── ...
│   └── employee-facing/
│       └── ...
└── src/
    ├── main.py                    — master benchmark runner (all 100 configurations)
    ├── shared/
    │   ├── prompt_guard.py        — PromptGuard implementation (shared)
    │   ├── guardrail_analysis.py  — post-benchmark guardrail + false positive reports
    │   ├── attack_prompt.txt      — 78 public attack prompts
    │   ├── attack_prompt_heldout.txt — 25 held-out attack prompts
    │   ├── benign_prompts_fp_sweep.csv — 214-prompt false-positive corpus
    │   └── benignPrompts.txt      — 49 benign prompts for false positive testing
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
python src/main.py
```

Runs **100 configurations** — both environments × both models × (1 unprotected + 2
classifier variants × 12 guard configs):

| Axis | Values |
|------|--------|
| Side | `customer-facing`, `employee-facing` |
| Model | `Qwen3-8B`, `Meta-Llama-3.1-8B-Instruct` |
| Classifier | `pg86m` (`Llama-Prompt-Guard-2-86M`), `pg22m` (`Llama-Prompt-Guard-2-22M`) |
| Mode | `unprotected` (×1), `protected_<variant>_<name>` (×12 guard configs each) |

That is 25 configurations per (side, model) pair. Result directories are named
`protected_<variant>_<config>`.

Each configuration runs all four system prompt variants (`100_prompt`, `500_prompt`,
`1000_prompt`, `1500_prompt`) against three prompt sets:

| Set | File | Count | Output |
|-----|------|-------|--------|
| Public attacks | `src/shared/attack_prompt.txt` | 78 | `<prompt_stem>.csv` |
| Held-out attacks | `src/shared/attack_prompt_heldout.txt` | 25 | `heldout_<prompt_stem>.csv` |
| Benign | `src/shared/benignPrompts.txt` | 49 | `benign_<prompt_stem>.csv` |

The held-out slice is scored separately from the 78 public templates. Every
configuration runs the benign set, the unprotected baseline included.

Total: 100 × 4 × (78 + 25 + 49) = **60,800 records**.

Each worker subprocess tags every output line with its side and model so interleaved output from parallel workers is readable:
```
[C/Qwen3-8B] [100_prompt] attack 34/78 (44%)  blocked: 28  passed: 6  errors: 0
[E/Qwen3-8B] [100_prompt] benign 21/49 (43%)  fp: 0  passed: 21
```
`C` = customer-facing, `E` = employee-facing.

CSV output per prompt file: `results/<side>/<model>/<mode>/<prompt_stem>.csv`

Columns: `system_prompt_file, question, response, duration_seconds`

After all 100 configurations complete, `guardrail_analysis.py` automatically scans each `protected_*/` results directory and writes:
- `guardrail_summary.csv` — block rate and false positive rate per system prompt length
- `guardrail_failures.csv` — every attack prompt that was not blocked

---

## Guardrail summary columns

| Column | Description |
|--------|-------------|
| `system_prompt_file` | System prompt variant (100, 500, 1000, 1500) |
| `positive_count` | Attack prompts correctly blocked (true positives) |
| `total_questions` | Total attack prompts run |
| `percentage` | True positive block rate |
| `false_positive_count` | Benign prompts incorrectly blocked |
| `total_benign` | Total benign prompts run |
| `false_positive_rate` | False positive rate |

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

Each protected run produces its own results directory: `protected_<variant>_<name>/`.

---

## Protected vs unprotected

| Flag | Protected | Unprotected |
|------|-----------|-------------|
| Prompt Guard — input screening | On | Off |
| Prompt Guard — retrieved doc screening | On | Off |
| PII redaction | On | Off |
| Benign prompt false positive run | Yes | Yes |
| Max LLM output tokens | 1024 | 1024 |

> Both arms share one generation budget, `BENCHMARK_MAX_NEW_TOKENS = 1024`, so
> protected and unprotected latencies are directly comparable.

---

## Runtime flow

Both environments share the same pipeline:

1. **Model prefetch** — `main.py` downloads all GGUF models, Prompt Guard, and the embedding model to the HuggingFace cache before any workers start.
2. **Parallel workers** — configs are grouped by model. For each model, two worker subprocesses are launched in parallel: one per side (customer-facing, employee-facing). Each worker loads the LLM **once** and reuses it across all 25 configs for that (side, model) pair.
3. For each config, `app.setup()` builds or loads the FAISS index and initialises Prompt Guard. The LLM is injected from the shared worker instance — it is never reloaded.
4. `app.rebuild_chain()` hot-swaps the system prompt between the four `*_prompt.txt` variants without reloading the LLM or FAISS index.
5. `app.login()` authenticates with Supabase. Login failure is non-fatal — RAG queries continue without DB access.
6. Each attack prompt is sent to `app.ask()`:
   - Prompt Guard screens the input; if it fires, a refusal is returned immediately (fail-closed).
   - FAISS retrieves the top-k relevant document chunks.
   - If the question matches financial keywords and the user is authenticated, `renter_transactions` rows are fetched.
   - The LLM generates a response from the retrieved context.
   - If `REDACT_PII` is enabled, the output is sanitised before returning.
7. Results are appended to `<prompt_stem>.csv`.
8. For protected configs, each benign prompt is then run through the same `app` instance and results appended to `benign_<prompt_stem>.csv`.

**Employee-facing additions:**

- `app.login()` uses `admin_client.auth.sign_in_with_password()` (Supabase SDK) and creates a user-scoped `user_client` for storage operations.
- `app.download_user_files()` lists and downloads files from the user's prefix in Supabase Storage, preserving full relative paths to avoid filename collisions.

---

## Configuration

Key flags in `rag.py → CONFIG` (overridden per run by `main.py`):

| Key | customer-facing default | employee-facing default | Description |
|-----|------------------------|------------------------|-------------|
| `DATA_PATHS` | `["./docs"]` | `["./downloads"]` | Directories scanned for source documents |
| `SKIP_FILENAMES` | see below | see below | Filenames excluded from RAG indexing |
| `REBUILD_INDEX` | `False` | `True` | Force a fresh FAISS build |
| `LLM_NAME` | `"Qwen3-8B"` | `"Qwen3-8B"` | `"Qwen3-8B"` or `"Meta-Llama-3.1-8B-Instruct"` |
| `CTX_TOKENS` | `8192` | `8192` | LLM context window size |
| `MAX_NEW_TOKENS` | `1024` | `1024` | Max LLM output tokens (benchmark applies 1024 to both arms) |
| `N_BATCH` | `512` | `512` | GPU prompt-processing batch size; increase to 1024+ for higher throughput |
| `GPU_OFFLOAD` | `True` | `True` | Offload all LLM layers to GPU (`n_gpu_layers=-1`) |
| `PROMPT_GUARD_ENABLED` | `True` | `True` | Toggle Llama Prompt Guard input screening |
| `PROMPT_GUARD_THRESHOLD` | `0.40` | `0.40` | Malicious-probability cutoff (0–1) |
| `PROMPT_GUARD_MAX_TOKENS` | `512` | `512` | Max tokens per chunk fed to Prompt Guard |
| `PROMPT_GUARD_CHUNK_OVERLAP` | `50` | `50` | Token overlap between adjacent chunks |
| `PROMPT_GUARD_TEMPERATURE` | `1.0` | `1.0` | Softmax temperature applied to guard logits |
| `REDACT_PII` | `False` | `False` | Mask credit cards, SSNs, and emails in output |

`SKIP_FILENAMES` always includes: `attack_prompt.txt`, `benignPrompts.txt`, `malicious prompt.txt`, and all four `*_prompt.txt` system prompt files.

---

## Security architecture

| Layer | Mechanism | Detail |
|-------|-----------|--------|
| Input screening | Prompt Guard | `meta-llama/Llama-Prompt-Guard-2-86M` classifies input; requests above threshold are rejected fail-closed |
| Retrieved doc screening | Prompt Guard | Same model applied to retrieved chunks before passing to LLM (protected mode only) |
| PII redaction | Regex-based | Masks credit cards (keep last 4), SSNs, and email local parts |
| Access control | Supabase RLS | DB queries scoped to `user_id=eq.<authenticated_user>` |
| Anon key separation | Header split | User REST calls use `SUPABASE_ANON_KEY` as `apikey`; service role key is not sent to the client |
| User-scoped storage | SDK client | Employee file downloads use a `user_client` scoped to the user JWT, not the admin client |
| Email sanitisation | Regex validation | User email is validated by regex before injection into the system prompt |
| Document filtering | `SKIP_FILENAMES` | Benchmark infrastructure files excluded from RAG indexing |
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

HF_TOKEN=hf_...                       # required — both Prompt Guard 2 variants are gated models
```

> Get a token at https://huggingface.co/settings/tokens and accept the model licence for **both** `meta-llama/Llama-Prompt-Guard-2-86M` and `meta-llama/Llama-Prompt-Guard-2-22M` before running — the sweep loads both.

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

All dependencies are pinned in `requirements.txt` to the versions the published
results were produced with (the same values are recorded in
`results/run_manifest.json`, so a replication can diff the two):

```bash
pip install -r requirements.txt
```

`llama-cpp-python` needs a CUDA build; a plain install is CPU-only and `GPU_OFFLOAD`
then offloads nothing without raising an error:

```bash
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" \
  pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python \
  llama-cpp-python==0.3.20
```

`CMAKE_CUDA_ARCHITECTURES` is the GPU's compute capability without the dot — `89` for
an RTX 4090, `86` for a 3090, `90` for an H100. The build needs cuBLAS headers
(`libcublas-dev-<major>-<minor>`). Verify it reached the GPU:

```bash
python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

---

## Datasets

| Role | Dataset | Source |
|------|---------|--------|
| Attack prompts (`attack_prompt.txt`) | ChatGPT Jailbreak Prompts | [rubend18/ChatGPT-Jailbreak-Prompts](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts) |
| Benign prompts (`benignPrompts.txt`) | Awesome ChatTensor Prompts | [NeuralInternet/awesome-chattensor-prompts](https://huggingface.co/datasets/NeuralInternet/awesome-chattensor-prompts) |

Both datasets are sourced from Hugging Face and are excluded from version control via `.gitignore`.

---

## Reproducing the paper results

The 60,800-row results tree in `results/` is regenerated end-to-end by:

```bash
python analysis/fp_sweep.py              # classifier-only false-positive sweep (~5 min, no LLM)
python src/main.py                       # runs all 100 (side × model × classifier × config) combinations
python analysis/analyze_results.py       # rebuilds the per-config / aggregated / winners tables
```

`fp_sweep.py` loads only the classifier and sweeps every guard configuration, for
both variants, against the 214-prompt corpus in
`src/shared/benign_prompts_fp_sweep.csv`. It needs no LLM and no Supabase.

Sampling is live (`TEMPERATURE` 0.2), so runs are pinned by `BENCHMARK_SEED = 7979`;
set it to `-1` to sample freely. Each run writes `results/run_manifest.json` recording
the seed, token cap, temperature, `n_ctx`, `n_batch`, `n_gpu_layers`, the GPU model,
library versions, the configuration matrix, and a SHA-256 digest of each prompt set.

---

*For the full security analysis, methodology, and discussion, see the paper.*
