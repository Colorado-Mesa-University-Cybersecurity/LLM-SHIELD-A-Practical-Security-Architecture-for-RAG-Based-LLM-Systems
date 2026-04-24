# Customer-Facing RAG App — Setup and Usage

This environment implements the external client-facing RAG (Retrieval-Augmented Generation) pipeline described in:

*Anthony Mora, Jayden Alonzo-Estrada, Ram Basnet. "A Practical Security Architecture for Company RAG‑LLM Systems"*

It is designed for external clients who query company documents. Security is strict: prompt injection is blocked, PII can be redacted, and clients are scoped to their own data.

---

## Quick start

1. Install dependencies:
   ```
   pip install langchain langchain-community langchain-core langchain-huggingface langchain-text-splitters
   pip install faiss-cpu huggingface-hub tiktoken python-dotenv pandas requests
   pip install transformers torch
   ```

2. Set environment variables (`.env` or export):
   ```
   SUPABASE_URL=https://xxxx.supabase.co
   SUPABASE_KEY=your-anon-or-service-key
   ```

3. Place documents in `docs/` and system prompt files in `docs/system_prompts/`.

4. Run the benchmark suite:
   ```
   python main.py
   ```
   This runs every system prompt in `SYSTEM_PROMPTS` against the attack prompts in `docs/attack_prompt.txt` and logs results to `results/unprotected/`.

---

## Directory layout

```
customer-facing/
├── main.py               — benchmark runner (loops system prompts × attack prompts)
├── rag.py                — RAG pipeline, security guards, Supabase session
├── prompt_guard.py       — re-exports PromptGuard from src/shared/
├── testGuard.py          — quick smoke-test for the prompt guard
├── docs/
│   ├── system_prompts/   — .txt files, one per token-length variant
│   ├── attack_prompt.txt — one adversarial prompt per line
│   └── <company docs>    — PDFs, DOCX, CSV files indexed by FAISS
├── .faiss_index/         — persisted vector index (auto-built)
└── results/
    ├── protected/        — results with PROMPT_GUARD_ENABLED=True
    └── unprotected/      — results with PROMPT_GUARD_ENABLED=False
```

---

## Configuration

Key flags in `rag.py → CONFIG`:

| Key | Default | Description |
|-----|---------|-------------|
| `DATA_PATHS` | `["./docs"]` | Directories scanned for source documents |
| `REBUILD_INDEX` | `False` | Set `True` to force a fresh FAISS build |
| `LLM_NAME` | `"Qwen3-8B"` | Model name (`"Qwen3-8B"` or `"Meta-Llama-3.1-8B-Instruct"`) |
| `PROMPT_GUARD_ENABLED` | `True` | Toggle Llama Prompt Guard input screening |
| `PROMPT_GUARD_THRESHOLD` | `0.6` | Malicious-probability cutoff (0–1) |
| `REDACT_PII` | `False` | Mask credit cards, SSNs, and emails in output |
| `TEST_EMAIL` / `TEST_PASSWORD` | hardcoded | Supabase test credentials for the benchmark |

---

## Runtime flow

1. `main.py` iterates over each `system_prompts/*.txt` file.
2. For each system prompt it creates a `RAGApp`, calls `setup()` (loads FAISS + LLM), then logs in with the test credentials.
3. Each attack prompt is sent through `app.ask()`:
   - Prompt Guard screens the input.
   - FAISS retrieves the top-k relevant document chunks.
   - The LLM generates a response using the retrieved context.
   - If `REDACT_PII` is enabled, the output is sanitized before logging.
4. Results are written to CSV in `results/unprotected/` (one file per system prompt).

---

## Security features

- **Prompt Guard** (`PromptGuard` from `src/shared/`): classifies input using `meta-llama/Llama-Prompt-Guard-2-22M`. Requests above `PROMPT_GUARD_THRESHOLD` are rejected.
- **PII redaction**: masks long digit sequences (credit card numbers), SSNs (`\d{3}-\d{2}-\d{4}`), and email local parts.
- **Supabase row-level security**: `query_user_transactions` filters by `user_id=eq.<authenticated_user>`.

---

## Differences from employee-facing

| Feature | Customer-facing | Employee-facing |
|---------|----------------|-----------------|
| Supabase auth | Raw HTTP (`requests`) | Supabase Python SDK |
| File downloads | Not supported | `download_user_files()` via SDK |
| Default data path | `./docs` | `./downloads` |
| Index rebuild default | `False` | `True` |

---

*See the top-level `README.md` for an architecture overview of both environments.*
