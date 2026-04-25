# Specialized AI Agents — Secure RAG Architecture

This repository implements a secure Retrieval-Augmented Generation (RAG) architecture designed for both internal staff and external clients, as described in:

*Anthony Mora, Jayden Alonzo-Estrada, Ram Basnet. "A Practical Security Architecture for Company RAG‑LLM Systems"*

---

## Repository layout

```
Specialized-AI-Agents/
├── main.py                    — master benchmark runner (all 8 configurations)
├── results/                   — all benchmark output CSVs
│   ├── customer-facing/
│   │   ├── Qwen3-8B/
│   │   │   ├── protected/
│   │   │   └── unprotected/
│   │   └── Meta-Llama-3.1-8B-Instruct/
│   │       ├── protected/
│   │       └── unprotected/
│   └── employee-facing/
│       ├── Qwen3-8B/
│       │   ├── protected/
│       │   └── unprotected/
│       └── Meta-Llama-3.1-8B-Instruct/
│           ├── protected/
│           └── unprotected/
└── src/
    ├── shared/
    │   ├── prompt_guard.py    — PromptGuard implementation (shared)
    │   └── attack_prompt.txt  — adversarial prompts used by all benchmarks
    ├── customer-facing/       — external client RAG environment
    └── employee-facing/       — internal staff RAG environment
```

---

## Environments

### 1. customer-facing

Designed for external clients querying company documents. Security emphasises strict access boundaries — clients can only interact with context relevant to them. Defensive system prompts, PII redaction, and prompt-guard input validation prevent cross-client data leakage and adversarial requests.

- **Audience:** External clients
- **Auth:** Raw HTTP to Supabase REST API
- **Features:** Prompt Guard, output PII redaction, Supabase row-level security
- **Example use:** Customer portals, knowledge bases

### 2. employee-facing

Designed for internal company staff with broader document access and Supabase Storage file download support. Security remains a priority (identity-based access control, PII redaction, prompt guard), but permission policies are suited for internal collaboration.

- **Audience:** Company employees
- **Auth:** Supabase Python SDK
- **Features:** Prompt Guard, file download from Supabase Storage, PII redaction
- **Example use:** Internal dashboards, HR document access, support tools

---

## Running the master benchmark

The root `main.py` runs **8 configurations** covering both environments, both supported models, and both protected/unprotected modes:

```bash
python main.py
```

| # | Side | Model | Guard |
|---|------|-------|-------|
| 1 | customer-facing | Qwen3-8B | protected |
| 2 | customer-facing | Qwen3-8B | unprotected |
| 3 | customer-facing | Meta-Llama-3.1-8B-Instruct | protected |
| 4 | customer-facing | Meta-Llama-3.1-8B-Instruct | unprotected |
| 5 | employee-facing | Qwen3-8B | protected |
| 6 | employee-facing | Qwen3-8B | unprotected |
| 7 | employee-facing | Meta-Llama-3.1-8B-Instruct | protected |
| 8 | employee-facing | Meta-Llama-3.1-8B-Instruct | unprotected |

Each configuration runs all four system prompt variants (`100_prompt`, `500_prompt`, `1000_prompt`, `1500_prompt`) against every line in `src/shared/attack_prompt.txt`, producing one CSV per system prompt under `results/<side>/<model>/<protected|unprotected>/`.

CSV columns: `system_prompt_file, question, response, duration_seconds`

Each side's individual `main.py` can also be run standalone (unprotected, Meta-Llama only):
```bash
cd src/customer-facing && python main.py
cd src/employee-facing && python main.py
```

---

## Security architecture

Both environments implement the same RAG security pipeline:

| Layer | Mechanism | Detail |
|-------|-----------|--------|
| Input screening | Prompt Guard | `meta-llama/Llama-Prompt-Guard-2-22M` classifies input; requests above threshold are rejected (fail-closed) |
| PII redaction | Regex-based | Masks credit cards (keep last 4), SSNs, and email local parts |
| Access control | Supabase RLS | DB queries scoped to `user_id=eq.<authenticated_user>` |
| Metadata safety | Key sanitisation | Metadata keys and values are normalised and length-capped |
| Document filtering | `SKIP_FILENAMES` | Configurable list of filenames excluded from RAG indexing |

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
