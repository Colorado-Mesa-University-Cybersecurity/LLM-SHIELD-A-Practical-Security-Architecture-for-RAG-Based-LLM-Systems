# RAG App — Code walkthrough and handoff notes

This document explains the `rag_app_setup.py` codebase in detail so another researcher can pick it up, run it, and continue development. It covers the overall purpose, file structure, key modules and classes, runtime flow, configuration and environment variables, debugging tips, and suggested next steps.

If anything here is unclear or you want extra diagrams or sequence traces, tell me what to expand.

---

## Quick summary

`rag_app_setup.py` is a small RAG (retrieval-augmented generation) app scaffold that:

- Connects to Supabase to authenticate a user and download that user's files from a storage bucket.
- Loads local files (PDF, text, CSV, Word, others) into LangChain `Document` objects.
- Optionally builds or loads a FAISS vector index of document chunks using HuggingFace embeddings.
- Loads a local chat model (GGUF via LangChain community Llama wrapper) and composes a basic RAG runnable chain.
- Provides PII redaction utilities and a prompt guard integration.
- Offers a CLI-style `__main__` that logs in, downloads files, and optionally builds an index.

Primary classes and functions you will work with:
- `CONFIG` — global configuration and feature flags
- `SupabaseSession` — Supabase auth, listing, and download helpers
- `RAGApp` — top-level app that wires Supabase, index, LLM, chain, and prompt guard
- Document loading, chunking, embedding, FAISS index build/load functions

---

## How to run

1. Create a Python environment with the required packages. The code uses:
   - supabase (supabase-py)
   - requests
   - python-dotenv (dotenv)
   - pandas
   - langchain- community modules in the script (see imports)
   - langchain-core, langchain-huggingface, langchain-text-splitters
   - huggingface-hub
   - tiktoken
   - FAISS backend and related binary dependencies
   - prompt_guard (a small helper, likely a third-party or in-repo tool)

   Exact package names and versions matter. If you need a requirements file, tell me and I can draft one.

2. Environment variables (via `.env` or export):
   - `SUPABASE_URL` — full Supabase URL (e.g. https://xxxx.supabase.co)
   - `SUPABASE_KEY` — service role or anon key (admin operations use the configured flow)
   - `SUPABASE_BUCKET` — optional string, default `"Company files"`
   - `SUPABASE_EMAIL` / `SUPABASE_PASSWORD` — test credentials used by the script main flow
   - Any Hugging Face credentials if your model access requires them

3. Run:
   - `python rag_app_setup.py`
   - The `__main__` section attempts to log in using `TEST_EMAIL` / `TEST_PASSWORD`. If login succeeds it will list and download user files into `downloads` and set `REBUILD_INDEX = True` if any files were retrieved, then call `setup()` which may build the index.

---

## File and function overview

I summarize the major code areas and explain intent, inputs, outputs, and gotchas.

### CONFIG
- Central place for runtime options, model names, and feature toggles.
- Important flags:
  - `DATA_PATHS`: directories to read documents from (default `./downloads`)
  - `VS_DIR`: persistent FAISS index directory
  - `REBUILD_INDEX`: if True, build a fresh vectorstore
  - `EMBED_MODEL`: HuggingFace embeddings model id
  - `LLM_NAME`, `CTX_TOKENS`, `MAX_NEW_TOKENS`, `TEMPERATURE`: local LLM settings
  - `TOP_K`: retrieval count for RAG responses
  - `PROMPT_GUARD_*`: prompt guard config
  - `REDACT_PII`: enable PII redaction for outputs and metadata

Change these to tune behavior. Many are read as strings or ints; be careful if loading from env.

### SupabaseSession
Responsibilities:
- Create and hold two supabase clients:
  - `admin_client` created with `SUPABASE_KEY` (the key from `CONFIG`) for authentication calls.
  - `user_client` created with the user's access token for storage reads that must run in the user's context.
- Authenticate using `admin_client.auth.sign_in_with_password(...)`. The code handles a few response shapes (object-like or dict).
- Provide storage helpers:
  - `list_objects(bucket, prefix, limit)` returns object names (normalized).
  - `download_object(bucket, object_path)` uses the user client to download a single object.
  - `download_objects_to_dir(bucket, object_paths, dest_dir)` saves object bytes to a local dir and returns result tuples.

Notes and pitfalls:
- The code expects the supabase SDK's `storage` APIs and normalizes several possible response shapes.
- The working script you provided earlier used `create_client(SUPABASE_URL, SUPABASE_KEY)` and then `supabase.storage.from_(BUCKET).list(UID, options)` and `download(...)` on a user-scoped client. The current class mimics that but builds a `user_client` with the user's access token. Different `supabase` library versions return different shapes for `list` and `download` results. The code tries to support both tuple/dict/object patterns. If you get unexpected responses, print the raw `auth_response` or `raw_list` to inspect their shape.
- Bucket names with spaces are supported; the SDK handles encoding. Avoid manual URL building for storage operations.

### Document loading and parsing
Functions:
- `_iter_paths` — yields file paths (supports directories recursively).
- `_load_one` — loads a single file based on suffix: PDF, txt, docx, csv/tsv, doc (word), or falls back to UnstructuredFileLoader for other types.
- `load_csv` — special CSV reading logic. It can create a `Document` per row or group rows into documents depending on `CSV_ROWS_PER_DOC`.
- `load_documents` — loads all files from `DATA_PATHS`.

Notes:
- For large CSVs, `CSV_READ_CHUNKSIZE` controls pandas chunked reading.
- Each returned `Document` has `page_content` and `metadata["source"]`.
- Some loaders rely on unstructured or other libraries that may have platform-specific dependencies.

### Token counting and chunking
- `_get_token_counter(model_hint)` uses `tiktoken` if available to estimate token counts. If `tiktoken` is unavailable, it falls back to a rough word-count approach.
- `chunk_docs` uses `RecursiveCharacterTextSplitter` with `chunk_size` and `chunk_overlap`. The `length_function` passed uses the token counter.
- This chunking is critical before building FAISS. If the tokens returned by the tokenizer differ from the model's expectations, tune `CHUNK_TOKENS`.

### Embeddings and FAISS
- `get_embeddings(model_name)` builds a `HuggingFaceEmbeddings` instance. It currently sets `device: cpu`. You can change `model_kwargs` to use GPU if available.
- `build_faiss(docs, embedder, persist_dir)` builds and saves a FAISS vectorstore.
- `load_faiss(embedder, persist_dir)` loads a saved FAISS store; it returns None on failure.

Behavior:
- `ensure_vectorstore` either loads an existing index or builds a new one from documents in `DATA_PATHS`. It prints progress messages. It raises `RuntimeError` if no documents were loaded and rebuild is requested.

### Model loading and chain construction
- `resolve_model_path(model_name)` downloads or resolves a GGUF file from the Hugging Face hub using `hf_hub_download` and a small `MODELS` mapping.
- `load_chat_model` constructs a `ChatLlamaCpp` instance with model path and runtime parameters.
- `build_rag_chain` composes:
  - A retriever from FAISS (`vectorstore.as_retriever(search_kwargs={"k": k})`)
  - A `ChatPromptTemplate` with `system` and `human` messages
  - A runnable chain: context composition followed by `prompt` then `llm` then `StrOutputParser()`

Notes:
- The system prompt string inside `build_rag_chain` is long and instructive. You can replace `DEFAULT_SYSTEM_PROMPT` in `CONFIG` if you need a different base behavior.
- The chain is built only when `self.vs` exists. If no vectorstore is available, `self.chain` is None and `ask()` raises an error.

### Prompt guard
- If `PROMPT_GUARD_ENABLED` is true and a `PROMPT_GUARD_MODEL` is configured, `PromptGuard(...)` is initialized.
- `ask()` runs guard checks on the user question and (optionally) on retrieved docs before invoking the chain.
- If the guard flags the input as malicious, the request is blocked and a short refusal message is returned.

Notes:
- `PromptGuard` is an external dependency; inspect its API and model requirements if you need to update behavior.
- The code prints guard labels and probabilities for debugging.

### PII redaction
- `redact_text` runs three canonical redactions:
  - Mask long digit sequences (likely credit cards), keeping last 4 digits.
  - Replace SSN patterns with `***-**-****`.
  - Obfuscate email local part to show first character then stars.
- `sanitize_metadata` ensures metadata keys are safe and lengths are bounded.
- If `REDACT_PII` is enabled, `ask()` applies `redact_output`.

---

## Runtime flow (end-to-end)

1. Script starts. `CONFIG` is loaded (reads environment variables).
2. `app = RAGApp(CONFIG)` creates the app and constructs `SupabaseSession`.
3. `app.login(email, pwd)` authenticates with Supabase and creates a user-scoped client.
4. `app.download_user_files(None, dest_dir="downloads")`:
   - If `filenames` is None, call `sb.list_objects(bucket, prefix=user_prefix)` to enumerate user files.
   - Build full object paths and call `sb.download_objects_to_dir`.
   - Save files to `downloads`.
5. If any files were downloaded, set `CONFIG['REBUILD_INDEX'] = True` and call `app.setup()`:
   - `setup()` ensures directories exist, then calls `ensure_vectorstore`. If REBUILD is true or no index exists, it builds FAISS with the specified embedding model.
   - It loads the chat LLM regardless, and then builds the RAG chain if the vectorstore exists.
   - It also initializes the prompt guard if enabled.

6. After setup, use `app.ask(question)` to:
   - Optionally run the prompt guard against the question.
   - If the user is authenticated and the question looks like it needs DB context (seen via `should_fetch_db()`), call `sb.query_user_transactions()` and format the DB rows for the chain.
   - Optionally check retrieved docs with the prompt guard.
   - Invoke the chain runnable with `input` and `db_context`.
   - Optionally redact the output.

---

## Debugging tips and common failure modes

1. Bucket not found error
   - If you see `Bucket not found` with HTTP 404, check `SUPABASE_BUCKET` value and its permissions.
   - Use the supabase-py SDK directly in a small REPL to confirm `storage.from_(BUCKET).list()` behavior and return shape. The SDK handles encoding and paths; constructing raw URLs often leads to mistakes.

2. SDK response shapes
   - Different versions of `supabase` return `list()` results as `(data, error)` tuple, or as a dict with `data` key, or as a plain list. The code attempts to normalize each case, but if you get errors, print the raw return value (e.g., `print(raw_list)`).

3. Authentication
   - `auth_response` shape varies by SDK version. If `user` or `session` are missing, print `auth_response` to inspect. You may need a more explicit extraction depending on lib version.
   - Token scopes: when creating the user-scoped client with `create_client(SUPABASE_URL, ACCESS_TOKEN)`, some setups expect that token to be passed differently. If downloads fail due to permission, the token may not be valid for storage. Try using the SDK pattern from the working script: authenticate through `supabase.auth.sign_in_with_password` and then reuse the returned client object for storage calls.

4. Model and LLM loading
   - `hf_hub_download` writes local model files. Ensure sufficient disk space and correct repo_id/filename in `MODELS`.
   - `ChatLlamaCpp` may require specific runtime dependencies (ggml, llama.cpp binding). Confirm those binaries are installed and match the model file format.

5. FAISS and embeddings
   - FAISS requires compiled binaries. If `FAISS.from_documents` fails, inspect binary compatibility or install a CPU-only wheel.
   - Embedding model memory: embedding models can be large. `HuggingFaceEmbeddings` currently sets `device: cpu`. To speed up embedding you can enable GPU if your environment supports it.

6. Token counting and chunk sizes
   - If retrieval quality is poor because chunks are too big or small, adjust `CHUNK_TOKENS` and `CHUNK_TOKEN_OVERLAP`. Use `_get_token_counter` with `tiktoken` if available to match model tokens.

7. Prompt guard false positives
   - If legitimate questions are blocked, examine the guard model output and lower the threshold or disable `PROMPT_GUARD_CHECK_RETRIEVED_DOCS`.

---

## Security and privacy notes

- `SUPABASE_KEY` is sensitive. Avoid embedding service-role keys in public repos. Use environment variables and a secrets manager.
- `REDACT_PII` is off by default. If your data contains PII, enable it, but test to ensure it redacts fields you expect.
- The code downloads user files to a local `downloads` folder. Ensure proper file permissions and consider encrypting disk if sensitive.
- Prompt guard model and LLMs should not be exposed to untrusted users if they can change system prompts.

---

## Quick reference — important function signatures

- SupabaseSession.login(email: str, password: str) -> bool
- SupabaseSession.list_objects(bucket: str, prefix: str = "", limit: int = 1000) -> List[str]
- SupabaseSession.download_objects_to_dir(bucket: str, object_paths: List[str], dest_dir: str) -> List[Tuple[str, bool, Optional[str]]]
- load_documents(inputs: Iterable[str]) -> List[Document]
- chunk_docs(docs: List[Document], chunk_tokens: Optional[int], chunk_overlap_tokens: Optional[int], model_hint: Optional[str]) -> List[Document]
- ensure_vectorstore(data_paths: List[str], vs_dir: Path, rebuild: bool, embed_model_name: str) -> Tuple[FAISS, HuggingFaceEmbeddings]
- load_chat_model(model_name: str, n_ctx: int, n_gpu_layers: int, temperature: float, max_tokens: int) -> ChatLlamaCpp
- RAGApp.ask(question: str) -> str