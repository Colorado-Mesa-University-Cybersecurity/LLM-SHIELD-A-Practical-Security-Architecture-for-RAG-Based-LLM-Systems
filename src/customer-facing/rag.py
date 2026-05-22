import os
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
import math
import re
import requests
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
# LangChain / LLM / embeddings / vector store
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompt_guard import PromptGuard

load_dotenv()

# Config
# You can login into hugging face for meta models by getting your token here: https://huggingface.co/settings/tokens
# and running: hf auth login in your terminal
CONFIG: Dict[str, Any] = {
    "DATA_PATHS": ["./docs"],
    "VS_DIR": "./.faiss_index",
    "REBUILD_INDEX": False,

    "EMBED_MODEL": "BAAI/bge-small-en-v1.5",

    "LLM_NAME": "Qwen3-8B",  # "Qwen3-8B" | "Meta-Llama-3.1-8B-Instruct"
    "GPU_OFFLOAD": True,
    "CTX_TOKENS": 8192,
    "MAX_NEW_TOKENS": 1024,
    "N_BATCH": 512,
    "TEMPERATURE": 0.2,

    "TOP_K": 5,

    "SYSTEM_PROMPT": "",

    # Chunking now measured in tokens (not characters)
    "CHUNK_TOKENS": 512,
    "CHUNK_TOKEN_OVERLAP": 20,

    # Supabase (optional)
    "SUPABASE_URL": os.getenv("SUPABASE_URL"),
    "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
    "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY"),

    "DB_LIMIT": 20,

    "TEST_EMAIL": "sarahjohnson@outlook.com",
    "TEST_PASSWORD": "123",

    # CSV handling
    "CSV_ROWS_PER_DOC": 1,
    "CSV_READ_CHUNKSIZE": 512,

    # Filenames (case-insensitive) to skip during document loading
    "SKIP_FILENAMES": ["attack_prompt.txt"],

    # PII redaction (optional)
    # Set to True to apply PII redaction to documents and DB output.
    "REDACT_PII": False,

    # Prompt Guard settings
    "Verbose": True,
    "PROMPT_GUARD_ENABLED": True,
    "PROMPT_GUARD_MODEL": "fmops/distilbert-prompt-injection",
    "PROMPT_GUARD_THRESHOLD": 0.6,
    # If True, also run the guard on retrieved documents before passing to LLM
    "PROMPT_GUARD_CHECK_RETRIEVED_DOCS": False,
}

MODELS = [
    {
        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "name": "Qwen3-8B",
    },
    {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "name": "Meta-Llama-3.1-8B-Instruct",
    },
]

# Basic PII redaction patterns
_CC_RE = re.compile(r'(?<!\d)(\d{13,19})(?!\d)')
_SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_EMAIL_RE = re.compile(r'([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')


def mask_digits_keep_last(s: str, keep: int = 4) -> str:
    if not s:
        return s
    digits = re.sub(r'\D', '', s)
    if not digits:
        return s  # nothing to mask
    if len(digits) <= keep:
        return '*' * len(digits)
    return '*' * (len(digits) - keep) + digits[-keep:]


def redact_text(text: str, max_len: int = 20000) -> str:
    if not text:
        return text
    t = text

    # Mask long digit sequences (credit cards)
    def _mask_cc(m):
        return mask_digits_keep_last(m.group(0), keep=4)

    t = _CC_RE.sub(_mask_cc, t)

    # Mask SSNs
    t = _SSN_RE.sub('***-**-****', t)

    # Mask emails: keep domain, partially mask user
    def _mask_email(m):
        user, domain = m.group(1), m.group(2)
        if len(user) <= 1:
            masked_user = '*'
        else:
            masked_user = user[0] + '*' * (len(user) - 1)
        return masked_user + '@' + domain

    t = _EMAIL_RE.sub(_mask_email, t)

    # Truncate overly long text
    if len(t) > max_len:
        t = t[: max_len - 3] + "..."
    return t


def redact_output(text: str, max_len: int = 20000) -> str:
    return redact_text(text, max_len=max_len)


def sanitize_metadata_key(k: str, max_len: int = 64) -> str:
    if not k:
        return "meta"
    nk = re.sub(r'[^0-9a-zA-Z_]', '_', k)
    return nk[:max_len]


def sanitize_metadata(metadata: Dict[str, Any], value_max_len: int = 1000) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metadata.items():
        sk = sanitize_metadata_key(str(k))
        if isinstance(v, str):
            sv = redact_text(v, max_len=value_max_len)
        else:
            try:
                sv = str(v)
            except Exception:
                sv = "<non-serializable>"
        if len(sv) > value_max_len:
            sv = sv[: value_max_len - 3] + "..."
        out[sk] = sv
    return out


def redact_document(doc: Document, max_content_len: int = 20000, max_meta_len: int = 1000) -> Document:
    content = redact_text(doc.page_content or "", max_len=max_content_len)
    metadata = sanitize_metadata(doc.metadata or {}, value_max_len=max_meta_len)
    return Document(page_content=content, metadata=metadata)


# Document loading

def _iter_paths(inputs: Iterable[str]) -> Iterable[Path]:
    for p in inputs:
        path = Path(p).expanduser().resolve()
        if path.is_dir():
            for f in path.rglob("*"):
                if f.is_file():
                    yield f
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(f"Path not found: {p}")


def load_csv(path: Path, rows_per_doc: int = 1, sep: Optional[str] = None, encoding: str = "utf-8",
             chunksize: Optional[int] = None) -> List[Document]:
    docs: List[Document] = []

    def df_to_docs(df: pd.DataFrame):
        df = df.fillna("").astype(str)
        num_rows = len(df)
        if rows_per_doc <= 1:
            for i, row in df.iterrows():
                # human-readable text
                content_lines = [f"{col}: {row[col]}" for col in df.columns]
                content = "\n".join(content_lines)
                metadata = {"source": str(path), "row_index": int(i)}
                # add column values to metadata (prefix to avoid clashes)
                for col in df.columns:
                    key = f"csv_{col}" if col else f"csv_col_{i}"
                    # metadata keys should be safe strings
                    metadata[key] = row[col]
                docs.append(Document(page_content=content, metadata=metadata))
        else:
            num_docs = math.ceil(num_rows / rows_per_doc)
            for d in range(num_docs):
                start = d * rows_per_doc
                end = min((d + 1) * rows_per_doc, num_rows)
                chunk_rows = []
                meta_rows = []
                for i in range(start, end):
                    row = df.iloc[i]
                    row_text = "; ".join([f"{col}: {row[col]}" for col in df.columns])
                    chunk_rows.append(row_text)
                    meta_rows.append({"row_index": int(i), **{f"csv_{col}": row[col] for col in df.columns}})
                content = "\n".join(chunk_rows)
                metadata = {"source": str(path), "row_start": start, "row_end": end - 1, "rows": meta_rows}
                docs.append(Document(page_content=content, metadata=metadata))

    if chunksize:
        # streaming read
        reader = pd.read_csv(path, sep=sep, encoding=encoding, chunksize=chunksize)
        for chunk in reader:
            df_to_docs(chunk)
    else:
        df = pd.read_csv(path, sep=sep, encoding=encoding)
        df_to_docs(df)

    return docs


def _load_one(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
    elif suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        rows_per_doc = int(CONFIG.get("CSV_ROWS_PER_DOC", 1))
        chunksize = CONFIG.get("CSV_READ_CHUNKSIZE")
        chunksize_val = int(chunksize) if chunksize else None
        docs = load_csv(path, rows_per_doc=rows_per_doc, sep=sep, encoding="utf-8", chunksize=chunksize_val)
    elif suffix == ".doc":
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader  # type: ignore
        except Exception as e:
            raise RuntimeError("Install 'unstructured[doc]' to handle .doc, or convert to .docx") from e
        loader = UnstructuredWordDocumentLoader(str(path))
        docs = loader.load()
    else:
        try:
            from langchain_community.document_loaders import UnstructuredFileLoader  # type: ignore
            loader = UnstructuredFileLoader(str(path))
            docs = loader.load()
        except Exception:
            raise RuntimeError(f"Unsupported file type: {path.suffix} for {path}")

    for d in docs:
        d.metadata.setdefault("source", str(path))
    return docs


def load_documents(inputs: Iterable[str], skip_filenames: Optional[List[str]] = None) -> List[Document]:
    docs: List[Document] = []
    skip = {fn.lower() for fn in (skip_filenames if skip_filenames is not None else CONFIG.get("SKIP_FILENAMES", []))}
    for inp in list(inputs):
        try:
            for path in _iter_paths([inp]):
                if path.name.lower() in skip:
                    print(f"[loader] Excluding {path.name}")
                    continue
                try:
                    loaded = _load_one(path)
                    docs.extend(loaded)
                except Exception as e:
                    print(f"[loader] Skipping {path}: {e}")
        except FileNotFoundError as e:
            print(f"[loader] Skipping missing path: {e}")
    return docs


def _get_token_counter(model_hint: Optional[str] = None) -> Callable[[str], int]:
    try:
        import tiktoken

        def _tiktoken_count(s: str) -> int:
            if not s:
                return 0
            try:
                # try to get an encoding for a model if possible, else use cl100k_base
                try:
                    enc = tiktoken.encoding_for_model(model_hint) if model_hint else tiktoken.get_encoding("cl100k_base")
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(s))
            except Exception:
                # fallback to simple split
                return len(s.split())

        return _tiktoken_count
    except ImportError:
        pass
    def _fallback_count(s: str) -> int:
        if not s:
            return 0
        return len(re.findall(r"\S+", s))

    return _fallback_count


def chunk_docs(docs: List[Document], chunk_tokens: Optional[int] = None, chunk_overlap_tokens: Optional[int] = None,
               model_hint: Optional[str] = None) -> List[Document]:
    if chunk_tokens is None:
        chunk_tokens = int(CONFIG.get("CHUNK_TOKENS", 512))
    if chunk_overlap_tokens is None:
        chunk_overlap_tokens = int(CONFIG.get("CHUNK_TOKEN_OVERLAP", 20))

    token_counter = _get_token_counter(model_hint=model_hint)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=chunk_overlap_tokens,
        length_function=token_counter,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


# Embeddings / Vector store

def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"[embeddings] Using device: {device} ({model_name})")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_faiss(docs: List[Document], embedder: HuggingFaceEmbeddings, persist_dir: Path) -> FAISS:
    persist_dir.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(docs, embedder)
    vs.save_local(str(persist_dir))
    return vs


def load_faiss(embedder: HuggingFaceEmbeddings, persist_dir: Path) -> Optional[FAISS]:
    if not persist_dir.exists():
        return None
    try:
        vs = FAISS.load_local(str(persist_dir), embedder, allow_dangerous_deserialization=True)
        return vs
    except Exception:
        return None


def ensure_vectorstore(
        data_paths: List[str],
        vs_dir: Path,
        rebuild: bool,
        embed_model_name: str,
        skip_filenames: Optional[List[str]] = None,
) -> Tuple[FAISS, HuggingFaceEmbeddings]:
    embedder = get_embeddings(embed_model_name)
    vs = None
    if not rebuild:
        vs = load_faiss(embedder, vs_dir)
        if vs:
            print(f"[index] Loaded existing FAISS index from {vs_dir}")
            return vs, embedder

    print("[index] Loading source documents...")
    docs = load_documents(data_paths, skip_filenames=skip_filenames)
    if not docs:
        raise RuntimeError("No documents loaded. Check your paths and supported file types.")
    print(f"[index] Loaded {len(docs)} raw documents.")

    print("[index] Splitting into chunks...")
    model_hint = embed_model_name or CONFIG.get("LLM_NAME")
    chunks = chunk_docs(docs,
                        chunk_tokens=int(CONFIG.get("CHUNK_TOKENS", 512)),
                        chunk_overlap_tokens=int(CONFIG.get("CHUNK_TOKEN_OVERLAP", 20)),
                        model_hint=model_hint)
    print(f"[index] Created {len(chunks)} chunks.")

    print(f"[index] Building FAISS at {vs_dir} ...")
    vs = build_faiss(chunks, embedder, vs_dir)
    print("[index] Done.")
    return vs, embedder


# LLM (llama.cpp, GGUF)

def resolve_model_path(model_name: str) -> str:
    specs = {m["name"]: m for m in MODELS}
    if model_name not in specs:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(specs.keys())}")
    spec = specs[model_name]
    local_path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"], local_dir=None)
    return local_path


def load_chat_model(
        model_name: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        n_batch: int = 512,
        verbose: bool = False,
) -> ChatLlamaCpp:
    model_path = resolve_model_path(model_name)
    if n_gpu_layers == -1:
        print(f"[llm] Using device: cuda (all layers offloaded) ({model_name})")
    elif n_gpu_layers > 0:
        print(f"[llm] Using device: cuda ({n_gpu_layers} layers offloaded) ({model_name})")
    else:
        print(f"[llm] Using device: cpu ({model_name})")
    llm = ChatLlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        temperature=temperature,
        max_tokens=max_tokens,
        n_batch=n_batch,
        verbose=verbose,
    )
    return llm


# RAG chain

DEFAULT_SYSTEM_PROMPT = CONFIG["SYSTEM_PROMPT"]


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "")
        pg = d.metadata.get("page", d.metadata.get("page_number", ""))
        tag = f" (source={src}{f'[p{pg}]' if pg != '' else ''})" if src else ""
        parts.append(d.page_content + tag)
    return "\n\n".join(parts)


def build_rag_chain(
        llm: BaseChatModel,
        vectorstore: FAISS,
        system_prompt: Optional[str] = None,
        k: int = 4,
) -> Runnable:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (system_prompt or DEFAULT_SYSTEM_PROMPT)
                + "\n\nYou are a highly capable, thoughtful, and precise assistant. Your goal is to deeply understand the user's intent, "
                  "ask clarifying questions when needed, think step-by-step through complex problems, provide clear and accurate answers, "
                  "and proactively anticipate helpful follow-up information. Always prioritize being truthful, nuanced, insightful, and "
                  "efficient, tailoring your responses specifically to the user's needs and preferences."
            ),
            (
                "human",
                "Question: {input}\n\n"
                "Retrieved Documents:\n{context}\n\n"
                "Database context for the authenticated user (if any):\n{db_context}\n\n"
                "Instructions:\n"
                "- Prefer precise, concise answers.\n"
                "- Use database context only when it directly answers a user-specific question.\n"
                "- If you use document content, cite with (source[page]) where possible.\n"
                "- If you don't have enough information, say you don't know."
            ),
        ]
    )

    # LCEL mapping: build inputs, then prompt -> llm -> str
    chain = (
            {
                "context": itemgetter("input") | retriever | _format_docs,
                "db_context": itemgetter("db_context"),
                "input": itemgetter("input"),
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain


def should_fetch_db(question: str) -> bool:
    q = question.lower()
    keywords = [
        "my transactions", "my transaction", "transactions", "renter transactions",
        "my payments", "my charges", "last transactions", "recent transactions",
        "account holder", "bank", "account type", "account mask", "routing",
        "card", "card brand", "card last4", "credit score", "annual income",
        "check number", "check date", "check payee", "check amount", "check memo", "credit card amount",
    ]
    return any(k in q for k in keywords)


# Supabase client (auth + query renter_transactions)

@dataclass
class SupabaseSession:
    supabase_url: str
    supabase_key: str
    supabase_anon_key: Optional[str] = None
    user_id: Optional[str] = None
    access_token: Optional[str] = None
    email: Optional[str] = None
    last_error: Optional[str] = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SupabaseSession":
        url = cfg.get("SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
        key = cfg.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY", "")
        anon_key = cfg.get("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY")
        return cls(supabase_url=url, supabase_key=key, supabase_anon_key=anon_key)

    def is_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    def is_authenticated(self) -> bool:
        return bool(self.user_id and self.access_token)

    def login(self, email: str, password: str) -> bool:
        self.last_error = None
        self.email = email

        if not self.is_configured():
            self.last_error = "Missing SUPABASE_URL or SUPABASE_KEY."
            return False

        auth_url = f"{self.supabase_url}/auth/v1/token?grant_type=password"
        try:
            r = requests.post(
                auth_url,
                headers={"apikey": self.supabase_key, "Content-Type": "application/json"},
                json={"email": email, "password": password},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            self.last_error = f"Auth request failed: {e}"
            return False

        self.user_id = data.get("user", {}).get("id")
        self.access_token = data.get("access_token")
        if not self.user_id or not self.access_token:
            self.last_error = "No user_id or access_token in auth response."
            self.user_id = None
            self.access_token = None
            return False

        return True

    def logout(self) -> None:
        self.user_id = None
        self.access_token = None
        self.email = None
        self.last_error = None

    def _auth_headers(self) -> Dict[str, str]:
        api_key = self.supabase_anon_key or self.supabase_key
        return {"apikey": api_key, "Authorization": f"Bearer {self.access_token}"}

    def query_user_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.is_authenticated():
            self.last_error = "Not authenticated."
            return []
        url = (
            f"{self.supabase_url}/rest/v1/renter_transactions"
            f"?user_id=eq.{self.user_id}&select=*&limit={int(limit)}"
        )
        try:
            r = requests.get(url, headers=self._auth_headers(), timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.last_error = f"Query failed: {e}"
            return []


def format_transactions_for_context(rows: List[Dict[str, Any]], max_len: int = 2000) -> str:
    if not rows:
        return "No transactions for this user."

    preferred_keys = [
        "Account Holder",
        "user_id",
        "bank",
        "account_type",
        "account_mask",
        "routing",
        "card_brand",
        "card_last4",
        "credit_score",
        "annual_income",
        "check_type",
        "check_number",
        "check_date",
        "check_payee",
        "check_amount",
        "check_memo",
        "credit_card_amount",
    ]

    redact_enabled = CONFIG.get("REDACT_PII", False)

    lines: List[str] = ["User renter_transactions rows:"]
    for idx, r in enumerate(rows, start=1):
        lines.append(f"- Row {idx}:")
        printed = set()
        for key in preferred_keys:
            val = r.get(key)
            if val is not None and val != "":
                sval = str(val)
                if redact_enabled:
                    sval = redact_text(sval)
                lines.append(f"  - {key}: {sval}")
                printed.add(key)
        for key, val in r.items():
            if key not in printed and val is not None and val != "":
                sval = str(val)
                if redact_enabled:
                    sval = redact_text(sval)
                lines.append(f"  - {key}: {sval}")

    text = "\n".join(lines)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


# High-level API

class RAGApp:
    def __init__(self, cfg: Dict[str, Any], llm: Optional[BaseChatModel] = None):
        self.cfg = cfg
        self.vs: Optional[FAISS] = None
        self.embedder: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[BaseChatModel] = llm
        self.chain: Optional[Runnable] = None
        self.sb = SupabaseSession.from_config(cfg)
        self.prompt_guard: Optional[PromptGuard] = None

    def setup(self) -> None:
        vs_dir = Path(self.cfg["VS_DIR"]).resolve()
        self.vs, self.embedder = ensure_vectorstore(
            data_paths=self.cfg["DATA_PATHS"],
            vs_dir=vs_dir,
            rebuild=self.cfg["REBUILD_INDEX"],
            embed_model_name=self.cfg["EMBED_MODEL"],
            skip_filenames=self.cfg.get("SKIP_FILENAMES"),
        )

        if self.llm is None:
            n_gpu_layers = -1 if self.cfg["GPU_OFFLOAD"] else 0
            self.llm = load_chat_model(
                model_name=self.cfg["LLM_NAME"],
                n_ctx=self.cfg["CTX_TOKENS"],
                n_gpu_layers=n_gpu_layers,
                temperature=self.cfg["TEMPERATURE"],
                max_tokens=self.cfg["MAX_NEW_TOKENS"],
                n_batch=self.cfg.get("N_BATCH", 512),
            )
        else:
            print(f"[llm] Reusing pre-loaded {self.cfg['LLM_NAME']}")
            try:
                self.llm.max_tokens = int(self.cfg["MAX_NEW_TOKENS"])
            except Exception:
                pass

        self.chain = build_rag_chain(
            self.llm, self.vs, system_prompt=self.cfg["SYSTEM_PROMPT"], k=self.cfg["TOP_K"]
        )

        # Initialize Prompt Guard if enabled
        if self.cfg.get("PROMPT_GUARD_ENABLED") and self.cfg.get("PROMPT_GUARD_MODEL"):
            try:
                self.prompt_guard = PromptGuard(
                    model_id=self.cfg.get("PROMPT_GUARD_MODEL"),
                    threshold=float(self.cfg.get("PROMPT_GUARD_THRESHOLD", 0.5)),
                    max_tokens=int(self.cfg.get("PROMPT_GUARD_MAX_TOKENS", 512)),
                    chunk_overlap=int(self.cfg.get("PROMPT_GUARD_CHUNK_OVERLAP", 50)),
                    temperature=float(self.cfg.get("PROMPT_GUARD_TEMPERATURE", 1.0)),
                )
                print(f"[security] Prompt Guard loaded: {self.cfg.get('PROMPT_GUARD_MODEL')}")
            except Exception as e:
                self.prompt_guard = None
                print(f"[security] Failed to load Prompt Guard: {e}")
        else:
            self.prompt_guard = None
            print("[security] Prompt Guard not enabled or no model configured.")

        if self.sb.is_configured():
            print("[db] Supabase is configured. Call app.login(email, password) to authenticate.")
        else:
            print(
                "[db] Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY in CONFIG or env to enable DB queries.")

        if self.cfg.get("REDACT_PII"):
            print("[security] PII redaction is ENABLED for documents and DB output.")
        else:
            print("[security] PII redaction is DISABLED. Set CONFIG['REDACT_PII']=True to enable redaction.")

    def rebuild_chain(self, system_prompt: str) -> None:
        """Swap the system prompt without reloading the LLM or FAISS index."""
        self.cfg["SYSTEM_PROMPT"] = system_prompt
        if self.llm is not None and self.vs is not None:
            self.chain = build_rag_chain(
                self.llm, self.vs, system_prompt=system_prompt, k=self.cfg["TOP_K"]
            )

    def login(self, email: str, password: str) -> bool:
        ok = self.sb.login(email, password)
        if not ok:
            print(f"[login] Failed: {self.sb.last_error or ''}".strip())
        else:
            print(f"[login] Logged in as {email} (user_id={self.sb.user_id})")
        return ok

    def logout(self) -> None:
        self.sb.logout()
        print("[login] Logged out.")

    def ask(self, question: str) -> str:
        if self.chain is None:
            raise RuntimeError("Call setup() first.")

        # First: run prompt guard on the raw user question (fast failure for direct injections/jailbreaks)
        if getattr(self, "prompt_guard", None):
            try:
                q_res = self.prompt_guard.check_text(question)
            except Exception as e:
                print(f"[guard] Error running prompt guard on question: {e}")
                return "We couldn't process your request due to an internal error. Please try again or contact support."

            if self.cfg.get("Verbose", True):
                label = q_res.get("label", "UNKNOWN")
                prob = float(q_res.get("malicious_prob", 0.0))
                is_mal = bool(q_res.get("is_malicious", False))
                print(
                    f"[guard] Question: label={label}, malicious_prob={prob:.3f}, is_malicious={is_mal}")

            if q_res.get("is_malicious"):
                if self.cfg.get("Verbose", True):
                    print(f"[guard] Blocked prompt: malicious_prob={q_res.get('malicious_prob', 0.0):.2f}")
                return "We couldn't process that request because it looks like an attempt to change system-level instructions. If you meant something else, please rephrase and try again. If this seems incorrect, contact support and include code: ERR_OVERRIDE_INSTRUCTIONS."

        # Decide whether to fetch DB context
        db_context = ""
        if self.sb.is_authenticated() and should_fetch_db(question):
            rows = self.sb.query_user_transactions(limit=self.cfg["DB_LIMIT"])
            db_context = format_transactions_for_context(rows)

        #  check top-k retrieved documents
        if getattr(self, "prompt_guard", None) and self.cfg.get("PROMPT_GUARD_CHECK_RETRIEVED_DOCS") and self.vs:
            try:
                docs = self.vs.similarity_search(question, k=int(self.cfg.get("TOP_K", 4)))
                docs_text = "\n\n".join([d.page_content for d in docs])
                d_res = self.prompt_guard.check_text(docs_text)
                if d_res.get("is_malicious"):
                    if self.cfg.get("Verbose", True):
                        print(f"[guard] Retrieved docs flagged: {d_res.get('malicious_prob', 0.0):.2f}")
                    return "Retrieved documents appear to include instructions that override system behavior. Cannot proceed."
            except Exception as e:
                print(f"[guard] Error checking retrieved docs: {e}")

        # Build inputs for the chain
        chain_inputs = {
            "input": question,
            "db_context": db_context,
        }

        # Invoke the RAG chain
        raw_answer = self.chain.invoke(chain_inputs)

        # Strip Qwen3 chain-of-thought blocks before returning or redacting.
        # <think>...</think> contains internal reasoning that must not be
        # surfaced to callers or stored in result CSVs.
        if "qwen" in self.cfg.get("LLM_NAME", "").lower():
            raw_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.S | re.I).strip()

        # Prepare user-facing answer: redact if enabled
        if self.cfg.get("REDACT_PII"):
            user_answer = redact_output(raw_answer)
        else:
            user_answer = raw_answer

        return user_answer