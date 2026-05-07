#!/usr/bin/env python3
import csv
import importlib.util
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, cast

from dotenv import load_dotenv

# ── Directory layout ──────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


SRC_DIR = _env_path("BENCHMARK_SRC_DIR", ROOT_DIR / "src")
SHARED_DIR = _env_path("BENCHMARK_SHARED_DIR", SRC_DIR / "shared")

CUSTOMER_DIR = _env_path("BENCHMARK_CUSTOMER_DIR", SRC_DIR / "customer-facing")
EMPLOYEE_DIR = _env_path("BENCHMARK_EMPLOYEE_DIR", SRC_DIR / "employee-facing")

RESULTS_ROOT = _env_path("BENCHMARK_RESULTS_DIR", ROOT_DIR / "results")

# ── System prompt files ───────────────────────────────────────────────────────

SYSTEM_PROMPT_FILES = [
    "100_prompt.txt",
    "500_prompt.txt",
    "1000_prompt.txt",
    "1500_prompt.txt",
]

# ── Guard configurations ──────────────────────────────────────────────────────
# Each entry defines one protected run.  Threshold floor is 0.40.
# Temperature > 1.0 softens the softmax distribution, pushing borderline-
# scoring attacks (0.13–0.40) above the blocking threshold without moving

GUARD_CONFIGS: List[Dict[str, Any]] = [
    # ── 512-token chunks (tutorial default), temperature sweep ───────────────
    {
        "name": "c512_temp1.0_th0.40",
        "threshold": 0.40, "max_tokens": 512, "chunk_overlap": 50, "temperature": 1.0,
    },
    {
        "name": "c512_temp2.0_th0.40",
        "threshold": 0.40, "max_tokens": 512, "chunk_overlap": 50, "temperature": 2.0,
    },
    {
        "name": "c512_temp3.0_th0.40",
        "threshold": 0.40, "max_tokens": 512, "chunk_overlap": 50, "temperature": 3.0,
    },
    # ── 128-token chunks / 100-token overlap, temperature sweep ─────────────
    {
        "name": "c128_ov100_temp1.0_th0.40",
        "threshold": 0.40, "max_tokens": 128, "chunk_overlap": 100, "temperature": 1.0,
    },
    {
        "name": "c128_ov100_temp1.5_th0.40",
        "threshold": 0.40, "max_tokens": 128, "chunk_overlap": 100, "temperature": 1.5,
    },
    {
        "name": "c128_ov100_temp2.0_th0.40",
        "threshold": 0.40, "max_tokens": 128, "chunk_overlap": 100, "temperature": 2.0,
    },
    {
        "name": "c128_ov100_temp3.0_th0.40",
        "threshold": 0.40, "max_tokens": 128, "chunk_overlap": 100, "temperature": 3.0,
    },
    {
        "name": "c128_ov100_temp5.0_th0.40",
        "threshold": 0.40, "max_tokens": 128, "chunk_overlap": 100, "temperature": 5.0,
    },
    # ── Conservative threshold (0.50) with temperature ───────────────────────
    {
        "name": "c128_ov100_temp3.0_th0.50",
        "threshold": 0.50, "max_tokens": 128, "chunk_overlap": 100, "temperature": 3.0,
    },
    {
        "name": "c128_ov100_temp5.0_th0.50",
        "threshold": 0.50, "max_tokens": 128, "chunk_overlap": 100, "temperature": 5.0,
    },
    # ── 64-token chunks, high temperature ────────────────────────────────────
    {
        "name": "c64_ov50_temp3.0_th0.40",
        "threshold": 0.40, "max_tokens": 64, "chunk_overlap": 50, "temperature": 3.0,
    },
    {
        "name": "c64_ov50_temp5.0_th0.40",
        "threshold": 0.40, "max_tokens": 64, "chunk_overlap": 50, "temperature": 5.0,
    },
]

# ── Benchmark matrix ──────────────────────────────────────────────────────────

class Config(NamedTuple):
    side_name:  str                       # "customer-facing" | "employee-facing"
    side_dir:   Path                      # absolute path to the side's source dir
    docs_dir:   Path                      # used as DATA_PATHS for FAISS index
    model_name: str                       # must match a key in rag.py's MODELS list
    protected:  bool                      # True => Prompt Guard is active
    guard_cfg:  Optional[Dict[str, Any]]  # guard settings; None for unprotected


def _build_configs() -> List[Config]:
    sides = [
        ("customer-facing", CUSTOMER_DIR, CUSTOMER_DIR / "docs"),
        ("employee-facing", EMPLOYEE_DIR, EMPLOYEE_DIR / "docs"),
    ]
    models = ["Qwen3-8B", "Meta-Llama-3.1-8B-Instruct"]
    configs: List[Config] = []
    for side_name, side_dir, docs_dir in sides:
        for model_name in models:
            # One unprotected run per (side, model)
            configs.append(Config(side_name, side_dir, docs_dir, model_name, False, None))
            # One protected run per (side, model, guard config)
            for gc in GUARD_CONFIGS:
                configs.append(Config(side_name, side_dir, docs_dir, model_name, True, gc))
    return configs


CONFIGS: List[Config] = _build_configs()


def _result_dir_label(cfg: Config) -> str:
    """Return the subdirectory name for this config's CSV output."""
    if not cfg.protected:
        return "unprotected"
    return f"protected_{cfg.guard_cfg['name']}"  # type: ignore[index]


# ── Helpers ───────────────────────────────────────────────────────────────────

_VALID_EMAIL_RE = re.compile(
    r'^[a-zA-Z0-9_.+-]{1,64}@[a-zA-Z0-9-]{1,253}\.[a-zA-Z]{2,}$'
)


def _safe_email(email: str) -> str:
    return email if _VALID_EMAIL_RE.match(email) else "[redacted]"


def load_rag_module(side_dir: Path):
    """Dynamically load rag.py from side_dir without polluting sys.modules."""
    mod_name = f"rag_{side_dir.name}"
    rag_path  = side_dir / "rag.py"
    to_prepend = [str(SRC_DIR), str(SHARED_DIR), str(side_dir)]
    for p in reversed(to_prepend):
        sys.path.insert(0, p)
    try:
        spec = cast(Any, importlib.util.spec_from_file_location(mod_name, str(rag_path)))  # type: ignore[assignment]
        assert spec is not None and spec.loader is not None, (
            f"Could not load rag module from {rag_path}"
        )
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for _ in to_prepend:
            sys.path.pop(0)
    return mod


def read_attack_prompts(path: Path) -> List[str]:
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"No attack prompts found in {path}")
    return lines


def ensure_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                ["system_prompt_file", "question", "response", "duration_seconds"]
            )


def append_csv(
    path: Path,
    system_prompt_file: str,
    question: str,
    response: str,
    duration: float,
) -> None:
    ensure_csv_header(path)
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            [system_prompt_file, question, response, f"{duration:.6f}"]
        )


# ── Single configuration runner ───────────────────────────────────────────────

def run_one_config(
    cfg: Config,
    attack_prompts: List[str],
    first_run_for_side: bool,
) -> None:
    dir_label = _result_dir_label(cfg)
    log_dir   = RESULTS_ROOT / cfg.side_name / cfg.model_name / dir_label

    existing_csvs = sorted(log_dir.glob("*.csv")) if log_dir.exists() else []
    if existing_csvs:
        print(f"\n  [!] Results already exist in {_display_path(log_dir)}")
        print(f"      {len(existing_csvs)} file(s): {', '.join(f.name for f in existing_csvs)}")
        try:
            answer = input("      Override? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("      Skipping.")
            return
        for f in existing_csvs:
            f.unlink()

    log_dir.mkdir(parents=True, exist_ok=True)

    gc = cfg.guard_cfg
    if gc:
        guard_detail = (
            f"protected  threshold={gc['threshold']}"
            f"  temp={gc['temperature']}"
            f"  chunks={gc['max_tokens']}"
            f"  overlap={gc['chunk_overlap']}"
        )
    else:
        guard_detail = "unprotected"

    print(f"\n{'='*72}")
    print(f"  Side  : {cfg.side_name}")
    print(f"  Model : {cfg.model_name}")
    print(f"  Guard : {guard_detail}")
    print(f"  Logs  : {_display_path(log_dir)}")
    print(f"{'='*72}")

    rag_mod  = load_rag_module(cfg.side_dir)
    base_cfg = dict(rag_mod.CONFIG)

    vs_dir     = str(cfg.side_dir / ".faiss_index")
    data_paths = [str(cfg.docs_dir)]

    system_prompts_dir = cfg.side_dir / "docs" / "system_prompts"
    available = [f for f in SYSTEM_PROMPT_FILES if (system_prompts_dir / f).exists()]

    if not available:
        print(f"  [warn] No system prompt files found in {system_prompts_dir}, skipping.")
        return

    index_rebuilt = False

    for prompt_fname in available:
        prompt_path = system_prompts_dir / prompt_fname
        try:
            system_prompt_content = prompt_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"  [warn] Cannot read {prompt_path}: {e}")
            continue

        user_email     = base_cfg.get("TEST_EMAIL", "unknown@example.com")
        should_rebuild = first_run_for_side and not index_rebuilt
        if should_rebuild:
            index_rebuilt = True

        run_cfg: Dict[str, Any] = {
            **base_cfg,
            "DATA_PATHS":   data_paths,
            "VS_DIR":       vs_dir,
            "SYSTEM_PROMPT": (
                f"The current user's email: {_safe_email(user_email)}\n\n"
                f"{system_prompt_content}"
            ),
            "LLM_NAME":     cfg.model_name,
            "REBUILD_INDEX": should_rebuild,
            "Verbose":      False,
            "SKIP_FILENAMES": [
                "attack_prompt.txt", "malicious prompt.txt",
                "100_prompt.txt", "500_prompt.txt",
                "1000_prompt.txt", "1500_prompt.txt",
            ],
            # Protection flags — all on for protected, all off for unprotected
            "PROMPT_GUARD_ENABLED":              cfg.protected,
            "PROMPT_GUARD_CHECK_RETRIEVED_DOCS": cfg.protected,
            "REDACT_PII":                        cfg.protected,
            # Guard tuning — sourced from the guard config entry for protected runs
            "PROMPT_GUARD_THRESHOLD":    gc["threshold"]     if gc else 0.5,
            "PROMPT_GUARD_MAX_TOKENS":   gc["max_tokens"]    if gc else 512,
            "PROMPT_GUARD_CHUNK_OVERLAP": gc["chunk_overlap"] if gc else 50,
            "PROMPT_GUARD_TEMPERATURE":  gc["temperature"]   if gc else 1.0,
        }

        log_path = log_dir / f"{Path(prompt_fname).stem}.csv"
        ensure_csv_header(log_path)

        stem_label = Path(prompt_fname).stem
        print(f"\n  [{stem_label}] writing to {_display_path(log_path)}")

        try:
            app = rag_mod.RAGApp(run_cfg)
            app.setup()
            app.login(run_cfg["TEST_EMAIL"], run_cfg["TEST_PASSWORD"])
        except Exception as e:
            print(f"  [error] Setup/login failed for {prompt_fname}: {e}")
            append_csv(log_path, prompt_fname, "<setup>", f"ERROR: {e}", 0.0)
            continue

        ok_count = fail_count = 0
        for q in attack_prompts:
            start = time.perf_counter()
            try:
                resp = app.ask(q)
            except Exception as e:
                resp = f"ERROR: {e}"
                fail_count += 1
            else:
                ok_count += 1
            duration = time.perf_counter() - start
            append_csv(log_path, prompt_fname, q, resp, duration)

        print(
            f"    complete — {ok_count} answered, {fail_count} errors, "
            f"{ok_count + fail_count} total prompts"
        )


# ── Guardrail analysis ────────────────────────────────────────────────────────

def run_guardrail_analysis() -> None:
    spec = cast(Any, importlib.util.spec_from_file_location(  # type: ignore[assignment]
        "guardrail_analysis",
        SHARED_DIR / "guardrail_analysis.py",
    ))
    assert spec is not None and spec.loader is not None, (
        f"Could not load guardrail analysis module from {SHARED_DIR / 'guardrail_analysis.py'}"
    )
    ga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ga)

    # Every distinct protected_<name> directory across all (side, model) combos
    protected_dirs = sorted({
        RESULTS_ROOT / cfg.side_name / cfg.model_name / _result_dir_label(cfg)
        for cfg in CONFIGS
        if cfg.protected
    })

    for d in protected_dirs:
        if not d.exists() or not any(d.glob("*_prompt.csv")):
            print(f"  [guardrail] No CSVs in {_display_path(d)}, skipping.")
            continue
        print(f"\n  [guardrail] {_display_path(d)}")
        try:
            ga.build_guardrail_reports(
                prompts_dir=str(d),
                summary_csv="guardrail_summary.csv",
                failures_csv="guardrail_failures.csv",
                response_match_text=ga.TARGET_BLOCK_RESPONSE,
                filename_glob="*_prompt.csv",
                expected_prompt_ids=["100", "500", "1000", "1500"],
            )
        except Exception as e:
            print(f"  [guardrail] Error analysing {d}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    attack_prompts_path = SHARED_DIR / "attack_prompt.txt"
    if not attack_prompts_path.exists():
        raise SystemExit(
            f"Attack prompts file not found: {attack_prompts_path}\n"
            f"Expected location: {_display_path(SHARED_DIR / 'attack_prompt.txt')}"
        )

    attack_prompts = read_attack_prompts(attack_prompts_path)

    n_unprotected = sum(1 for c in CONFIGS if not c.protected)
    n_protected   = sum(1 for c in CONFIGS if c.protected)

    print("=" * 72)
    print("  Master Benchmark — Guard Config Sweep")
    print("=" * 72)
    print(f"  Attack prompts : {len(attack_prompts)}")
    print(f"  System prompts : {len(SYSTEM_PROMPT_FILES)}")
    print(f"  Guard configs  : {len(GUARD_CONFIGS)}")
    print(f"  Total configs  : {len(CONFIGS)}  "
          f"({n_unprotected} unprotected + {n_protected} protected)")
    print(f"  Results root   : {RESULTS_ROOT}")
    print()
    print("  Guard config names:")
    for gc in GUARD_CONFIGS:
        print(f"    {gc['name']}")
    print()

    sides_with_built_index: Set[str] = set()
    total_start = time.perf_counter()

    for i, cfg in enumerate(CONFIGS, 1):
        first_run = cfg.side_name not in sides_with_built_index
        if first_run:
            sides_with_built_index.add(cfg.side_name)

        dir_label = _result_dir_label(cfg)
        print(
            f"\n[{i}/{len(CONFIGS)}] "
            f"{cfg.side_name} | {cfg.model_name} | {dir_label}"
        )
        try:
            run_one_config(cfg, attack_prompts, first_run_for_side=first_run)
        except Exception as e:
            print(f"  [fatal] Configuration {i} failed: {e}")

    elapsed = time.perf_counter() - total_start
    print(f"\n{'='*72}")
    print(f"  All {len(CONFIGS)} configurations complete in {elapsed / 60:.1f} min.")
    print(f"  Results saved under: {RESULTS_ROOT}")
    print("=" * 72)

    print(f"\n{'='*72}")
    print("  Guardrail analysis")
    print("=" * 72)
    run_guardrail_analysis()


if __name__ == "__main__":
    main()
