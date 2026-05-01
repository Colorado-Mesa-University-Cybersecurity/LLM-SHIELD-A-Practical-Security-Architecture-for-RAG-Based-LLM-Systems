#!/usr/bin/env python3
"""
Master benchmark runner — 8 test configurations:

  Sides  : customer-facing, employee-facing
  Models : Qwen3-8B, Meta-Llama-3.1-8B-Instruct
  Guards : protected (PROMPT_GUARD_ENABLED=True), unprotected (False)

Results are saved to:
  results/<side>/<model>/<protected|unprotected>/<system_prompt_stem>.csv

Run from the project root:
  python main.py
"""

import csv
import importlib.util
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Set

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"

CUSTOMER_DIR = SRC_DIR / "customer-facing"
EMPLOYEE_DIR = SRC_DIR / "employee-facing"

RESULTS_ROOT = ROOT_DIR / "results"

# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FILES = [
    "100_prompt.txt",
    "500_prompt.txt",
    "1000_prompt.txt",
    "1500_prompt.txt",
]

class Config(NamedTuple):
    side_name:  str       # "customer-facing" | "employee-facing"
    side_dir:   Path      # absolute path to the side module
    docs_dir:   Path      # directory used as DATA_PATHS for the index
    model_name: str       # key matching MODELS list in rag.py
    protected:  bool      # True => PROMPT_GUARD_ENABLED

CONFIGS: List[Config] = [
    # ── customer-facing ────────────────────────────────────────────────────
    Config("customer-facing", CUSTOMER_DIR, CUSTOMER_DIR / "docs", "Qwen3-8B",                   True),
    Config("customer-facing", CUSTOMER_DIR, CUSTOMER_DIR / "docs", "Qwen3-8B",                   False),
    Config("customer-facing", CUSTOMER_DIR, CUSTOMER_DIR / "docs", "Meta-Llama-3.1-8B-Instruct", True),
    Config("customer-facing", CUSTOMER_DIR, CUSTOMER_DIR / "docs", "Meta-Llama-3.1-8B-Instruct", False),
    # ── employee-facing ────────────────────────────────────────────────────
    Config("employee-facing", EMPLOYEE_DIR, EMPLOYEE_DIR / "docs", "Qwen3-8B",                   True),
    Config("employee-facing", EMPLOYEE_DIR, EMPLOYEE_DIR / "docs", "Qwen3-8B",                   False),
    Config("employee-facing", EMPLOYEE_DIR, EMPLOYEE_DIR / "docs", "Meta-Llama-3.1-8B-Instruct", True),
    Config("employee-facing", EMPLOYEE_DIR, EMPLOYEE_DIR / "docs", "Meta-Llama-3.1-8B-Instruct", False),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_EMAIL_RE = re.compile(
    r'^[a-zA-Z0-9_.+-]{1,64}@[a-zA-Z0-9-]{1,253}\.[a-zA-Z]{2,}$'
)


def _safe_email(email: str) -> str:
    return email if _VALID_EMAIL_RE.match(email) else "[redacted]"


def load_rag_module(side_dir: Path):
    """
    Dynamically load rag.py from *side_dir* without polluting sys.modules.

    Each side's rag.py imports `from prompt_guard import PromptGuard`.
    The local prompt_guard.py shim does:
        sys.path.insert(0, Path(__file__).parent.parent)   # -> src/
        from shared.prompt_guard import PromptGuard
    That resolution still works when loaded via importlib because __file__
    is set to the real file path by spec_from_file_location.
    """
    mod_name = f"rag_{side_dir.name}"
    rag_path  = side_dir / "rag.py"

    # Temporarily expose both the side dir and src/ so all local imports resolve.
    to_prepend = [str(SRC_DIR), str(side_dir)]
    for p in reversed(to_prepend):
        sys.path.insert(0, p)
    try:
        spec = importlib.util.spec_from_file_location(mod_name, str(rag_path))
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


# ---------------------------------------------------------------------------
# Single configuration runner
# ---------------------------------------------------------------------------

def run_one_config(
    cfg: Config,
    attack_prompts: List[str],
    first_run_for_side: bool,
) -> None:
    guard_label = "protected" if cfg.protected else "unprotected"
    log_dir = RESULTS_ROOT / cfg.side_name / cfg.model_name / guard_label

    existing_csvs = sorted(log_dir.glob("*.csv")) if log_dir.exists() else []
    if existing_csvs:
        print(f"\n  [!] Results already exist in {log_dir.relative_to(ROOT_DIR)}")
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

    print(f"\n{'='*72}")
    print(f"  Side  : {cfg.side_name}")
    print(f"  Model : {cfg.model_name}")
    print(f"  Guard : {guard_label}")
    print(f"  Logs  : {log_dir.relative_to(ROOT_DIR)}")
    print(f"{'='*72}")

    rag_mod  = load_rag_module(cfg.side_dir)
    base_cfg = dict(rag_mod.CONFIG)          # snapshot — never mutate the original

    vs_dir     = str(cfg.side_dir / ".faiss_index")
    data_paths = [str(cfg.docs_dir)]

    system_prompts_dir = cfg.side_dir / "docs" / "system_prompts"
    available = [f for f in SYSTEM_PROMPT_FILES if (system_prompts_dir / f).exists()]

    if not available:
        print(f"  [warn] No system prompt files found in {system_prompts_dir}, skipping.")
        return

    # Track whether the FAISS index has been built yet for this side.
    # The first system-prompt run for the first config of each side rebuilds;
    # every subsequent run (different system prompt or different config) reuses it.
    index_rebuilt = False

    for prompt_fname in available:
        prompt_path = system_prompts_dir / prompt_fname
        try:
            system_prompt_content = prompt_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"  [warn] Cannot read {prompt_path}: {e}")
            continue

        user_email  = base_cfg.get("TEST_EMAIL", "unknown@example.com")
        should_rebuild = first_run_for_side and not index_rebuilt
        if should_rebuild:
            index_rebuilt = True

        run_cfg: Dict[str, Any] = {
            **base_cfg,
            "DATA_PATHS":            data_paths,
            "VS_DIR":                vs_dir,
            "SYSTEM_PROMPT":         (
                f"The current user's email: {_safe_email(user_email)}\n\n"
                f"{system_prompt_content}"
            ),
            "LLM_NAME":              cfg.model_name,
            "REBUILD_INDEX":         should_rebuild,
            "Verbose":               False,
            # Never index benchmark infrastructure files
            "SKIP_FILENAMES":        [
                "attack_prompt.txt",
                "malicious prompt.txt",
                "100_prompt.txt",
                "500_prompt.txt",
                "1000_prompt.txt",
                "1500_prompt.txt",
            ],
            # --- protection flags: all on for protected, all off for unprotected ---
            "PROMPT_GUARD_ENABLED":              cfg.protected,
            "PROMPT_GUARD_CHECK_RETRIEVED_DOCS": cfg.protected,
            "REDACT_PII":                        cfg.protected,
            # Lower threshold for protected configs to reduce false negatives
            **({"PROMPT_GUARD_THRESHOLD": 0.4} if cfg.protected else {}),
        }

        log_path = log_dir / f"{Path(prompt_fname).stem}.csv"
        ensure_csv_header(log_path)

        stem_label = Path(prompt_fname).stem
        print(f"\n  [{stem_label}] writing to {log_path.relative_to(ROOT_DIR)}")

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


# ---------------------------------------------------------------------------
# Guardrail analysis (runs after all protected configs complete)
# ---------------------------------------------------------------------------

def run_guardrail_analysis() -> None:
    spec = importlib.util.spec_from_file_location(
        "guardrail_analysis",
        SRC_DIR / "shared" / "guardrail_analysis.py",
    )
    ga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ga)

    protected_dirs = sorted({
        RESULTS_ROOT / cfg.side_name / cfg.model_name / "protected"
        for cfg in CONFIGS
        if cfg.protected
    })

    for d in protected_dirs:
        if not d.exists() or not any(d.glob("*_prompt.csv")):
            print(f"  [guardrail] No CSVs found in {d.relative_to(ROOT_DIR)}, skipping.")
            continue
        print(f"\n  [guardrail] {d.relative_to(ROOT_DIR)}")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    attack_prompts_path = SRC_DIR / "shared" / "attack_prompt.txt"
    if not attack_prompts_path.exists():
        raise SystemExit(
            f"Attack prompts file not found: {attack_prompts_path}\n"
            "Expected location: src/shared/attack_prompt.txt"
        )

    attack_prompts = read_attack_prompts(attack_prompts_path)

    print("=" * 72)
    print("  Specialized-AI-Agents — Master Benchmark")
    print("=" * 72)
    print(f"  Attack prompts : {len(attack_prompts)} ({attack_prompts_path})")
    print(f"  System prompts : {SYSTEM_PROMPT_FILES}")
    print(f"  Configurations : {len(CONFIGS)}")
    print(f"  Results root   : {RESULTS_ROOT}")
    print()

    # Track which sides have had their FAISS index built so we only rebuild once.
    sides_with_built_index: Set[str] = set()

    total_start = time.perf_counter()

    for i, cfg in enumerate(CONFIGS, 1):
        first_run = cfg.side_name not in sides_with_built_index
        if first_run:
            sides_with_built_index.add(cfg.side_name)

        guard_label = "protected" if cfg.protected else "unprotected"
        print(
            f"\n[{i}/{len(CONFIGS)}] "
            f"{cfg.side_name} | {cfg.model_name} | {guard_label}"
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
