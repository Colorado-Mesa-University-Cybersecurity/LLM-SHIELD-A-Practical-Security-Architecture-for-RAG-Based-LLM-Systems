import csv
import os
import time
from pathlib import Path
from typing import List, Set
from rag import RAGApp, CONFIG

SYSTEM_PROMPTS_DIR = Path("docs/system_prompts")  # directory containing the prompt files
SYSTEM_PROMPTS: Set[str] = {
    "100_prompt.txt",
    "300_prompt.txt",
    "500_prompt.txt",
    "1000_prompt.txt",
}

PROMPTS_FILE = Path("docs/asks.txt")  # each line is a user prompt (blank lines and lines starting with # are ignored)
LOG_DIR = Path("results/prompts")
DEFAULT_LOG_STEM = "run_results"


def ensure_log_header(path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["system_prompt_file", "question", "response", "duration_seconds"])


def append_log(path: Path, system_prompt_file: str, question: str, response: str, duration_seconds: float):
    ensure_log_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([system_prompt_file, question, response, f"{duration_seconds:.6f}"])


def read_system_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"System prompt file not found: {path}. Please create the file and add your desired system prompt.")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"System prompt file is empty: {path}. Please add your system prompt.")
    return content


def read_asks_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Asks file not found: {path}. Please create it with one prompt per line.")
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    if not lines:
        raise ValueError(f"Asks file {path} contains no prompts (after skipping blank/comment lines).")
    return lines


def run_for_system_prompt(system_prompt_path: Path, asks: List[str]):
    system_prompt_content = read_system_prompt(system_prompt_path)
    user_email = CONFIG.get("TEST_EMAIL", "unknown@example.com")
    CONFIG["SYSTEM_PROMPT"] = f"The current user's email: {user_email}\n\n{system_prompt_content}"
    CONFIG["PROMPT_GUARD_ENABLED"] = False
    CONFIG["REBUILD_INDEX"] = True
    CONFIG["REDACT_PII"] = True
    CONFIG["LLM_NAME"] = "Meta-Llama-3.1-8B-Instruct"

    log_name = f"{system_prompt_path.stem}.csv"
    log_path = LOG_DIR / log_name
    ensure_log_header(log_path)

    app = RAGApp(CONFIG)
    app.setup()
    app.login(CONFIG["TEST_EMAIL"], CONFIG["TEST_PASSWORD"])

    print(f"Running prompts for system prompt file: {system_prompt_path} -> logging to {log_path}")
    for q in asks:
        print(f"\nQuestion: {q}")
        start = time.perf_counter()
        try:
            resp = app.ask(q)
        except Exception as e:
            resp = f"ERROR: {e}"
        finally:
            end = time.perf_counter()
            duration = end - start

        print(resp)
        print(f"Response time: {duration:.3f} s")
        append_log(log_path, system_prompt_path.name, q, resp, duration)


def main():
    if not SYSTEM_PROMPTS:
        raise SystemExit("SYSTEM_PROMPTS is empty. Add filenames to SYSTEM_PROMPTS set in the script.")

    if not SYSTEM_PROMPTS_DIR.exists():
        raise SystemExit(f"System prompts directory not found: {SYSTEM_PROMPTS_DIR}")

    prompt_files_to_run: List[Path] = []
    for fname in sorted(SYSTEM_PROMPTS):
        p = SYSTEM_PROMPTS_DIR / fname
        if not p.exists():
            print(f"Warning: system prompt file not found, skipping: {p}")
            continue
        if not p.is_file():
            print(f"Warning: not a file, skipping: {p}")
            continue
        prompt_files_to_run.append(p)

    if not prompt_files_to_run:
        raise SystemExit("No valid system prompt files found to run. Check SYSTEM_PROMPTS contents and the directory.")

    try:
        asks = read_asks_file(PROMPTS_FILE)
    except Exception as e:
        raise SystemExit(f"Error reading asks file: {e}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for sys_pf in prompt_files_to_run:
        try:
            run_for_system_prompt(sys_pf, asks)
        except Exception as e:
            print(f"ERROR running prompts for {sys_pf}: {e}")
            fallback = LOG_DIR / f"{DEFAULT_LOG_STEM}_errors.csv"
            append_log(fallback, sys_pf.name, "<all asks>", f"ERROR: {e}", 0.0)


if __name__ == "__main__":
    main()