#!/usr/bin/env python3
"""False-positive sweep over the classifier alone.

The guard runs before the language model, so a blocked prompt costs no
generation. Measuring the false-positive rate therefore does not need the
end-to-end pipeline: loading only the classifier lets the rate be measured over
a few hundred benign prompts in seconds rather than hours.

That matters because the rate reported from the end-to-end benchmark rests on
49 benign prompts. Two blocks out of 49 is 4.08% with a 95% Wilson interval of
1.1%-13.7%, which is too wide to support quoting a figure to three significant
digits. See CLAUDE.md.

Every guard configuration in main.py is swept, for every classifier variant in
GUARD_MODELS, against every prompt in the corpus. No LLM is loaded and no
Supabase connection is required.

Usage:
    python analysis/fp_sweep.py
    python analysis/fp_sweep.py --corpus src/shared/benignPrompts.txt
    python analysis/fp_sweep.py --out analysis/fp_sweep_results.csv
"""

import argparse
import csv
import importlib.util
import io
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

csv.field_size_limit(10 ** 9)

_spec = importlib.util.spec_from_file_location("benchmark_main", ROOT / "src" / "main.py")
main = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(main)                         # type: ignore[union-attr]

from prompt_guard import PromptGuard  # noqa: E402  (needs sys.path above)

DEFAULT_CORPUS = ROOT / "src" / "shared" / "benign_prompts_fp_sweep.csv"


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Point estimate and 95% Wilson score interval for a binomial proportion.

    Wilson rather than normal-approximation because k is small and p is near
    zero, where the normal interval understates the upper bound and can dip
    below zero.
    """
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_corpus(path: Path) -> List[Dict[str, str]]:
    with io.open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("no rows parsed from %s" % path)
    key = "prompt" if "prompt" in rows[0] else list(rows[0].keys())[0]
    out = []
    for r in rows:
        text = (r.get(key) or "").strip()
        if text:
            out.append({"prompt": text, "category": (r.get("category") or "").strip()})
    return out


def sweep(corpus: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for gm in main.GUARD_MODELS:
        allowed = gm.get("configs")
        for gc in main.GUARD_CONFIGS:
            if allowed is not None and gc["name"] not in allowed:
                continue
            print("[sweep] %s / %s" % (gm["label"], gc["name"]), flush=True)
            guard = PromptGuard(
                model_id=gm["model_id"],
                threshold=float(gc["threshold"]),
                max_tokens=int(gc["max_tokens"]),
                chunk_overlap=int(gc["chunk_overlap"]),
                temperature=float(gc["temperature"]),
            )
            for row in corpus:
                t0 = time.perf_counter()
                verdict = guard.check_text(row["prompt"])
                dt = time.perf_counter() - t0
                blocked = bool(verdict["is_malicious"])
                score = float(verdict["malicious_prob"])
                records.append({
                    "guard_model": gm["label"],
                    "config": gc["name"],
                    "threshold": gc["threshold"],
                    "chunk_tokens": gc["max_tokens"],
                    "chunk_overlap": gc["chunk_overlap"],
                    "temperature": gc["temperature"],
                    "category": row["category"],
                    "prompt": row["prompt"],
                    "blocked": int(blocked),
                    "score": score,
                    "seconds": round(dt, 6),
                })
            del guard
    return records


def report(records: List[Dict[str, Any]], n_prompts: int) -> None:
    print()
    print("=" * 78)
    print("  FALSE-POSITIVE RATE BY CONFIGURATION  (n = %d benign prompts each)" % n_prompts)
    print("=" * 78)
    print("  %-7s %-26s %6s %8s %19s %9s" % (
        "variant", "config", "blocks", "rate", "95% Wilson CI", "mean s"))
    keys = sorted({(r["guard_model"], r["config"]) for r in records})
    for gmlabel, cfgname in keys:
        rows = [r for r in records if r["guard_model"] == gmlabel and r["config"] == cfgname]
        k = sum(r["blocked"] for r in rows)
        p, lo, hi = wilson(k, len(rows))
        mean_s = sum(r["seconds"] for r in rows) / len(rows)
        print("  %-7s %-26s %6d %7.2f%% %7.2f%% - %6.2f%% %9.4f" % (
            gmlabel, cfgname, k, 100 * p, 100 * lo, 100 * hi, mean_s))

    print()
    print("  Blocked benign prompts, by how many configurations flagged each")
    print("  (a prompt flagged everywhere is a property of the classifier, not of tuning)")
    counts: Dict[str, int] = {}
    for r in records:
        if r["blocked"]:
            counts[r["prompt"]] = counts.get(r["prompt"], 0) + 1
    n_cfgs = len(keys)
    if not counts:
        print("    none")
    for prompt, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("    %2d/%-2d  %s" % (c, n_cfgs, prompt[:90]))

    print()
    print("  By prompt category (pooled across configurations)")
    cats = sorted({r["category"] for r in records if r["category"]})
    for cat in cats:
        rows = [r for r in records if r["category"] == cat]
        k = sum(r["blocked"] for r in rows)
        p, lo, hi = wilson(k, len(rows))
        print("    %-14s %5d/%-6d %6.2f%%  (%.2f%% - %.2f%%)" % (
            cat, k, len(rows), 100 * p, 100 * lo, 100 * hi))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "fp_sweep_results.csv")
    a = ap.parse_args()

    corpus = load_corpus(a.corpus)
    print("corpus     : %s" % a.corpus)
    print("prompts    : %d" % len(corpus))
    print("variants   : %s" % ", ".join(gm["label"] for gm in main.GUARD_MODELS))
    print("configs    : %d" % len(main.GUARD_CONFIGS))
    print()

    records = sweep(corpus)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    report(records, len(corpus))
    print()
    print("per-prompt results written to %s" % a.out)
