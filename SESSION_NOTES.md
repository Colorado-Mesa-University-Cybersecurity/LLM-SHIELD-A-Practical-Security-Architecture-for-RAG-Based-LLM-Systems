# Session Work Log — Specialized-AI-Agents (Branch: `50GPU`)

## 1. GPU Fine-Tuning Setup (`experiments/train.py`)
- Created a training script to fine-tune **Qwen2.5-0.5B** on the tiny-codes dataset
- Configured for the RTX 5090 using the `llm-finetuning` conda env (torch 2.10+cu128, which supports Blackwell sm_120)
- Hyperparameters: bfloat16, SDPA attention, gradient checkpointing, AdamW fused, batch size 2 + 8 grad accumulation steps, 1e-5 LR
- Script auto-resumes from the latest checkpoint
- **Currently at 56% complete (51,643 / 91,818 steps), loss ~0.368**

---

## 2. Academic Paper (`analysis.md`)
- Full rewrite to conference/journal standard (~961 lines)
- Sections: Abstract, Introduction (3 RQs, 5 contributions), Related Work (26 refs), Threat Model, System Architecture, Security Mechanisms, Benchmark Design, Domain Fine-Tuning, **Regulatory Compliance**, Feasibility Analysis, Discussion, Limitations, Conclusion, Appendices
- Added a dedicated **§9 Regulatory Compliance** section covering:
  - **HIPAA** (45 CFR §164.312) — PHI handling in healthcare RAG deployments
  - **DOE Order 205.1C** — CUI/OUO classification for energy sector
  - **CMMC 2.0 Level 2 / NIST SP 800-171** — DoD supply chain requirements
  - **ITAR / BIS ECCN 4E091** — export control implications for LLM technology
  - **NIST AI RMF 1.0 + NIST AI 600-1** — federal AI governance framework (interpreted as "FAISA" in the original request)

---

## 3. Prompt Datasets (`src/shared/`)
- **`attack_prompt.txt`** — 201 adversarial prompts across 10 attack categories:
  - Direct instruction override, role/persona hijacking, indirect/document injection, jailbreak (DAN, token manipulation), multi-turn Crescendo, social engineering, exfiltration, RLHF exploitation, context window saturation, obfuscation (Base64, leetspeak)
  - Grounded in JailbreakBench, HarmBench, Greshake et al., MITRE ATLAS AML.T0068
- **`benignPrompts.txt`** — 151 normal tenant/property-management queries covering rent, maintenance, leasing, amenities, move-in/out, compliance, and accessibility

---

## 4. Benchmark Sweep Environment Fixes (`aienv` conda env)
A number of dependency issues were resolved to get `main.py` running:
- Installed missing packages: `langchain-community`, `langchain-huggingface`, `faiss-cpu`, `supabase`, `tiktoken`, `docx2txt`, `pypdf`, `sentence-transformers`
- Removed `torchvision 0.24.0.dev20250901` (dev build incompatible with torch 2.6.0 — was breaking all HuggingFace transformer model loading)
- Upgraded `pyarrow` 19 → 24 and `datasets` 2.14 → 4.8 (ABI incompatibility fix)
- **Swapped prompt guard model** from gated `meta-llama/Llama-Prompt-Guard-2-86M` (requires Meta approval) to `fmops/distilbert-prompt-injection` (publicly accessible, same label convention — LABEL_1 = injection, matches `_MALICIOUS_IDX=1` in `prompt_guard.py`)

---

## 5. Benchmark Sweep (`main.py`)
- Launched the full guard config sweep: **52 configurations** (4 unprotected + 48 protected) × 4 system-prompt sizes = **208 CSV output files**
- Matrix: 12 guard configs (chunk size, overlap, softmax temperature, threshold) × 2 RAG sides (customer-facing, employee-facing) × 2 LLMs (Qwen3-8B Q4_K_M, Meta-Llama-3.1-8B-Instruct Q4_K_M)
- Both LLMs and embeddings confirmed running on GPU (`cuda, all layers offloaded`)
- **Currently paused at ~2/208 CSVs** (early in unprotected Qwen3-8B run)

---

## 6. Git / GitHub
- All work committed to branch `50GPU`
- Fixed staging errors: removed `.env` (credentials), `.faiss_index/` (FAISS build cache), and `results/` (benchmark output) from commits
- Extended `.gitignore` to permanently exclude `results/` and `**/.faiss_index/`
- Pushed to: `github.com/Colorado-Mesa-University-Cybersecurity/A-Practical-Security-Architecture-for-Company-RAG-LLM-System`

---

## Next Steps (when ready to resume)
```bash
# Resume both paused experiments
kill -CONT 34944 142448 142450 142573 142576

# Monitor progress
tail -f experiments/train.log   # fine-tuning
tail -f benchmark.log           # benchmark sweep
```

Once the 208 CSVs are complete, run `src/shared/guardrail_analysis.py` to compute TPR/FPR metrics and add the empirical results to `analysis.md`.
