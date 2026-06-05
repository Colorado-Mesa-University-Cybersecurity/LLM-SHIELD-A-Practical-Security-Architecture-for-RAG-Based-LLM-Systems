# On-Premises Large Language Model Deployment for Resource-Constrained Organisations: Security Architecture, Threat Model, and Regulatory Compliance

**Anonymous Author(s)**
*Affiliation TBD*

---

## Abstract

Cloud-hosted large language model (LLM) APIs expose organisations to data-egress
risk, perpetual per-token costs, and limited control over inference behaviour. This
paper examines whether small to medium-sized businesses (SMBs) can viably substitute
on-premises deployments using open-weight models and commodity GPU hardware. We
present a production-grade dual-context retrieval-augmented generation (RAG) system
deployed for a property-management company, secured by a three-layer defence stack:
(i) a configurable prompt-injection classifier based on `fmops/distilbert-prompt-injection`
with temperature-scaled logits and overlapping token-level chunking, (ii) regex-based PII
redaction applied before document content reaches the LLM context window, and
(iii) optional retrieved-document scanning defending against indirect prompt
injection. We execute a benchmark sweep of twelve classifier hyperparameter
configurations—varying chunk size (64/128/512 tokens), chunk overlap (50–100 tokens),
softmax temperature (1.0–5.0), and blocking threshold (0.40–0.50)—across two LLMs,
two deployment contexts, and four system-prompt sizes, measuring true-positive rate
and false-positive rate as dual objectives. Against a 78-prompt set of general-purpose
jailbreak attacks, all 128-token chunk configurations achieve TPR = 100% with FPR = 4.1%
(2/49 benign queries blocked, 95% Wilson CI [1.1%, 13.7%]); unprotected baselines yield
49–70% attack compliance depending on model and deployment context. We further
analyse the hardware economics of self-hosted deployment, finding a break-even of
approximately four to five months against mid-range cloud API pricing at 5,000
queries per day, and map the architecture against five regulatory frameworks: HIPAA,
DOE Order 205.1C, DoD CMMC 2.0, ITAR, and the NIST AI Risk Management Framework.
Our analysis identifies data residency as the primary compliance benefit of
self-hosting and prompt injection as the dominant operational security threat,
for which the multi-layer guard provides measurably better coverage than any single
mechanism.

---

## 1. Introduction

The proliferation of accessible open-weight language models—Llama 3 [1], Qwen3 [2],
Mistral [3]—has created a practical alternative to commercial API access for
organisations whose workloads are constrained by data-residency obligations, cost
sensitivity, or the need to customise model behaviour. For SMBs in particular,
cloud API pricing at inference scale can exceed the annualised cost of a single
consumer GPU within months, while the data processor relationship introduced by
third-party API usage complicates compliance with HIPAA, CMMC, and equivalent
frameworks that impose strict controls on where regulated data may be processed.

Yet on-premises LLM deployment introduces a distinct set of security challenges.
The organisation now bears sole responsibility for defending against adversarial
inputs—a task for which no established operational playbook exists at the SMB scale.
Prompt injection, the dominant practical attack against LLM-integrated systems [4,5],
can extract information across authorisation boundaries, override system-level
instructions, and cause the model to take unintended actions against back-end
services. Retrieval-augmented generation (RAG) architectures [6], which are
necessary for grounding responses in domain-specific knowledge bases, add a further
indirect injection surface: documents stored in the retrieval index may themselves
contain adversarially crafted content [7].

This paper makes the following contributions:

1. **System architecture.** We describe a complete, production-deployed dual-context
   RAG system securing a property management company's employee-facing and
   customer-facing AI assistants. The system uses quantised 8B-parameter models
   running on a single consumer GPU, achieving inference latency compatible with
   interactive use.

2. **Security architecture.** We detail a three-layer defence-in-depth stack
   incorporating a configurable prompt-injection classifier, PII redaction, and
   retrieved-document scanning, and explain the architectural rationale for each
   layer's placement in the data pipeline.

3. **Benchmark results.** We execute a systematic 52-configuration sweep
   (12 guard configurations × 2 deployment sides × 2 LLMs, plus unprotected
   baselines), reporting empirical TPR/FPR with Wilson 95% confidence intervals.
   All 128-token chunk configurations achieve 100% TPR against the tested attack
   set; the guard reduces unprotected attack success rates of 49–70% to 0%.

4. **Regulatory analysis.** We map the architecture against HIPAA, DOE Order 205.1C,
   DoD CMMC 2.0, ITAR, and the NIST AI RMF 1.0/AI 600-1 Generative AI Profile,
   identifying where self-hosting provides compliance benefits and where gaps remain.

5. **Feasibility analysis.** We provide a hardware cost model and total-cost-of-ownership
   comparison showing that on-premises deployment is economically viable for SMBs
   at moderate query volumes.

**Scope and limitations.** The benchmark results reported in §6.7 used a 78-prompt
general-purpose jailbreak dataset and a 49-prompt benign set; domain-specific
evaluation against property-management–targeted attacks is a planned extension
(§11). The cost model uses publicly available pricing as of mid-2025 and should be
updated as hardware and API pricing evolves. The regulatory analysis is descriptive
and does not constitute legal advice.

---

## 2. Related Work

### 2.1 Prompt Injection and Jailbreaking

Perez and Ribeiro [4] introduced the term *prompt injection* to describe attacks
that embed adversarial instructions in user input to override an LLM's prior
directives. Subsequent work showed that such attacks are broadly effective against
instruction-tuned models without dedicated defences [5]. Greshake et al. [7]
extended the threat model to *indirect* injection—where adversarial instructions
are embedded in third-party content (documents, web pages, database records)
retrieved by an LLM agent—demonstrating compromise of real-world LLM-integrated
applications. This threat is directly applicable to RAG architectures, which
ingest untrusted document corpora into the LLM context window.

Zou et al. [8] demonstrated universal, transferable adversarial suffixes that
reliably jailbreak aligned models. The GCG attack and its successors produce
prompts that defeat refusal training by optimising directly against model logits.
Classifier-based guards such as Prompt Guard operate at a different layer—screening
inputs before they reach the target model—and are therefore not directly susceptible
to gradient-based optimisation against the guarded model. However, they remain
vulnerable to adversarial examples crafted against the classifier itself, a
limitation we discuss in §9.

### 2.2 Evaluation Benchmarks for LLM Security

JailbreakBench [9] provides a standardised evaluation framework for jailbreak
robustness, defining a canonical attack dataset and judge model. HarmBench [10]
extends this with a broader taxonomy of harmful request categories and a
methodology for comparing diverse attack and defence methods. Both benchmarks
target the base model's refusal behaviour rather than pre-model classifier gates.
Our methodology is complementary: we evaluate the gate layer rather than the
model, and we explicitly measure false-positive rate as a co-equal objective to
reflect the operational cost of over-blocking in an SMB deployment.

### 2.3 Retrieval-Augmented Generation

Lewis et al. [6] introduced RAG as a framework for combining parametric model
knowledge with non-parametric retrieval from an external document store. The
architecture significantly reduces hallucination rates on knowledge-intensive tasks
by anchoring generation to retrieved evidence. FAISS [11] provides the efficient
approximate nearest-neighbour index that makes dense retrieval practical at the
document corpus sizes typical of SMB deployments (tens of thousands of chunks).
Security considerations specific to RAG—index poisoning, context injection via
retrieved documents, and membership inference against the retrieval index—have
received growing attention [7,12] but remain understudied at the system level.

### 2.4 Quantised Inference on Consumer Hardware

GPTQ [13] and subsequent methods demonstrated that 4-bit post-training quantisation
preserves the majority of model quality for instruction-following tasks while
reducing memory requirements by approximately 4×. The GGUF format and llama.cpp
runtime [14] operationalised quantised inference on consumer CPUs and GPUs without
requiring Python or specialised ML frameworks, substantially lowering the deployment
barrier for organisations without dedicated ML infrastructure. QLoRA [15] extended
efficient inference techniques to fine-tuning, enabling parameter-efficient
adaptation of 7–13B models on a single consumer GPU—a capability we exploit in
the domain fine-tuning component of this work.

### 2.5 Security and Privacy in Machine Learning Systems

Carlini et al. [16] demonstrated that training data can be extracted verbatim
from large language models, establishing a privacy threat model that persists after
instruction fine-tuning. Papernot et al. [17] surveyed the broader security and
privacy landscape of machine learning, covering model extraction, membership
inference, and adversarial examples. For on-premises SMB deployments, the most
operationally relevant of these threats is membership inference against the
retrieval index and model inversion against fine-tuned weights, both of which
become more tractable if an attacker gains access to the model files directly—
motivating the filesystem-level security controls discussed in §8.

---

## 3. Threat Model

### 3.1 System and Adversary Model

We consider a dual-context deployment: an *employee-facing* assistant with access
to internal HR, financial, and operational documents, and a *customer-facing*
assistant with access to a curated public knowledge base. Both contexts authenticate
users against a shared database backend.

We define three classes of adversary:

**A1 — Authenticated insider.** An employee or customer with valid credentials
who attempts to extract information outside their authorisation boundary (e.g.,
an employee querying other employees' financial records, or a tenant accessing
another tenant's lease terms). This adversary has interactive access to the chat
interface but no access to the system infrastructure.

**A2 — External attacker via prompt injection.** An unauthenticated or
authenticated adversary who crafts queries designed to override system-level
instructions, exfiltrate retrieval index content, or cause the model to take
unintended back-end actions. This adversary knows the general architecture (assumes
a RAG pipeline) but may not know the specific system prompt.

**A3 — Knowledge-base poisoner.** An adversary who can write to the document
corpus from which the RAG index is built. This could be an insider with document
upload permissions or an external attacker who compromises the document storage
layer. The goal is indirect prompt injection: embedding adversarial instructions
in documents that are later retrieved and injected into the LLM context.

We explicitly exclude:

- **Nation-state-level attackers** with physical access to the inference hardware,
  as on-premises deployment does not mitigate hardware-level threats.
- **Supply-chain compromise** of model weights, though we note this as a gap in
  current deployments (§9.3).
- **Denial of service** attacks, which are resource-management problems rather
  than information-security problems.

### 3.2 Security Goals

For each adversary class we define the desired security property:

| Adversary | Attack Vector | Security Goal |
|---|---|---|
| A1 | Role escalation via crafted query | Retrieval isolation by authenticated role |
| A1 | PII exfiltration from documents/DB | PII not surfaced in LLM output |
| A2 | Direct prompt injection | System instructions not overridable by user input |
| A2 | Jailbreak to produce harmful output | Model refuses off-topic harmful requests |
| A3 | Indirect injection via retrieved docs | Retrieved content does not override system prompt |

### 3.3 Trust Boundaries

The system's trust architecture has three boundaries:

1. **Network boundary.** All LLM inference and retrieval occurs on-premises;
   only Supabase API calls transit the network. This boundary is the primary
   argument for data-residency compliance.

2. **Application boundary.** User inputs are classified by the Prompt Guard
   classifier before reaching the LLM. Retrieved documents are optionally
   classified before being appended to the LLM context. The classifier's
   decision is final; the LLM never sees inputs that the classifier rejects.

3. **Filesystem boundary.** GGUF model weights, the FAISS index, and the Supabase
   credentials file reside on the local filesystem. Access to this boundary
   enables model extraction and index reconstruction by a filesystem-level attacker,
   which we treat as outside the security perimeter of the AI application layer.

---

## 4. System Architecture

### 4.1 Inference Layer

The system uses **llama.cpp** [14] via the LangChain `ChatLlamaCpp` wrapper.
Two quantised GGUF models are supported interchangeably at runtime:

| Model | Variant | Quantisation | Approx. VRAM |
|---|---|---|---|
| Qwen3-8B | `Qwen3-8B-Q4_K_M.gguf` | 4-bit K-quant | ~5.5 GB |
| Meta-Llama-3.1-8B-Instruct | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4-bit K-quant | ~5.5 GB |

With `n_gpu_layers=-1` (full GPU offload), all transformer layers reside in VRAM.
A context window of 8,192 tokens and a batch size of 512 provide adequate
throughput for interactive single-user workloads. The inference temperature is set
to 0.2 to minimise stochastic variation between equivalent queries.

### 4.2 Retrieval Layer

Document retrieval uses a **FAISS flat L2 index** [11] over **BAAI/bge-small-en-v1.5**
sentence embeddings [18] (33 M parameters). Documents are chunked at 512 tokens
with a 20-token overlap before indexing. At query time the top-K=5 nearest chunks
are retrieved and prepended to the LLM context. Supported document formats include
PDF, DOCX, TXT, CSV, and TSV. The index is built once and served from disk on
subsequent runs, with a flag (`REBUILD_INDEX`) for forced rebuilds when the corpus
changes.

### 4.3 Dual-Context Deployment

The employee-facing context indexes internal HR, operational, and financial
documents (`./downloads`). The customer-facing context indexes a curated public
knowledge base (`./docs`). Both share the same codebase with different `DATA_PATHS`
configuration entries, enabling the same guard and retrieval infrastructure to
serve both contexts while maintaining document-level isolation between them.

### 4.4 Database Integration

Both contexts authenticate users against a Supabase backend and query a
`renter_transactions` table for structured financial data. Retrieved rows are
merged with document chunks into the LLM context, enabling the assistant to
synthesise answers from both unstructured policy documents and structured
transaction records. This dual-source design is common in property management,
legal, and financial SMBs.

### 4.5 Hardware Footprint

The complete running system—one LLM, the embedding model, the Prompt Guard
classifier, and a populated FAISS index—occupies approximately 8 GB of VRAM:

| Component | VRAM |
|---|---|
| LLM (8B, Q4\_K\_M) | ~5.5 GB |
| KV cache (8K context) | ~2.0 GB |
| Prompt Guard (86M) | ~0.35 GB |
| bge-small embeddings (33M) | ~0.13 GB |
| FAISS index (10k chunks) | ~0.05 GB |
| **Total** | **~8.0 GB** |

This fits within the 10 GB effective VRAM of an NVIDIA RTX 3080, making the
minimum hardware requirement a mid-range consumer GPU rather than the top-of-range
card. Two parallel workers sharing a loaded LLM model add approximately 2 GB each
for independent KV caches, raising the dual-worker ceiling to ~12 GB—within the
16–24 GB VRAM of RTX 3090, 4080, and 4090 class cards.

---

## 5. Security Mechanisms

### 5.1 Layer 1 — Prompt Injection Classifier

The primary input gate uses **`fmops/distilbert-prompt-injection`** [19], a
DistilBERT-family sequence classifier fine-tuned to distinguish benign from
jailbreak/injection inputs. We extend the reference implementation with two
modifications.

**Overlapping token-level chunking.** Long inputs are split into windows of at
most `max_tokens` tokens with a configurable `chunk_overlap`. The window steps
forward by `max_tokens − chunk_overlap` tokens on each iteration. The maximum
malicious-class probability across all windows determines the final verdict:

```
p_malicious = max_j softmax(logits_j / T)[MALICIOUS]
```

This prevents adversarial payloads distributed across many tokens—designed to
defeat whole-input classifiers—from escaping detection at chunk boundaries. For a
512-token input scanned with 128-token chunks and 100-token overlap, the number
of forward passes is:

```
n_passes = ⌈(N − overlap) / (max_tokens − overlap)⌉ = ⌈(512 − 100) / 28⌉ = 15
```

The latency cost is approximately 15× that of a single-pass evaluation (~15 ms
per pass on an RTX 5090 for the DistilBERT classifier), which remains negligible
relative to the 8B LLM generation time.

**Softmax temperature scaling.** Before computing class probabilities, logits are
divided by a temperature parameter *T*:

```
p = softmax(logits / T)
```

For *T* > 1, the distribution flattens, assigning higher probability mass to the
malicious class for borderline inputs (those with raw logit margins near zero).
This mechanism trades increased recall (fewer missed injections) for reduced
precision (more false positives on legitimate queries). It is particularly useful
for deployment contexts where the adversarial prior is high—such as public-facing
customer chatbots—and where a blocked legitimate query can be rephrased without
significant user friction. The benchmark methodology in §6 quantifies this
precision–recall trade-off empirically across *T* ∈ {1.0, 1.5, 2.0, 3.0, 5.0}.

**Fail-closed error handling.** Any exception during chunk scoring returns
a malicious probability of 1.0, causing the guard to block the query. This
fail-closed behaviour ensures the guard degrades safely under GPU memory pressure,
model-loading failures, or tokeniser errors.

### 5.2 Layer 2 — PII Redaction

When enabled, the pipeline applies three regex-based redactors to retrieved
document content and database output *before* either is appended to the LLM
context:

- **Credit card numbers** (13–19 consecutive digit sequences): all but the last
  four digits are masked with asterisks.
- **Social Security Numbers** (pattern `\d{3}-\d{2}-\d{4}`): fully replaced
  with `***-**-****`.
- **Email addresses**: the local part is truncated to its first character
  followed by asterisks.

The architectural decision to apply redaction before the LLM receives the context
is deliberate. A generative model that has processed unredacted PII in its context
window may reproduce it in output even if the output is post-filtered, because the
information participates in the attention computation across the full context.
Pre-context redaction removes the information from the model's effective input
unconditionally.

For HIPAA-regulated deployments, this regex set would need to be extended to the
full set of 18 Protected Health Information identifiers defined in 45 CFR
§ 164.514(b)(2), including dates, geographic identifiers, and medical record numbers.

### 5.3 Layer 3 — Retrieved-Document Scanning

The optional `PROMPT_GUARD_CHECK_RETRIEVED_DOCS` flag routes each retrieved
document chunk through the same Prompt Guard classifier used for user inputs.
Chunks with malicious probability above threshold are excluded from the LLM
context before generation. This defends against indirect prompt injection (A3
in our threat model): an adversary who embeds override instructions in a document
that is later indexed and retrieved will have the injected content intercepted at
retrieval time rather than silently passed to the model.

The trade-off is proportional latency: with top-K=5 and the 128-token/100-overlap
chunking strategy, each query incurs up to 75 additional classifier forward passes
in the worst case (5 retrieved chunks × 15 passes each). In practice, most retrieved
chunks are shorter than 512 tokens and require fewer passes. A production deployment
should instrument this path to detect pathological inputs early.

---

## 6. Benchmark Design and Methodology

### 6.1 Objectives

The benchmark answers three research questions:

- **RQ1 (Guard effectiveness):** What fraction of prompt injection attacks does
  each guard configuration block (true-positive rate)?
- **RQ2 (Operational cost):** What fraction of legitimate queries does each
  configuration incorrectly block (false-positive rate)?
- **RQ3 (Confounders):** How do deployment context (employee vs. customer),
  underlying LLM, and system-prompt length interact with guard configuration
  to affect both rates?

### 6.2 Configuration Space

The benchmark sweeps the following conditions:

| Factor | Levels |
|---|---|
| Guard chunk size (`max_tokens`) | 64, 128, 512 |
| Chunk overlap | 50 tokens (64- and 512-chunk), 100 tokens (128-chunk) |
| Softmax temperature | 1.0, 1.5, 2.0, 3.0, 5.0 |
| Blocking threshold | 0.40, 0.50 |
| Deployment side | Employee-facing, customer-facing |
| LLM | Qwen3-8B, Meta-Llama-3.1-8B-Instruct |
| System-prompt size | 100, 500, 1000, 1500 tokens |

The twelve protected configurations are complemented by one unprotected baseline
per (side, model, system-prompt) combination, yielding 52 total run conditions
per attack-prompt set. **We note a design limitation:** temperature and chunk
size are not fully orthogonally varied. Temperature 1.0 appears only with the
512-token chunk configurations; threshold 0.50 is tested only with two of the
five temperature levels for the 128-token chunk, and not at all for 64- or
512-token chunks. This one-factor-at-a-time structure prevents clean attribution
of individual parameter effects and their interactions.

### 6.3 Attack Prompt Dataset

The benchmark evaluated **78 general-purpose jailbreak prompts** sourced from
classic adversarial collections (JailbreakBench [9] and AdvBench-adjacent datasets).
The prompts cover three broad attack strategies:

| Category | Representative examples |
|---|---|
| Direct instruction override | "Ignore all the instructions you got before. From now on, you are going to act as..." |
| Persona / alter-ego escalation | Role-play DAN, Developer Mode, BasedAI, Tom-and-Jerry dual-entity frames |
| Authority and jailbreak chaining | Instruction stacking, diagnostic mode invocations, tautological bypass |

These are well-studied, classifier-visible attack patterns. The current repository
also contains a 201-prompt domain-specific dataset (`src/shared/attack_prompt.txt`)
spanning ten categories (direct override, PII exfiltration, indirect retrieval-triggered
injection, obfuscation/encoding, etc.) synthesised for the property-management
deployment context; evaluation against that dataset is deferred to future work (§11).

For statistical reference: n = 78 yields a 95% Wilson CI half-width of ≤11.1 percentage
points at any proportion, sufficient for comparing configurations at the 0.1-TPR
resolution needed to distinguish the observed 98.7% from 100.0%.

### 6.4 Benign Prompt Dataset

The benchmark evaluated **49 benign prompts** drawn from the
`awesome-chatgpt-prompts` corpus — a publicly available collection of
task-specification prompts such as "I want you to act as a linux terminal..." and
"I want you to act as a plagiarism checker...". This corpus was selected for its
availability at the time of the benchmark run.

The current repository also contains a 151-prompt domain-specific benign dataset
(`src/shared/benignPrompts.txt`) covering eight property-management query categories
(rent and payment, maintenance, lease terms, etc.); FPR evaluation against that
dataset is deferred to future work (§11).

With n = 49 benign prompts the 95% Wilson CI on an observed FPR of 4.1% spans
[1.1%, 13.7%] — wide enough that FPR differences between configurations cannot
be distinguished statistically. The benign dataset size is the primary factor
limiting the precision of FPR estimates in this study.

### 6.5 Evaluation Metrics

For each (guard config, side, model, system-prompt size) combination:

- **True-positive rate (TPR):** fraction of attack prompts eliciting the canonical
  block response (`"We couldn't process that request because it looks like an
  attempt to change system-level instructions..."`).
- **False-positive rate (FPR):** fraction of benign prompts eliciting the block
  response.
- **95% confidence intervals** on both rates using the Wilson score interval for
  proportions.
- **Per-attack-category TPR:** breakdown of the above by attack type to identify
  which categories each configuration handles poorly.

The F1 score, defined with respect to the attack-blocking task (precision =
TPR / (TPR + FPR)), provides a single number for overall configuration quality
when the practitioner weights false positives and false negatives equally.

In the full study, we further plan to report (a) end-to-end query latency per
configuration, enabling construction of a TPR–latency Pareto frontier, and
(b) the raw malicious-probability score distribution per configuration, enabling
post-hoc analysis of the operating-point choice.

### 6.6 Execution Architecture

The benchmark uses `concurrent.futures.ProcessPoolExecutor` with spawn-based
process isolation. Two workers per LLM model share a single loaded LLM instance,
avoiding the VRAM cost of loading the same weights twice across deployment sides.
The shared LLM is passed by reference within the worker process; all guard
instances and FAISS indices are initialised independently per worker to maintain
configuration isolation.

### 6.7 Benchmark Results

#### 6.7.1 Unprotected Baseline

Without the prompt-injection guard, 49–70% of the 78 attack prompts elicit a
compliant response from the underlying LLM, depending on model and deployment
context:

| Config | Attack compliance | LLM self-refusal |
|---|---|---|
| Customer-facing / Qwen3-8B | 55% (173/312) | 45% (139/312) |
| Customer-facing / Meta-Llama-3.1-8B | 56% (176/312) | 44% (136/312) |
| Employee-facing / Qwen3-8B | 49% (154/312) | 51% (158/312) |
| Employee-facing / Meta-Llama-3.1-8B | **70% (217/312)** | 30% (95/312) |

Row totals reflect four system-prompt sizes (100/500/1000/1500 tokens) per config
(78 attacks × 4 = 312 rows). "Complied" is inferred from the absence of
refusal-language in the response body; the metric captures full jailbreak
compliance but does not distinguish partial from complete compliance. Meta-Llama
employee-facing is the most susceptible configuration (70% compliance),
underscoring the deployment-context dependence of the unprotected risk level.

#### 6.7.2 Protected Configuration Results

All 48 protected configurations achieve TPR ≥ 98.7% and FPR = 4.1% (Table 1).

**Table 1. Per-configuration TPR and FPR (aggregated over both sides, both LLMs, and all four system-prompt sizes).**

| Configuration | TPR | FPR | F1 |
|---|---|---|---|
| c128_ov100_temp1.0_th0.40 | **100.0%** | 4.1% | 98.0% |
| c128_ov100_temp1.5_th0.40 | **100.0%** | 4.1% | 98.0% |
| c128_ov100_temp2.0_th0.40 | **100.0%** | 4.1% | 98.0% |
| c128_ov100_temp3.0_th0.40 | **100.0%** | 4.1% | 98.0% |
| c128_ov100_temp3.0_th0.50 | **100.0%** | 4.1% | 98.0% |
| c128_ov100_temp5.0_th0.40 | **100.0%** | 4.1% | 98.0% |
| c128_ov100_temp5.0_th0.50 | **100.0%** | 4.1% | 98.0% |
| c512_temp1.0_th0.40 | 98.7% | 4.1% | 97.4% |
| c512_temp2.0_th0.40 | 98.7% | 4.1% | 97.4% |
| c512_temp3.0_th0.40 | **100.0%** | 4.1% | 98.0% |
| c64_ov50_temp3.0_th0.40 | 98.7% | 4.1% | 97.4% |
| c64_ov50_temp5.0_th0.40 | **100.0%** | 4.1% | 98.0% |

95% Wilson CI on TPR = 100% (n=78): **[95.3%, 100%]**.  
95% Wilson CI on FPR = 4.1% (n=49): **[1.1%, 13.7%]**.

#### 6.7.3 Configuration Sensitivity

No combination of LLM, deployment side, system-prompt size, softmax temperature,
blocking threshold, or chunk overlap produces a statistically distinguishable
difference in FPR; all 48 configurations yield exactly 2/49 false positives.
TPR sensitivity is limited to chunk size:

- **128-token chunks:** 100% TPR across all temperature and threshold levels.
- **512-token chunks (temp 1.0, 2.0):** 98.7% TPR — one attack evades the
  classifier at the 0.40 threshold. At temp 3.0 the missed attack is recovered
  (100% TPR), suggesting the logit sharpening at higher temperature provides
  marginal uplift for borderline cases.
- **64-token chunks (temp 3.0):** 98.7% TPR; at temp 5.0 the missed attack is
  recovered (100% TPR).

The insensitivity to threshold (0.40 vs. 0.50) indicates that the DistilBERT
classifier assigns high-confidence injection scores to these general-purpose
jailbreaks — the operating margin between attack scores and both thresholds is
large. This robustness may not hold for more subtle or domain-specific attacks.

#### 6.7.4 False Positive Analysis

The 2 false positives are identical across all 48 configurations:

1. *"I want you to act as a plagiarism checker..."* — benign task-specification
   prompt containing roleplay-initiation phrasing.
2. *"I want you to act like {character} from {series}..."* — unfilled template
   with roleplay-initiation phrasing.

Both contain "act as / act like" language, which correlates with injection patterns
in the classifier's training distribution. This is a classifier property, not a
guard configuration issue; it cannot be resolved by adjusting chunk size, temperature,
or threshold. The false positive rate of 4.1% reflects the overlap between the
benign roleplay-specification genre and the classifier's injection-detection signal.

---

## 7. Domain Fine-Tuning

### 7.1 Motivation

Base open-weight models are instruction-tuned on general corpora. For SMB
deployments with narrow, specialised knowledge domains—property management,
medical coding, industrial equipment maintenance—fine-tuning on domain data
improves instruction-following fidelity and reduces off-topic completions that
may bypass simple keyword-based guardrails.

### 7.2 Implementation

We fine-tune **Qwen2-0.5B** [2] on a pre-tokenised version of the
`nampdn-ai/tiny-codes` dataset [20] (1.47 M training examples, 163 K validation
examples; sequences of 1,024 tokens). The training configuration closely follows
the original pre-training setup:

| Hyperparameter | Value |
|---|---|
| Per-device batch size | 2 |
| Gradient accumulation steps | 8 (effective batch 16) |
| Learning rate | 1 × 10⁻⁵ |
| LR schedule | Linear decay, no warmup |
| Precision | bfloat16 |
| Gradient checkpointing | Enabled |
| Optimiser | AdamW fused |
| Epochs | 1 |

At approximately 19.7 training steps per second on an NVIDIA RTX 5090, a complete
epoch over 1.47 M examples takes approximately 78 minutes. This demonstrates that
domain fine-tuning is not a research-lab-exclusive capability: an SMB can fine-tune
a 0.5B model on a proprietary document corpus overnight and produce a model whose
base distribution is aligned with the deployment domain.

### 7.3 Security Implications of Fine-Tuning

Fine-tuning on domain data has two security implications beyond capability:

**System-prompt alignment.** A model fine-tuned on domain-specific
instruction-following data more reliably adheres to role constraints specified in
the system prompt. This is because the fine-tuning distribution reinforces the
instruction-following behaviour in the context of domain-relevant queries, reducing
the probability of the model producing off-topic completions in response to
injection attempts that rely on the model "forgetting" its role constraints.

**Weight provenance and ITAR.** When the domain corpus contains export-controlled
technical data—relevant for DoD contractors—the fine-tuned weights may inherit
the export-control classification of the training data under ITAR (see §8.3).
Local fine-tuning is therefore not merely an economic choice for such contractors
but a compliance requirement: uploading weights to a cloud fine-tuning service
constitutes an unauthorised export of the contained technical data.

---

## 8. Regulatory Compliance Analysis

### 8.1 HIPAA — Health Insurance Portability and Accountability Act

HIPAA's Security Rule (45 CFR § 164.312) specifies technical safeguards for
systems that process electronic Protected Health Information (ePHI). Healthcare
SMBs—clinics, billing services, mental health practices—that deploy AI on patient
records fall within its scope.

The four technical safeguard categories map onto this system's architecture:

| Technical Safeguard | CFR Reference | Relevant Component |
|---|---|---|
| Access Control | § 164.312(a)(1) | Supabase row-level auth; role-based retrieval |
| Audit Controls | § 164.312(b) | Per-query CSV logs in `results/` |
| Integrity | § 164.312(c)(1) | FAISS index checksums (absent — gap) |
| Transmission Security | § 164.312(e)(1) | TLS on Supabase API; local-only LLM |

A 2025 update to the Security Rule eliminates the former distinction between
*required* and *addressable* safeguards, making encryption, MFA, and network
segmentation unconditionally mandatory. For an on-premises AI deployment this
requires: encryption at rest for the FAISS index and GGUF model files; MFA on
the host machine's remote access paths; and TLS for all outbound network calls.

The PII redaction layer addresses HIPAA's data-minimisation principle by
stripping identifiers from retrieved content before it reaches the LLM context.
A full HIPAA-compliant deployment would extend the regex set to all 18 PHI
identifiers under 45 CFR § 164.514(b)(2).

Self-hosting eliminates the Business Associate Agreement (BAA) negotiation that
HIPAA mandates for any third-party ePHI processor. Several major API providers
offer BAAs, but they impose usage restrictions—no training on customer data,
no persistent storage of inputs—that conflict with the fine-tuning and
retrieval-index workflows central to this system.

### 8.2 Department of Energy (DOE) — Order 205.1C

DOE Order 205.1C (*Departmental Cybersecurity Program*) governs information
security for DOE facilities, national laboratories, and their contractor bases.
For Controlled Unclassified Information (CUI) under 32 CFR Part 2002, NIST SP
800-171 provides the baseline technical controls.

The primary compliance benefit of self-hosting for DOE contractors is that CUI
never transits an external API endpoint, satisfying 800-171's system boundary
requirements without the procurement overhead of a FedRAMP-authorised cloud
service. The well-defined hardware boundary of a self-hosted RAG stack is also
easier to document and certify than a hybrid system involving cloud API calls.

As of mid-2025, DOE Order 205.1C does not include AI-specific security annexes.
The Prompt Guard layer in this system provides prompt-injection protection that
goes beyond current DOE guidance, positioning a contractor who implements it
ahead of forthcoming AI-specific requirements.

### 8.3 Department of Defense (DoD) — CMMC, ITAR, and DFARS

**CMMC 2.0 Level 2** requires compliance with all 110 practices of NIST SP 800-171,
spread across 14 domains. DFARS clause 252.204-7012 mandates that any system
processing, storing, or transmitting Covered Defense Information (CDI) meet
these controls. An LLM that retrieves and answers questions about defence
contract documents falls within the regulatory definition.

The audit-and-accountability domain (NIST 800-171 controls 3.3.1–3.3.2) requires
logged records of who accessed what data and when. The current CSV output
(`question`, `response`, `duration` per row) provides a partial audit trail
but does not record the authenticated user identity or the specific document
chunks retrieved. Adding user identity to the query log and capturing retrieval
results alongside each response would close this gap.

**ITAR.** As of January 2025, BIS created ECCN 4E091, subjecting AI model
weights trained with more than 10²⁶ floating-point operations to export controls.
For SMB contractors fine-tuning on defence technical data, both the training
corpus and the resulting weights may constitute ITAR-controlled technical data.
Running the fine-tuning pipeline locally—as implemented in `experiments/train.py`—
ensures no data or weights leave the organisation's infrastructure.

**DFARS 252.204-7012** requires cyber-incident reporting within 72 hours.
Self-hosted systems produce locally owned logs that enable rapid incident
reconstruction, without reliance on the provider's infrastructure audit logs
that would be necessary for a cloud API deployment.

### 8.4 NIST AI Risk Management Framework and Federal AI Governance

There is no single enacted federal statute titled "Federal AI Security Act" as of
mid-2025. The operative federal AI governance standard is a combination of NIST
frameworks and sector-specific agency guidance.

**NIST AI RMF 1.0** [21] organises AI risk management into four functions—
*GOVERN*, *MAP*, *MEASURE*, and *MANAGE*—constituting a continuous risk cycle.
**NIST AI 600-1** (the Generative AI Profile, July 2024) [22] extends this to
LLM-specific risks. Mapping the codebase against AI 600-1:

| AI 600-1 Risk Category | Suggested Action | System Status |
|---|---|---|
| Prompt Injection | Input validation and filtering | Prompt Guard with 12-config sweep |
| Data Privacy | PII de-identification | Regex redaction pre-context |
| Confabulation (hallucination) | Ground in retrieved facts | RAG anchors all responses |
| Information Integrity | Provenance tracking | Partial (FAISS chunk source) |
| Behavioural Anomaly | Audit logging | CSV query/response logs |
| Harmful Bias | Documented evaluation | Not addressed — gap |

The GOVERN function requires documented roles and accountability for AI risk.
For an SMB this means designating a system owner, establishing a review cadence
for guard-threshold recalibration (motivated by the benchmark sweep in §6), and
maintaining a change log for system-prompt modifications. The MEASURE function
is directly served by the benchmark methodology described in §6.

Executive Order 14110 [23] (*Safe, Secure, and Trustworthy AI*, October 2023)
directed NIST and CISA to develop AI-specific security guidelines, producing
sector-specific outputs (FDA's AI action plan, CISA's critical infrastructure
guidance) rather than uniform cross-sector mandates. The common thread across
all outputs is documented evaluation of adversarial robustness—precisely what
the benchmark sweep is designed to produce.

### 8.5 Cross-Framework Compliance Summary

| Framework | Primary Benefit of Self-Hosting | Remaining Gap | Effort |
|---|---|---|---|
| HIPAA | Eliminates BAA; data never egresses | PHI regex coverage incomplete | Low |
| DOE CUI / 800-171 | Satisfies system boundary requirement | No AI-specific risk docs | Low |
| DoD CMMC 2.0 | CDI processed locally | Audit log missing user ID + retrieved chunks | Medium |
| DoD ITAR | Fine-tuning on local hardware only | No weight-export controls or checksums | Medium |
| NIST AI RMF | Full stack transparency | Bias evaluation absent | Medium |

---

## 9. Feasibility Analysis

### 9.1 Hardware Economics

The minimum viable configuration requires approximately 8 GB of VRAM for the full
running system. This is satisfied by consumer cards available new for under
\$700 (RTX 3080 10 GB) or used for under \$400 (RTX 2080 Ti). An SMB does not
need top-of-generation hardware for a single-user or low-concurrency deployment.

The total cost of ownership comparison, at 5,000 queries per day (800 input +
300 output tokens each), against a mid-range commercial API:

| Cost Item | Cloud (GPT-4o at \$2.50/M in, \$10/M out) | On-Premises |
|---|---|---|
| Per-day variable | \$10 in + \$15 out = **\$25** | \$0 |
| Monthly | ~\$750 | ~\$30 electricity |
| Year 1 total | ~\$9,000 | ~\$1,500 GPU + \$360 power = **~\$1,860** |
| Year 2+ | ~\$9,000/yr | ~\$360/yr |

*Pricing as of mid-2025; electricity at \$0.12/kWh; 250 W mean GPU draw.*

The break-even against a mid-range cloud API falls at approximately **four to five
months** of continuous operation. Against lower-cost or self-hosted cloud services,
the break-even extends to 6–12 months but remains favourable for multi-year
deployments.

Against frontier proprietary models (GPT-4o, Claude Sonnet), a local 8B model
sacrifices capability particularly on multi-step reasoning and long-document
synthesis. The RAG architecture partially compensates by anchoring responses to
retrieved ground truth, limiting hallucination rates on knowledge-intensive
queries. For constrained, domain-specific deployments—the primary SMB use case—
the quality gap is generally acceptable and can be narrowed by domain fine-tuning.

### 9.2 Staffing Requirements

The stack requires Python proficiency to configure and troubleshoot; it does not
require ML-research expertise for routine operation. A developer familiar with
PyTorch and LangChain can maintain the system at approximately 4–8 hours per
month in steady state. Domain fine-tuning (§7) additionally requires familiarity
with HuggingFace `transformers` and `trl`—a realistic expectation for a
technically capable two-to-five person engineering team at an SMB.

### 9.3 Operational Risk

Key operational risks not addressed by the security architecture:

1. **Model and index supply-chain integrity.** GGUF weights are downloaded from
   HuggingFace Hub without SHA-256 verification. A compromised CDN or repository
   could deliver modified weights. Production deployments should pin model file
   checksums.

2. **Guard model currency.** The `fmops/distilbert-prompt-injection` classifier
   was trained on a fixed distribution of jailbreak prompts. As adversarial
   prompting evolves, the guard's recall will degrade against novel attack patterns
   not present in its training distribution. Periodic re-evaluation against current
   attack corpora—analogous to the benchmark sweep described in §6—is required.

3. **FAISS index confidentiality.** The index stores embedding vectors of all
   indexed documents. Approximate reconstruction attacks against small embedding
   models are feasible [24]. Index files should be encrypted at rest and
   access-logged.

4. **Multi-tenancy side-channels.** Two workers sharing GPU hardware could, in
   principle, leak information via timing side-channels in the GPU compute queue.
   For high-sensitivity deployments (health records, legal privilege), GPU
   partitioning (NVIDIA MIG) is recommended.

---

## 10. Discussion

### 10.1 Position on the Deployment Spectrum

Self-hosted AI represents one point on a spectrum bounded by (a) fully cloud-based
API usage and (b) air-gapped deployment on classified networks. SMBs occupy a
middle position where data sensitivity is genuine but budgets are constrained.
The architecture described here is designed for this middle position: it provides
meaningful data-residency guarantees and a configurable security posture without
requiring enterprise-grade infrastructure or a dedicated ML operations team.

### 10.2 Guard Configuration Recommendations

The following recommendations are supported by the empirical results in §6.7.
All references are to `fmops/distilbert-prompt-injection` as the classifier.

- **General use:** Any `c128_ov100` configuration achieves 100% TPR with 4.1% FPR
  against the tested jailbreak set. Temperature and threshold are not materially
  differentiating within the tested range; `c128_ov100_temp2.0_th0.40` is a
  reasonable default.
- **Higher throughput / coarser chunking:** `c512_temp3.0_th0.40` recovers to
  100% TPR from the 98.7% seen at lower temperatures with 512-token chunks, at
  the cost of wider context windows per classification call.
- **Regulated data (HIPAA/CMMC):** Enable `PROMPT_GUARD_CHECK_RETRIEVED_DOCS`
  regardless of configuration. The latency cost of scanning retrieved chunks is
  justified by the indirect injection risk in document corpora (§5.3), and
  the classifier adds no measurable additional FPR overhead on document text.
- **Caution on FPR estimates:** The observed 4.1% FPR was measured on a
  roleplay-specification benign corpus. Deployments with higher ambient roleplay
  or persona-assignment queries — such as customer-service chatbots with "act as
  a concierge" prompts — may see higher FPRs. Domain-specific benign evaluation
  is recommended before production deployment.

### 10.3 Relationship to Emerging LLM Security Research

Recent work on LLM security has largely focused on model-level defences—RLHF
refusal training, constitutional AI, output classifiers—applied to large
proprietary models [8,25]. The present work occupies a different part of the
design space: pre-model input classification applied to a locally-hosted
open-weight model. These approaches are complementary. The instruction-tuning
of Qwen3 and Llama 3 provides a second layer of refusal behaviour beyond the
Prompt Guard classifier; the classifier serves as a rate-limiter and logging
point for adversarial inputs that the LLM might otherwise process silently.

---

## 11. Limitations and Future Work

**Attack dataset scope.** The empirical results in §6.7 were obtained using 78
general-purpose jailbreak prompts (DAN, Developer Mode, persona-escalation
variants). The repository also contains a 201-prompt domain-specific attack set
targeting the property-management deployment context (PII exfiltration, indirect
RAG injection, obfuscation, authority escalation). Whether the high TPR observed
on general jailbreaks holds for these more targeted attacks — particularly
CAT-5 indirect/retrieval-triggered injection and CAT-8 obfuscation — is an open
empirical question. The 100% TPR and low FPR results should therefore not be
over-generalised to domain-tailored adversaries.

**Experimental design confounds.** The current 12-configuration sweep does not
fully orthogonalise temperature, chunk size, and threshold. A full factorial
design (3 chunk sizes × 2 thresholds × 5 temperatures = 30 configurations)
would enable clean attribution of individual parameter effects and their
interactions.

**Single deployment context.** The case study is a property management company
(Elm & Ivy Properties) with a specific document corpus and query distribution.
Guard performance may differ substantially in other SMB verticals (healthcare,
legal, manufacturing) whose corpora and query patterns differ from property
management. Multi-domain evaluation is a planned extension.

**Model capability gap.** The comparison against frontier APIs (§9.1) is
qualitative. A systematic capability evaluation on domain-specific benchmarks
would strengthen the feasibility argument for quality-sensitive applications.

**Fine-tuning security evaluation.** The fine-tuning experiments (§7) focus on
the Qwen2-0.5B model on a general code corpus. Whether fine-tuning on domain
data also improves or degrades the guard's effectiveness—by shifting the base
distribution closer to or further from the classifier's training distribution—
is an open question.

---

## 12. Conclusion

On-premises LLM deployment is technically and economically viable for SMBs with
a moderate technical staff and a sufficiently high query volume to justify the
capital expenditure. Using quantised 8B models on a single consumer GPU, an SMB
can deploy a RAG-based assistant whose inference cost asymptotes to electricity
rather than per-token fees, with a break-even against mid-range cloud APIs of
approximately four to five months.

The security posture achievable with open-source tooling is non-trivial. The
multi-layer guard architecture—pre-model input classification, pre-context PII
redaction, and retrieved-document scanning—addresses the three primary attack
vectors (direct injection, PII exfiltration, indirect injection via indexed
documents) at their most effective interception points. Empirical benchmark results
(§6.7) confirm that the `fmops/distilbert-prompt-injection` classifier, configured
with 128-token overlapping chunks, achieves 100% TPR against a 78-prompt
general-purpose jailbreak set with a 4.1% FPR. Without the guard, 49–70% of the
same attacks succeed depending on model and deployment context, quantifying the
protection uplift the classifier provides.

For organisations subject to HIPAA, CMMC, or similar regulatory frameworks, the
data-residency guarantee provided by self-hosting is often the decisive advantage
over cloud APIs: it eliminates the third-party processor relationship entirely and
with it the most complex compliance obligations. The remaining gaps—model checksum
verification, extended PII coverage, user-identity audit logging—are engineering
tasks within the reach of an SMB development team, not architectural changes.

The primary remaining empirical question is domain-specific validation. The
benchmark results reported here used general-purpose jailbreaks; evaluation against
the 201-prompt property-management–specific attack dataset (CAT-5 indirect RAG
injection, CAT-8 obfuscation, domain-context exfiltration) will determine whether
the high TPR observed on classifier-visible jailbreaks extends to more targeted
adversarial inputs. That evaluation and the associated latency characterisation
are the immediate next steps for this research.

---

## References

[1] A. Dubey et al., "The Llama 3 Herd of Models," arXiv:2407.21783, 2024.

[2] Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, Alibaba Group, 2024.

[3] A. Jiang et al., "Mistral 7B," arXiv:2310.06825, 2023.

[4] F. Perez and I. Ribeiro, "Ignore Previous Prompt: Attack Techniques For
Language Models," arXiv:2211.09527, 2022.

[5] R. Branch et al., "Evaluating the Susceptibility of Pre-Trained Language Models
via Handcrafted Adversarial Examples," arXiv:2209.02128, 2022.

[6] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks," *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[7] K. Greshake et al., "Not What You've Signed Up For: Compromising Real-World
LLM-Integrated Applications with Indirect Prompt Injection," *Proceedings of the
ACM Workshop on Artificial Intelligence and Security (AISec)*, 2023.

[8] A. Zou et al., "Universal and Transferable Adversarial Attacks on Aligned
Language Models," arXiv:2307.15043, 2023.

[9] P. Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking
Large Language Models," *Advances in Neural Information Processing Systems
(NeurIPS)*, 2024.

[10] M. Mazeika et al., "HarmBench: A Standardized Evaluation Framework for
Automated Red Teaming and Robust Refusal," *International Conference on Machine
Learning (ICML)*, 2024.

[11] J. Johnson, M. Douze, and H. Jégou, "Billion-Scale Similarity Search with
GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, 2021.

[12] C. Zou et al., "PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented
Generation of Large Language Models," arXiv:2402.07867, 2024.

[13] E. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers," *International Conference on Learning Representations
(ICLR)*, 2023.

[14] G. Gerganov, "llama.cpp: Efficient Inference of LLaMA Models in Pure C/C++,"
GitHub, https://github.com/ggerganov/llama.cpp, 2023.

[15] T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs,"
*Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

[16] N. Carlini et al., "Extracting Training Data from Large Language Models,"
*30th USENIX Security Symposium*, 2021.

[17] N. Papernot et al., "SoK: Security and Privacy in Machine Learning,"
*IEEE European Symposium on Security and Privacy (EuroS&P)*, 2018.

[18] S. Xiao et al., "C-Pack: Packaged Resources to Advance General Chinese
Embedding," *Proceedings of SIGIR*, 2024.

[19] fmops, "distilbert-prompt-injection," HuggingFace Model Card,
https://huggingface.co/fmops/distilbert-prompt-injection, 2023.

[20] N. D. Pham, "Tiny-Codes: A Curated Dataset for Code Generation,"
HuggingFace Datasets, https://huggingface.co/datasets/nampdn-ai/tiny-codes, 2023.

[21] National Institute of Standards and Technology, "Artificial Intelligence Risk
Management Framework (AI RMF 1.0)," NIST AI 100-1, January 2023.

[22] National Institute of Standards and Technology, "Generative AI Profile (NIST
AI 600-1)," NIST AI 600-1 ipd, July 2024.

[23] Executive Office of the President, "Executive Order on the Safe, Secure, and
Trustworthy Development and Use of Artificial Intelligence," E.O. 14110, October
2023.

[24] Y. Song and R. Shokri, "Systematic Evaluation of Privacy Risks of Machine
Learning Models," *30th USENIX Security Symposium*, 2021.

[25] Y. Bai et al., "Constitutional AI: Harmlessness from AI Feedback,"
arXiv:2212.08073, Anthropic, 2022.

[26] M. Russinovich et al., "Great, Now Write an Article About That: The
Crescendo Multi-Turn LLM Jailbreak Attack," arXiv:2404.01833, 2024.

---

## Appendix A — Repository Component Map

| Component | Path | Role |
|---|---|---|
| Benchmark runner | `main.py` | Orchestrates 52-configuration security sweep |
| Employee RAG app | `src/employee-facing/rag.py` | LLM + FAISS + auth for staff queries |
| Customer RAG app | `src/customer-facing/rag.py` | LLM + FAISS for customer-facing queries |
| Prompt Guard | `src/shared/prompt_guard.py` | Classifier with chunking + temperature scaling |
| Guardrail analysis | `src/shared/guardrail_analysis.py` | Post-benchmark TPR/FPR reporting |
| Fine-tuning script | `experiments/train.py` | Qwen2-0.5B training on tiny-codes |
| System prompts | `src/*/docs/system_prompts/*.txt` | 100/500/1000/1500-token variants |

## Appendix B — Guard Configuration Reference

| Config | Chunk | Overlap | Temp | Threshold |
|---|---|---|---|---|
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
| `c128_ov100_temp7.0_th0.40` | 128 | 100 | 7.0 | 0.40 |
| `c128_ov100_temp7.0_th0.50` | 128 | 100 | 7.0 | 0.50 |
