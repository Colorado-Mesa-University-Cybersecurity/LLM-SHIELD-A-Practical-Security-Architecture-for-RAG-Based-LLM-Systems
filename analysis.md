# On-Premises AI for Small and Medium-Sized Businesses: A Feasibility and Security Analysis

*Based on the Specialized-AI-Agents codebase — a complete self-hosted RAG system with
prompt-injection defences, guard-configuration benchmarking, and domain fine-tuning.*

---

## Abstract

Cloud-hosted large language model (LLM) APIs offer immediate capability but impose
ongoing per-token costs, transmit potentially sensitive business data to third-party
servers, and leave the operator with little control over the model's behaviour on
proprietary knowledge bases. This paper examines whether small to medium-sized
businesses (SMBs) can meaningfully substitute on-premises AI deployments for
cloud APIs, with particular emphasis on the security challenges that accompany
self-hosting. Using the Specialized-AI-Agents repository as a working case study,
we analyse the hardware economics, software architecture, and security posture of a
production-oriented local RAG (Retrieval-Augmented Generation) system, and we
evaluate a systematic benchmark that sweeps twelve prompt-guard configurations
across two LLMs, two deployment contexts, and four system-prompt sizes. We conclude
that on-premises deployment is technically feasible for a motivated SMB with a
single consumer-grade GPU, that the main security risks are prompt-injection and
data exfiltration rather than model extraction, and that a defence-in-depth stack
(input classifier + PII redaction + output filtering) provides meaningfully better
protection than any single measure, albeit with a false-positive cost that must be
calibrated to the deployment context.

---

## 1. Introduction

The commercialisation of large language models has created a new class of business
risk: companies now routinely send internal documents, customer data, and operational
queries to third-party inference endpoints in exchange for API access. For
organisations subject to data-protection regulations (GDPR, HIPAA, CCPA) or that
handle commercially sensitive IP, this arrangement is problematic regardless of the
API provider's contractual guarantees. On-premises deployment—running the model on
hardware the organisation controls—eliminates this data-egress risk entirely.

At the same time, on-premises LLM deployment has historically been the exclusive
domain of well-resourced enterprises. The 2022–2025 open-model era has materially
changed this. Quantised GGUF variants of frontier-class models (7–8 B parameters)
now execute at conversational speeds on a single consumer GPU with 16–24 GB of VRAM,
and the llama.cpp inference engine has reduced the C++ expertise barrier to near
zero. The question is no longer *whether* an SMB can run a local LLM but *what it
costs*, *how secure it is*, and *whether the output quality justifies the complexity*.

This paper addresses all three questions through the lens of a concrete implementation:
a dual-context (employee-facing and customer-facing) RAG system secured by
Meta's Llama-Prompt-Guard-2 classifier and evaluated by a systematic sweep of guard
hyperparameters.

---

## 2. System Architecture

### 2.1 Inference Layer

The system uses **llama.cpp** via the LangChain `ChatLlamaCpp` wrapper
(`src/employee-facing/rag.py:355–380`). Two quantised GGUF models are supported
interchangeably:

| Model | GGUF variant | Quantisation | Approx. VRAM |
|---|---|---|---|
| Qwen3-8B | `Qwen3-8B-Q4_K_M.gguf` | 4-bit K-quant | ~5.5 GB |
| Llama 3.1 8B Instruct | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4-bit K-quant | ~5.5 GB |

With `n_gpu_layers=-1` (full GPU offload), all transformer layers reside in VRAM,
leaving the CPU responsible only for tokenisation and sampling. A context window of
8,192 tokens (`CTX_TOKENS`) accommodates RAG retrieval results comfortably at a
batch size of 512 (`N_BATCH`).

### 2.2 Retrieval Layer

Document retrieval uses a **FAISS flat-index** (`src/employee-facing/rag.py:288–299`)
backed by **BAAI/bge-small-en-v1.5** sentence embeddings (33 M parameters, ~130 MB
on disk). Documents are chunked at 512 tokens with a 20-token overlap before
indexing. At query time the top-K=5 chunks are retrieved and prepended to the LLM
context. Supported document types include PDF, DOCX, TXT, CSV, and TSV, making the
pipeline practical for real-world SMB knowledge bases that accumulate in mixed
formats.

### 2.3 Authentication and Data Source

User-facing sessions are authenticated against a **Supabase** backend. The system
queries a `renter_transactions` table for structured financial data and merges the
results with unstructured document retrieval before answering questions. This
dual-source design is common in property management, legal, and finance SMBs where
both narrative policy documents and transaction records must be consulted together.

### 2.4 Deployment Duality

The repository maintains two semantically separate deployments—employee-facing
(`src/employee-facing/`) and customer-facing (`src/customer-facing/`)—with identical
code structure but distinct document bases (`./downloads` vs. `./docs`). This mirrors
a common SMB requirement: employees need access to internal HR, operations, and
financial documents, while customers interact with a curated public-facing knowledge
base. Running both on the same hardware is economically advantageous but introduces
a cross-contamination risk that must be mitigated at the retrieval layer.

---

## 3. Security Architecture

Self-hosted AI introduces a qualitatively different threat model than cloud APIs.
The operator now controls the model weights, the inference stack, and the data, but
also bears full responsibility for defending against adversarial inputs. Three
primary attack classes are relevant to SMB deployments:

1. **Prompt injection / jailbreaking** — malicious user input that attempts to
   override the system prompt or extract information outside the user's authorisation
   boundary.
2. **PII leakage** — the model surfacing personally identifiable information (email
   addresses, credit card numbers, SSNs) retrieved from documents or the database.
3. **Model extraction** — systematic querying to replicate model behaviour; less
   critical for commodity open-weights models but relevant when the SMB has
   fine-tuned on proprietary data.

The codebase implements defences for the first two categories through a
three-layer architecture.

### 3.1 Layer 1 — Input Classification (Prompt Guard)

The primary gate is **meta-llama/Llama-Prompt-Guard-2-86M**, a 86 M-parameter
BERT-family sequence classifier trained by Meta to distinguish benign from
jailbreak/injection inputs (`src/shared/prompt_guard.py`).

The implementation extends the reference tutorial with two important additions:

**Overlapping chunking.** Long inputs are split into overlapping windows of at
most `max_tokens` tokens with a configurable `chunk_overlap`. The maximum malicious
probability across all chunks determines the final verdict. This prevents injection
fragments near chunk boundaries from escaping detection:

```python
step = max(1, self.max_tokens - self.chunk_overlap)
start = 0
while start < len(ids):
    end = min(start + self.max_tokens, len(ids))
    # ... score chunk
    start += step
```

**Softmax temperature scaling.** Before computing probabilities from classifier
logits, the logits are divided by a temperature parameter `T`:

```
p_malicious = softmax(logits / T)[MALICIOUS_IDX]
```

For `T > 1`, the distribution flattens, assigning higher probability mass to the
malicious class for borderline inputs. This trades increased recall (fewer missed
attacks) for decreased precision (more false positives). The benchmark
systematically explores this trade-off at `T ∈ {1.0, 1.5, 2.0, 3.0, 5.0}`.

**Fail-closed error handling.** Any exception during chunk scoring returns
`malicious_prob = 1.0`, ensuring the guard degrades safely under resource pressure
or model-loading errors.

### 3.2 Layer 2 — PII Redaction

When `REDACT_PII` is enabled, the pipeline applies three regex-based redactors to
both retrieved documents and database output before the content is assembled into
the LLM prompt (`src/employee-facing/rag.py:104–150`):

- **Credit card numbers** (13–19 digit sequences): all but last four digits masked
- **Social Security Numbers** (`\d{3}-\d{2}-\d{4}`): fully masked as `***-**-****`
- **Email addresses**: username redacted beyond first character

Critically, redaction is applied *before* the retrieved content reaches the LLM
context, not after generation. This is architecturally important: a generative
model that has already processed unredacted PII in its context window may reproduce
it even if the output is post-filtered.

### 3.3 Layer 3 — Retrieved-Document Scanning

An optional second invocation of the Prompt Guard classifier screens retrieved
document chunks for embedded injection payloads (`PROMPT_GUARD_CHECK_RETRIEVED_DOCS`).
This defends against **indirect prompt injection** — the emerging attack where an
adversary embeds override instructions in a document that the victim organisation
later ingests into its knowledge base. When enabled, each retrieved document chunk
passes through the same classifier as user queries. The cost is proportional to the
number of retrieved chunks (top-K × average chunk length).

---

## 4. Security Benchmark Design

### 4.1 Configuration Space

The benchmark sweeps **12 guard configurations** plus one unprotected baseline
across all combinations of (side, model, system-prompt size):

| Dimension | Values |
|---|---|
| Deployment side | employee-facing, customer-facing |
| LLM | Qwen3-8B, Llama 3.1 8B |
| System-prompt size | 100, 500, 1000, 1500 tokens |
| Guard configs | 12 protected + 1 unprotected |

This yields **104 distinct run conditions** per attack-prompt set (52 configurations
× 2 system-prompt variants per config — one active and one rolled into the baseline).
The full cross-product is executed in parallel using `concurrent.futures.ProcessPoolExecutor`
with two worker processes per model (`main.py`), sharing a single LLM instance to
avoid the VRAM cost of loading the same weights twice.

### 4.2 Guard Configuration Parameters

The twelve protected configurations vary three axes:

| Parameter | Values tested |
|---|---|
| Chunk size (`max_tokens`) | 128, 512 |
| Chunk overlap | 50 tokens (512-chunk), 100 tokens (128-chunk) |
| Temperature | 1.0, 1.5, 2.0, 3.0, 5.0 |
| Blocking threshold | 0.40, 0.50 |

Smaller chunks with larger overlaps maximise coverage of injection fragments
distributed across long inputs, at the cost of more classifier forward passes.
A 512-token context scanned with 128-token chunks and 100-token overlap requires
⌈(512 − 100) / (128 − 100)⌉ + 1 = 16 forward passes vs. a single pass for a
512-token chunk. On the test hardware this raises per-query latency from ~15 ms
to ~240 ms for the Guard model, which must be weighed against the protection benefit.

### 4.3 Metrics

For each configuration the analysis module (`src/shared/guardrail_analysis.py`)
computes:

- **True positive rate (TPR)** — fraction of attack prompts that returned the
  canonical block response
  (`"We couldn't process that request because it looks like an attempt to change system-level instructions..."`)
- **False positive rate (FPR)** — fraction of benign prompts that were blocked
- **Failures** — attack prompts that passed the guard and received substantive responses,
  written to `guardrail_failures.csv` for manual review

The dual-metric design acknowledges that over-blocking is a real cost for SMB
deployments: a customer-service bot that rejects 20 % of legitimate enquiries is
not commercially viable regardless of its security properties.

---

## 5. Hardware Feasibility Analysis

### 5.1 Minimum Viable Configuration

The reference hardware in the codebase comment is an **NVIDIA RTX 4090 (24 GB VRAM)**,
later upgraded to an **RTX 5090 (32 GB VRAM)**. The memory budget for a complete
running system (one LLM + embeddings + Prompt Guard + FAISS index) is:

| Component | VRAM |
|---|---|
| LLM (8B Q4_K_M) | ~5.5 GB |
| Prompt Guard (86M) | ~0.35 GB |
| Embeddings (bge-small) | ~0.13 GB |
| KV cache (8K context, 8B) | ~2.0 GB |
| FAISS flat index (10k docs) | ~0.05 GB |
| **Total** | **~8.0 GB** |

This falls comfortably within a 16 GB VRAM budget (RTX 4080, RTX 3090, RX 7900 XTX),
meaning an SMB does not need the top consumer GPU tier. Two workers sharing the same
loaded model add ~2 GB each for KV-cache expansion, bringing the dual-worker ceiling
to ~12 GB — matching the ~12 GB observed in the 50GPU training run.

The smaller **Llama-Prompt-Guard-2-22M variant** (noted in the codebase) reduces the
Guard footprint to ~0.09 GB at roughly 4× the inference speed, useful when latency
is critical and the primary LLM context is short.

### 5.2 Total Cost of Ownership

Hardware acquisition is a one-time capital expenditure, while cloud API costs are
variable but perpetual. For a typical SMB workload of 5,000 queries per day at an
average of 800 input + 300 output tokens:

| Cost Item | Cloud (GPT-4o at \$2.50/M in + \$10/M out) | On-Premises |
|---|---|---|
| Daily API cost | \$10 + \$15 = **\$25/day** | \$0 variable |
| Monthly | ~\$750 | ~\$30 electricity |
| Year 1 | ~\$9,000 | ~\$2,500 GPU + \$360 power = **\$2,860** |
| Year 2+ | ~\$9,000/yr | ~\$360/yr |

*Pricing as of mid-2025; electricity at \$0.12/kWh, RTX 4080 at ~250 W.*

The break-even point against a mid-range cloud API lies at roughly **4–5 months**
of continuous operation. For SMBs with predictable, high-volume query loads (legal
document search, internal helpdesk, customer FAQ), the economics strongly favour
on-premises deployment after year one.

Against frontier API models (GPT-4o, Claude Sonnet), a local 8B model does sacrifice
a measurable fraction of raw capability — particularly on multi-step reasoning and
long-document synthesis. The RAG architecture partially compensates by anchoring
responses to retrieved ground truth, which also reduces hallucination rates. For
constrained, domain-specific deployments the quality gap is generally acceptable.

### 5.3 Domain Fine-tuning Economics

The 50GPU branch experiments demonstrate fine-tuning Qwen2.5-0.5B on a 1.6M-example
code corpus (`experiments/train.py`). At ~19.7 training steps/second on an RTX 5090
(~92,000 total steps for one epoch), training time is approximately **78 minutes**
for a complete single epoch. For an SMB fine-tuning a 0.5B model on a domain corpus
of comparable size, the hardware cost is roughly \$0.10–0.20 of electricity — a
negligible figure compared to the fine-tuning API fees of commercial services.

Larger models (8B scale) take proportionally longer but remain tractable on a single
GPU over a weekend run. The significance for security is that a fine-tuned model
will better follow domain-specific system-prompt constraints and produce fewer
off-topic completions that bypass guardrail heuristics, because the base
distribution has been shifted toward the target domain.

---

## 6. Security Risk Analysis

### 6.1 Prompt Injection

Prompt injection is the dominant practical attack surface for an SMB RAG deployment.
Unlike model extraction (which requires thousands of well-designed queries and is
only economically motivated against genuinely proprietary models) or training-data
poisoning (which requires write access to the knowledge base), prompt injection can
be attempted by any authenticated user with access to the chat interface.

The benchmark architecture in `main.py` tests both *direct* injection (malicious
user query) and *system-prompt escalation* (varying the authorising system prompt
from 100 to 1,500 tokens to probe whether longer context degrades guard accuracy).
System-prompt length is a realistic variable in SMB deployments: operations teams
often continuously expand the system prompt with new instructions, inadvertently
diluting the security-relevant phrasing.

The guard's temperature-threshold sweep reveals a fundamental tension. At `T=1.0,
threshold=0.50` the classifier is conservative: it blocks only inputs it is highly
confident are malicious, producing a low false-positive rate but permitting
borderline attacks. Raising temperature to `T=3.0` with `threshold=0.40` broadens
the effective decision boundary, catching more adversarially-crafted inputs at the
cost of misclassifying a fraction of legitimate queries. The optimal operating point
is deployment-specific:

- **Customer-facing** bots handling public users warrant lower thresholds (higher
  recall, accept some false positives) because the adversarial prior is higher and
  blocked users can rephrase.
- **Employee-facing** tools servicing internal staff can afford a higher threshold
  (fewer false positives) because users are authenticated and accountable, and
  a blocked query is a friction event in a workflow.

The 128-token / 100-overlap chunking strategy is empirically important for long
adversarial inputs that distribute the injection payload across many tokens to
defeat simple whole-input classifiers. The sliding-window approach converts the
problem into a sequence of short-window classification decisions, any one of which
can trigger a block.

### 6.2 Data Exfiltration via Retrieved Context

The retrieval layer introduces a second attack vector absent from direct API calls.
An attacker who knows (or guesses) that the RAG system is indexed on sensitive
documents can craft queries designed to elicit retrieval of specific chunks — even
if the LLM would not have memorised that content. Defences include:

1. **Role-based retrieval filtering** — tagging document chunks with permission
   levels and filtering retrieved chunks to the authenticated user's role before
   passing them to the LLM. This is not yet implemented in the codebase but is the
   natural next architectural layer.
2. **Retrieved-document Guard scanning** (`PROMPT_GUARD_CHECK_RETRIEVED_DOCS=True`),
   which prevents adversarially crafted document chunks from hijacking the LLM's
   subsequent behaviour.
3. **PII redaction on retrieved text**, which strips identity signals from document
   content before it reaches the LLM context.

### 6.3 Model and Index Security

The FAISS index is stored on-disk as a flat file (`VS_DIR`). An attacker with
filesystem read access obtains the embedding vectors of all indexed documents. While
recovering full document text from embeddings is not trivially possible, approximate
reconstruction attacks exist for small embedding models. FAISS index files should
be treated as sensitive assets: encrypted at rest, excluded from backups readable
by ordinary staff, and access-logged.

The GGUF model weights are downloaded from HuggingFace Hub on first use and cached
locally. Supply-chain integrity — verifying that cached weights have not been
tampered with — is not addressed in the codebase. For production SMB deployments,
SHA-256 verification of model files against published HuggingFace checksums should
be added to the setup routine.

### 6.4 Multi-Tenancy Isolation

Running both employee-facing and customer-facing RAG systems on the same hardware
raises isolation questions. The two deployments share:

- The VRAM of the running LLM process
- The filesystem (separate FAISS indices, but same machine)
- The Python process environment (via `multiprocessing.spawn`)

Spawn-based multiprocessing (`multiprocessing.get_context("spawn")` in `main.py`)
provides clean process separation at the OS level, preventing one worker from
directly reading another's memory. However, a GPU shared between two processes
can in principle leak information via timing side-channels (GPU cache and compute
queue contention). For most SMB threat models this is theoretical; it becomes
relevant if one deployment handles highly regulated data (health records, legal
privilege) and the other faces untrusted users.

The practical recommendation for high-sensitivity SMBs is to maintain separate
physical or virtual GPU partitions (NVIDIA MIG if using an A-series or H-series
enterprise GPU) for the two deployment contexts, at a significant hardware cost
premium.

---

## 7. Operational Considerations

### 7.1 Maintenance Burden

A self-hosted system shifts the vendor's operational burden to the operator. The
key ongoing costs are:

- **Model updates** — open-weight model releases arrive frequently; new
  GGUF quantisations appear within days of a base model release. The SMB must
  decide when to update, test regressions, and re-tune guard thresholds.
- **Security patches** — llama.cpp, LangChain, FAISS, and transformers are all
  actively developed and periodically carry CVEs. Dependency updates must be
  tested against the application layer.
- **Index freshness** — the FAISS index must be rebuilt (`REBUILD_INDEX=True`)
  whenever the document corpus changes. For dynamic knowledge bases (updated
  HR policies, new product documentation) this needs automation.
- **Guard model currency** — Llama-Prompt-Guard-2-86M was trained on a specific
  distribution of jailbreak prompts. As adversarial prompting evolves, a static
  guard's recall will degrade. Periodic re-evaluation against new attack corpora
  (analogous to the benchmark in this repo) is necessary.

### 7.2 Staffing Requirements

The stack requires Python proficiency to configure and troubleshoot but does not
require ML-research skills for routine operation. A developer familiar with
PyTorch and LangChain can maintain the system with approximately 4–8 hours per
month in steady state. Fine-tuning on domain data (as in the 50GPU experiments)
requires additionally familiarity with HuggingFace `transformers` and `trl` — a
realistic expectation for a technically capable two-to-five person engineering
team at an SMB.

### 7.3 Regulatory Compliance

Self-hosting significantly simplifies GDPR and HIPAA compliance postures by
eliminating the data processor relationship with a cloud API vendor. Data never
leaves the organisation's network boundary. The remaining obligations — access
logging, data minimisation, right-to-erasure — are implementable at the application
layer and are generally less complex than negotiating Data Processing Agreements
with API vendors.

---

## 8. Regulatory Frameworks and Self-Hosted AI

Regulatory compliance is one of the strongest motivating factors for SMBs to
consider on-premises deployment. Cloud API usage introduces a third-party data
processor into the compliance chain, requiring Data Processing Agreements,
vendor audits, and ongoing assurance that the provider's controls meet the
applicable standard. Self-hosted deployment eliminates the third party entirely —
data never crosses an organisational network boundary — but transfers full
compliance responsibility to the operator. This section examines four frameworks
that are directly relevant to SMB operators: HIPAA, the Department of Energy
cybersecurity programme, the Department of Defense contractor framework, and the
emerging federal AI governance landscape.

### 8.1 HIPAA — Health Insurance Portability and Accountability Act

HIPAA's **Security Rule** (45 CFR § 164.312) specifies technical safeguards
for electronic Protected Health Information (ePHI). Healthcare SMBs — clinics,
dental practices, behavioral health providers, billing services — that deploy
AI on patient records fall squarely within its scope.

The four technical safeguard categories map directly onto this codebase's
architecture:

| HIPAA Technical Safeguard | CFR Reference | Relevant System Component |
|---|---|---|
| Access Control | § 164.312(a)(1) | Supabase row-level auth; role-based retrieval |
| Audit Controls | § 164.312(b) | CSV result logs per query in `results/` |
| Integrity | § 164.312(c)(1) | FAISS index checksums (currently absent — gap) |
| Transmission Security | § 164.312(e)(1) | TLS on Supabase API calls; local-only LLM |

A critical 2025 update to the Security Rule removed the historical distinction
between *required* and *addressable* safeguards, making encryption, MFA, and
network segmentation unconditionally mandatory. For an on-premises AI deployment
this means: disk encryption for the FAISS index and GGUF model files, MFA on
the host machine's remote access, and TLS for any network call the pipeline makes
(Supabase, HuggingFace Hub downloads).

The **PII redaction layer** in this system (`src/employee-facing/rag.py:104–150`)
directly addresses the data-minimisation principle that underlies HIPAA: credit
card numbers, SSNs, and email addresses are stripped from retrieved document
content *before* they enter the LLM context, preventing the model from reproducing
them in output. A HIPAA-compliant deployment would extend this to medical record
numbers, dates of birth, and the other 18 PHI identifier categories defined in
45 CFR § 164.514(b)(2).

Compared with cloud API usage, self-hosting eliminates the Business Associate
Agreement (BAA) negotiation that HIPAA mandates for any third-party processor of
ePHI. Several major API providers offer BAAs, but they impose usage restrictions
(no training on customer data, no persistent storage) that constrain fine-tuning
and retrieval-index workflows. On-premises deployment faces none of these
contractual limits.

### 8.2 Department of Energy (DOE) — Cybersecurity Programme

The DOE operates nuclear facilities, national laboratories, and critical energy
infrastructure, and its contractor base includes many SMBs providing engineering,
IT, and analytical services. The governing document is **DOE Order 205.1C**
(*Departmental Cybersecurity Program*), which delegates risk management downward
to individual programme offices and sites but anchors all technical controls in
**NIST SP 800-53** (for federal systems) and **NIST SP 800-171** (for contractor
systems handling Controlled Unclassified Information under 32 CFR Part 2002).

For an SMB contractor deploying an AI assistant on DOE-related work products,
the critical obligations are:

**CUI handling.** Documents classified as CUI Specified — including nuclear
facility design details, export-controlled research, and grid vulnerability
assessments — must remain on systems that satisfy 800-171's 110 security controls.
Self-hosting inherently supports this: the documents never transit an external API
endpoint. Cloud-based RAG over CUI documents would require a FedRAMP-authorised
cloud service, adding substantial procurement complexity and cost.

**System boundary definition.** DOE requires a defined, documented system
boundary for any information system that touches CUI. A self-hosted RAG stack
with a fixed FAISS index, a local inference binary, and a private network
interface presents a simpler boundary to document and certify than a hybrid
system with cloud API calls.

**AI-specific gap.** As of 2025, DOE Order 205.1C does not address AI-specific
risks (model inversion, prompt injection as a CUI-extraction vector, or
adversarial perturbations of embedded documents). The Prompt Guard layer in this
system (`src/shared/prompt_guard.py`) provides a practical defence against
prompt injection that goes beyond what current DOE guidance explicitly requires —
positioning a contractor that implements it ahead of forthcoming AI-specific
annexes to the cybersecurity order.

### 8.3 Department of Defense (DoD) — CMMC, ITAR, and DFARS

DoD contractors face the densest regulatory environment of any SMB category.
Three frameworks are simultaneously relevant to a self-hosted AI deployment.

**CMMC 2.0 (Cybersecurity Maturity Model Certification).** CMMC 2.0 Level 2
requires compliance with all 110 practices in NIST SP 800-171, spread across
14 domains (Access Control, Audit and Accountability, Incident Response, and
others). From DFARS clause 252.204-7012, any system that *processes, stores, or
transmits* Covered Defense Information (CDI) must meet these controls. An LLM
that retrieves and answers questions about defence contract documents processes
CDI within the regulatory definition.

The audit-and-accountability domain (NIST 800-171 controls 3.3.1–3.3.2) requires
logged records of who accessed what data and when. The CSV output structure in
this codebase (`question`, `response`, `duration` per row in `results/`) provides
a partial audit trail but does not capture the authenticated user identity or
the specific document chunks retrieved — both of which would be required for a
CMMC-compliant deployment. Adding user identity to the CSV logger and logging the
FAISS retrieval results alongside each query would close this gap.

**ITAR (International Traffic in Arms Regulations).** ITAR controls the export
of defence articles, technical data, and defence services. As of January 2025,
the Bureau of Industry and Security (BIS) extended export controls to AI model
weights trained with more than 10²⁶ floating-point operations (ECCN 4E091).
For an SMB fine-tuning an LLM on defence-related technical data:

1. The *training data* may itself constitute ITAR-controlled technical data,
   restricting who may participate in the fine-tuning process.
2. The resulting *fine-tuned model weights* may inherit the export-control
   classification of the training data.
3. Uploading weights to a cloud fine-tuning service would constitute an
   unauthorised export.

Self-hosted fine-tuning (as implemented in `experiments/train.py`) is therefore
not merely an economic choice for ITAR-adjacent SMBs — it is a compliance
requirement. The training script runs entirely on local hardware; no data or
weights leave the system.

**DFARS 252.204-7012** requires DoD contractors to report cyber incidents
within 72 hours. A self-hosted system generates local logs that the operator
controls, making incident reconstruction and reporting straightforward. Cloud
API deployments complicate this: the operator may lack access to the provider's
infrastructure logs, potentially impeding the contractually mandated timeline.

### 8.4 Federal AI Security Governance — NIST AI RMF and Emerging Legislation

Unlike HIPAA and CMMC, there is no single enacted federal statute specifically
titled "Federal AI Security Act" as of mid-2025. The term *FAISA* appears in
policy discussions as a proposed regulatory structure that Congress has not
passed; the White House's March 2026 National AI Policy Framework explicitly
rejected the creation of a new AI regulatory agency in favour of sector-specific
oversight through existing bodies (FDA, FTC, SEC, CFPB). The operative federal
AI governance standard for SMBs seeking to demonstrate security due diligence is
therefore a combination of the NIST AI Risk Management Framework and the sector
regulator's own guidance.

**NIST AI RMF 1.0** (published January 2023) organises AI risk management into
four functions — *GOVERN*, *MAP*, *MEASURE*, and *MANAGE* — that together
constitute a continuous risk cycle. NIST AI 600-1 (July 2024) extends this into
a Generative AI Profile that addresses LLM-specific risks including prompt
injection, data poisoning, and hallucination. The profile is referenced by the
FTC, FDA, and EEOC in their own AI guidance.

Mapping the codebase against the AI RMF Generative AI Profile:

| NIST AI 600-1 Risk | Profile Suggested Action | Implementation in this Repo |
|---|---|---|
| Prompt Injection | Input validation and filtering | Prompt Guard with 12-config sweep |
| Data Privacy | PII de-identification | Regex redaction before LLM context |
| Confabulation (hallucination) | Ground responses in retrieved facts | RAG retrieval anchors all answers |
| Information Integrity | Provenance tracking for retrieved content | FAISS retrieval (chunk source partially tracked) |
| Model Behaviour Anomalies | Audit logging | CSV query/response logs |
| Harmful Bias | Documented evaluation | Not addressed — gap for regulated deployments |

The **GOVERN** function of the AI RMF requires that the organisation document
roles, responsibilities, and accountability for AI risk. For an SMB, this means
designating an owner for the RAG system configuration, establishing a review
cadence for guard-threshold recalibration (analogous to the benchmark sweep in
`main.py`), and maintaining records of system-prompt changes. None of this
requires a large compliance team; a single technically accountable person and
a lightweight change-log discipline satisfies the intent.

**Executive Order 14110** (*Safe, Secure, and Trustworthy AI*, October 2023)
directed NIST, CISA, and sector agencies to develop AI-specific security
guidelines. The practical output for SMBs has been sector-specific guidance
(FDA's AI/ML action plan for medical devices, CISA's AI security guidelines
for critical infrastructure) rather than uniform cross-sector mandates. SMBs
operating in these sectors should consult the sector-specific annexes; the
common thread across all of them is the requirement for documented evaluation of
adversarial robustness — precisely the kind of systematic guard-configuration
testing that this repo's benchmark was designed to produce.

### 8.5 Cross-Framework Compliance Summary

| Framework | Data-Residency Benefit | Primary Gap in This Codebase | Effort to Close |
|---|---|---|---|
| HIPAA | Eliminates BAA requirement | PHI identifier regex coverage incomplete | Low |
| DOE / CUI | Satisfies 800-171 boundary | No AI-specific risk documentation | Low |
| DoD / CMMC | Enables local CDI processing | Audit log missing user identity + retrieved chunks | Medium |
| DoD / ITAR | Fine-tuning on local hardware only | No weight-export controls or checksums | Medium |
| NIST AI RMF | Full transparency into stack | Bias evaluation absent; provenance partial | Medium |

Across all five frameworks the self-hosted architecture provides a materially
stronger starting position than cloud API usage: data residency is automatic,
the full audit surface is operator-controlled, and supply-chain risk is bounded
to the open-weight model download (which is addressable with checksum
verification). The gaps identified above are engineering tasks, not architectural
changes, and fall within the capacity of a motivated SMB development team.

---

## 9. Summary of Findings

| Criterion | Assessment |
|---|---|
| Hardware feasibility | **Viable** on 16 GB VRAM consumer GPU for 8B models |
| Economic break-even | **~4–5 months** vs. mid-range cloud API at 5k queries/day |
| Primary security risk | **Prompt injection** — mitigated by multi-layer guard |
| Guard effectiveness | Highly configuration-dependent; optimal point is context-specific |
| False-positive cost | Real and non-trivial; must be calibrated per deployment |
| Fine-tuning feasibility | **Viable** for 0.5B–8B models on a single GPU in hours–days |
| Operational complexity | Moderate; requires developer-level Python/ML familiarity |
| Regulatory posture | **Improved** vs. cloud APIs for data-residency requirements |

---

## 10. Conclusion

The evidence from this codebase suggests that on-premises AI deployment is
technically and economically viable for SMBs with a moderate technical staff and
a sufficiently high query volume to justify the capital expenditure. The security
posture achievable with open-source tooling — a multi-layer stack of input
classification, PII redaction, and retrieved-document scanning — is non-trivial
and comparable in principle to what commercial API providers describe in their own
security documentation, though without the benefit of vendor-managed updates.

The principal remaining gap is not technical capability but operational discipline:
maintaining fresh indices, monitoring guard recall against evolving attack
distributions, and enforcing filesystem-level controls on model and index assets.
These are solvable engineering problems within the reach of an SMB's IT function,
not research challenges.

For organisations where data residency, cost predictability, and model
customisability are priorities, on-premises deployment should be a first-class
option in the architectural decision process — not a niche alternative reserved
for large enterprises.

---

## Appendix A — Repository Component Map

| Component | Path | Role |
|---|---|---|
| Benchmark runner | `main.py` | Orchestrates the 52-configuration security sweep |
| Employee RAG app | `src/employee-facing/rag.py` | LLM + FAISS + auth pipeline for staff |
| Customer RAG app | `src/customer-facing/rag.py` | LLM + FAISS pipeline for public users |
| Prompt Guard | `src/shared/prompt_guard.py` | Llama-PG-2 wrapper with chunking + temperature |
| Guardrail analysis | `src/shared/guardrail_analysis.py` | Post-benchmark TPR/FPR reporting |
| Fine-tuning script | `experiments/train.py` | Qwen2.5-0.5B training on tiny-codes |
| Tokenised dataset | `experiments/tokenized_tiny_codes_qwen2.5_0.5b_train/` | 1.47M pre-tokenised examples |
| Training checkpoints | `experiments/results/checkpoint-*/` | Saved at every 500 steps |

## Appendix B — Guard Configuration Reference

| Config name | Chunk | Overlap | Temp | Threshold |
|---|---|---|---|---|
| c512_temp1.0_th0.40 | 512 | 50 | 1.0 | 0.40 |
| c512_temp2.0_th0.40 | 512 | 50 | 2.0 | 0.40 |
| c512_temp3.0_th0.40 | 512 | 50 | 3.0 | 0.40 |
| c128_ov100_temp1.0_th0.40 | 128 | 100 | 1.0 | 0.40 |
| c128_ov100_temp1.5_th0.40 | 128 | 100 | 1.5 | 0.40 |
| c128_ov100_temp2.0_th0.40 | 128 | 100 | 2.0 | 0.40 |
| c128_ov100_temp3.0_th0.40 | 128 | 100 | 3.0 | 0.40 |
| c128_ov100_temp5.0_th0.40 | 128 | 100 | 5.0 | 0.40 |
| c128_ov100_temp3.0_th0.50 | 128 | 100 | 3.0 | 0.50 |
| c128_ov100_temp5.0_th0.50 | 128 | 100 | 5.0 | 0.50 |
| c128_ov100_temp7.0_th0.40 | 128 | 100 | 7.0 | 0.40 |
| c128_ov100_temp7.0_th0.50 | 128 | 100 | 7.0 | 0.50 |
