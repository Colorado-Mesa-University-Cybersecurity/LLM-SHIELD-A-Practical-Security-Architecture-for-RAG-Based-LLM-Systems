from typing import List, Dict, Any, Optional

try:
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception as e:
    raise RuntimeError(
        "Install transformers and torch to use PromptGuard: "
        "pip install transformers torch"
    ) from e


class PromptGuard:

    # Label index for the MALICIOUS/jailbreak class per the official tutorial.
    _MALICIOUS_IDX: int = 1

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Prompt-Guard-2-86M",
        device: Optional[int] = None,
        max_tokens: int = 512,
        chunk_overlap: int = 50,
        threshold: float = 0.5,
        temperature: float = 1.0,
        debug: bool = False,
    ):
        """
        Args:
            model_id:      HuggingFace model ID.  Defaults to the 86M variant
                           for best accuracy.  Substitute the 22M variant
                           ('meta-llama/Llama-Prompt-Guard-2-22M') when
                           latency matters more than recall (~4× faster).
            device:        GPU device index (0, 1, …) or -1 / None for CPU.
            max_tokens:    Maximum tokens per chunk fed to the model.
                           Must be ≤ the model's positional embedding limit
                           (512 for Llama-Prompt-Guard-2).
            chunk_overlap: Token overlap between adjacent chunks so that
                           injections near chunk boundaries are not missed.
            threshold:     Malicious-probability threshold for blocking.
                           0.5 is the neutral point; lower values block more
                           aggressively (higher recall, more false positives).
            temperature:   Softmax temperature.  Values > 1 produce softer
                           (less confident) probability distributions and can
                           be useful during threshold calibration.  Use 1.0
                           for standard production blocking.
            debug:         Print per-chunk scores to stdout.
        """
        self.model_id      = model_id
        self.max_tokens    = max_tokens
        self.chunk_overlap = chunk_overlap
        self.threshold     = float(threshold)
        self.temperature   = float(temperature)
        self.debug         = bool(debug)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device >= 0:
            self._device = torch.device(f"cuda:{device}")
        else:
            self._device = torch.device("cpu")

        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.to(self._device)
        self.model.eval()

        if self.debug:
            print(
                f"[prompt_guard] Loaded {self.model_id} on {self._device}  "
                f"threshold={self.threshold}  temperature={self.temperature}"
            )

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        """Split *text* into overlapping token-level chunks of at most
        *max_tokens* tokens.  Returns the original string unchanged when it
        fits within a single chunk."""
        if not text:
            return []

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_tokens:
            return [text]

        chunks: List[str] = []
        step = max(1, self.max_tokens - self.chunk_overlap)
        start = 0
        while start < len(ids):
            end = min(start + self.max_tokens, len(ids))
            chunk_text = self.tokenizer.decode(
                ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            chunks.append(chunk_text)
            if end == len(ids):
                break
            start += step
        return chunks

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_chunk(self, text: str) -> float:
        """Return the malicious probability for a single chunk.

        Implements the tutorial's `get_class_probabilities` / `get_jailbreak_score`
        pattern:
            logits  → scaled_logits / temperature → softmax → prob[MALICIOUS_IDX]
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        scaled_logits = logits / self.temperature
        probabilities = softmax(scaled_logits, dim=-1)
        return probabilities[0, self._MALICIOUS_IDX].item()

    # ── Public interface ──────────────────────────────────────────────────────

    def check_text(self, text: str) -> Dict[str, Any]:
        """Classify *text* and return a result dict compatible with the rest of
        the pipeline.

        Returns:
            {
                "is_malicious":  bool,
                "malicious_prob": float,   # max across chunks
                "label":         "MALICIOUS" | "BENIGN",
                "per_chunk":     [{"text": str, "malicious_prob": float}, …],
            }

        Error policy: if scoring a chunk raises an exception the chunk is
        treated as fully malicious (prob = 1.0) so the guard fails closed.
        """
        chunks = self._chunk_text(text)
        if not chunks:
            return {
                "is_malicious":  False,
                "malicious_prob": 0.0,
                "label":         "BENIGN",
                "per_chunk":     [],
            }

        per_chunk: List[Dict[str, Any]] = []
        max_mal_prob = 0.0

        for chunk in chunks:
            try:
                mal_prob = self._score_chunk(chunk)
            except Exception as exc:
                if self.debug:
                    print(f"[prompt_guard] scoring error (fail-closed): {exc}")
                mal_prob = 1.0

            if self.debug:
                print(f"[prompt_guard] chunk mal_prob={mal_prob:.6f}  "
                      f"text={chunk[:60]!r}")

            per_chunk.append({"text": chunk, "malicious_prob": mal_prob})
            if mal_prob > max_mal_prob:
                max_mal_prob = mal_prob

        label = "MALICIOUS" if max_mal_prob >= self.threshold else "BENIGN"
        return {
            "is_malicious":  label == "MALICIOUS",
            "malicious_prob": max_mal_prob,
            "label":         label,
            "per_chunk":     per_chunk,
        }
