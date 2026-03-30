import re
from typing import Dict, List, Optional
import json
import pandas as pd

CONFIG = {
    "input_csv": "prompts/1000_prompt.csv",
    "output_csv": "prompts/1000_prompt.csv",
    "keywords_json": "keywords.json",  # e.g. "keywords.json" or None
    "keyword_column": None,  # e.g. "keywords" if present in CSV
    "save_keywords": None,  # e.g. "final_kw.json" or None
    "debug": False,  # set True to print debug info for misses
}

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    text = str(s)
    # normalize common unicode punctuation to ASCII equivalents
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # collapse whitespace
    text = " ".join(text.split())
    return text.strip().lower()

def _token_set(s: str) -> set:
    # remove punctuation (keep word characters and whitespace), split into tokens
    s = s or ""
    cleaned = re.sub(r"[^\w\s]", " ", s)
    tokens = [t for t in cleaned.split() if len(t) > 1]  # drop 1-char tokens
    return set(tokens)

def parse_keywords_cell(cell: Optional[str]) -> List[str]:
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell if str(x).strip()]
    text = str(cell).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    # Fallback: split on comma, semicolon, or newline
    parts = re.split(r"[,\n;]+", text)
    return [p.strip() for p in parts if p.strip()]

def load_keywords_from_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        key_norm = normalize_text(k)
        if isinstance(v, (list, tuple)):
            out[key_norm] = [str(x).strip() for x in v if str(x).strip()]
        else:
            out[key_norm] = parse_keywords_cell(v)
    return out

def match_response(response: str, keywords: List[str],
                   token_overlap_frac: float = 0.5) -> List[str]:
    if not response:
        response = ""
    resp_norm = normalize_text(response)
    resp_tokens = _token_set(resp_norm)

    matched: List[str] = []
    for kw in keywords:
        if not kw or not str(kw).strip():
            continue
        orig_kw = str(kw).strip()
        k_norm = normalize_text(orig_kw)

        # exact substring check
        if k_norm and k_norm in resp_norm:
            matched.append(orig_kw)
            continue

        # token overlap: fraction of keyword tokens present in response tokens
        kw_tokens = _token_set(k_norm)
        if kw_tokens:
            overlap = len(kw_tokens & resp_tokens) / len(kw_tokens)
            if overlap >= token_overlap_frac:
                matched.append(orig_kw)
                continue

    return matched

def process_csv(
    input_csv: str,
    output_csv: str = "annotated_output.csv",
    keywords_json: Optional[str] = None,
    keyword_column: Optional[str] = None,
    save_keywords: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv, dtype=str).fillna("")
    if "question" not in df.columns or "response" not in df.columns:
        raise ValueError("Input CSV must contain 'question' and 'response' columns.")

    # Load JSON mapping
    keywords_map: Dict[str, List[str]] = {}
    if keywords_json:
        keywords_map = load_keywords_from_json(keywords_json)

    # Parse per-row keyword column if provided
    if keyword_column and keyword_column in df.columns:
        df["_parsed_keywords"] = df[keyword_column].apply(parse_keywords_cell)
    else:
        df["_parsed_keywords"] = [[] for _ in range(len(df))]

    # For rows without per-row keywords try to apply mapping by normalized question
    for i, row in df.iterrows():
        q = row["question"]
        q_norm = normalize_text(q)
        row_kws: List[str] = df.at[i, "_parsed_keywords"]
        if row_kws:
            continue
        if q_norm in keywords_map and keywords_map[q_norm]:
            df.at[i, "_parsed_keywords"] = keywords_map[q_norm]

    # Matching
    matched_list: List[List[str]] = []
    results: List[str] = []
    keyword_strings: List[str] = []
    debug = CONFIG.get("debug", False)

    for i, row in df.iterrows():
        resp = row["response"]
        kws: List[str] = df.at[i, "_parsed_keywords"]
        matched = match_response(resp, kws) if kws else []
        matched_list.append(matched)
        results.append("pass" if matched else "failed")
        keyword_strings.append(",".join(kws))

        if debug and kws and not matched:
            print("DEBUG MISS:")
            print(" Question:", row["question"])
            print(" Response:", repr(resp))
            print(" Keywords:", kws)
            print(" Normalized response:", normalize_text(resp))
            print(" Normalized keywords:", [normalize_text(k) for k in kws])
            print("-" * 40)

    df["keywords"] = keyword_strings
    df["matched_keywords"] = [";".join(m) for m in matched_list]
    df["result"] = results

    # Clean up internal column
    if "_parsed_keywords" in df.columns:
        df = df.drop(columns=["_parsed_keywords"])

    df.to_csv(output_csv, index=False)
    if save_keywords:
        mapping: Dict[str, List[str]] = {}
        for i, row in df.iterrows():
            q = row["question"]
            if q not in mapping:
                mapping[q] = parse_keywords_cell(row.get("keywords", ""))
        with open(save_keywords, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Wrote annotated CSV to: {output_csv}")
    if save_keywords:
        print(f"Saved question->keywords mapping to: {save_keywords}")

    return df

if __name__ == "__main__":
    process_csv(
        input_csv=CONFIG["input_csv"],
        output_csv=CONFIG["output_csv"],
        keywords_json=CONFIG["keywords_json"],
        keyword_column=CONFIG["keyword_column"],
        save_keywords=CONFIG["save_keywords"],
    )