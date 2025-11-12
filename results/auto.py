from typing import Dict, List, Optional
import json
import sys
import pandas as pd

CONFIG = {
    "input_csv": "ragUnsecurePrompt.csv",  # path to your CSV
    "output_csv": "secureOutput.csv",
    "keywords_json": "keywords.json",  # e.g. "keywords.json" or None
    "keyword_column": None,  # e.g. "keywords" if present in CSV
    "save_keywords": None,  # e.g. "final_kw.json" or None
    "interactive": True,  # True to prompt for missing keywords
}


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
    # Try JSON array first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    # Fallback: comma-separated
    return [p.strip() for p in text.split(",") if p.strip()]


def load_keywords_from_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, (list, tuple)):
            out[str(k)] = [str(x).strip() for x in v if str(x).strip()]
        else:
            out[str(k)] = parse_keywords_cell(v)
    return out


def match_response(response: str, keywords: List[str]) -> List[str]:
    """
    Substring-only matching (case-insensitive).
    Returns a list of keywords that appear as substrings in the response.
    """
    if not response:
        response = ""
    resp = str(response)
    low = resp.lower()
    matched: List[str] = []
    for kw in keywords:
        if kw and kw.lower() in low:
            matched.append(kw)
    return matched


def interactive_collect_keywords(unique_questions: List[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    print("Interactive keyword entry. For each question enter comma-separated keywords.")
    print("Leave blank to skip (treated as no keywords). Type 'quit' to stop early.")
    for q in unique_questions:
        print("\nQuestion:")
        print(q)
        s = input("Keywords (comma-separated) > ").strip()
        if s.lower() == "quit":
            break
        mapping[q] = parse_keywords_cell(s)
    return mapping


def process_csv(
    input_csv: str,
    output_csv: str = "annotated_output.csv",
    keywords_json: Optional[str] = None,
    keyword_column: Optional[str] = None,
    save_keywords: Optional[str] = None,
    interactive: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv, dtype=str).fillna("")
    if "question" not in df.columns or "response" not in df.columns:
        raise ValueError("Input CSV must contain 'question' and 'response' columns.")

    # Load JSON mapping if provided
    keywords_map: Dict[str, List[str]] = {}
    if keywords_json:
        keywords_map = load_keywords_from_json(keywords_json)

    # Parse per-row keyword column if provided
    if keyword_column and keyword_column in df.columns:
        df["_parsed_keywords"] = df[keyword_column].apply(parse_keywords_cell)
    else:
        df["_parsed_keywords"] = [[] for _ in range(len(df))]

    # For rows without per-row keywords try to apply mapping by question
    questions_to_prompt: List[str] = []
    for i, row in df.iterrows():
        q = row["question"]
        row_kws: List[str] = df.at[i, "_parsed_keywords"]
        if row_kws:
            continue
        if q in keywords_map and keywords_map[q]:
            df.at[i, "_parsed_keywords"] = keywords_map[q]
            continue
        questions_to_prompt.append(q)

    # Deduplicate questions to prompt
    seen = set()
    dedup = []
    for q in questions_to_prompt:
        if q not in seen:
            seen.add(q)
            dedup.append(q)

    if dedup and interactive and sys.stdin.isatty():
        collected = interactive_collect_keywords(dedup)
        # apply collected to rows
        for i, row in df.iterrows():
            if df.at[i, "_parsed_keywords"]:
                continue
            q = row["question"]
            if q in collected and collected[q]:
                df.at[i, "_parsed_keywords"] = collected[q]
            elif q in keywords_map and keywords_map[q]:
                df.at[i, "_parsed_keywords"] = keywords_map[q]
    # else non-interactive: leave empty where not provided

    # Matching (substring-only)
    matched_list: List[List[str]] = []
    results: List[str] = []
    keyword_strings: List[str] = []
    for i, row in df.iterrows():
        resp = row["response"]
        kws: List[str] = df.at[i, "_parsed_keywords"]
        matched = match_response(resp, kws) if kws else []
        matched_list.append(matched)
        results.append("pass" if matched else "failed")
        keyword_strings.append(",".join(kws))

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
        interactive=CONFIG["interactive"],
    )