"""
Create an instruction-tuning dataset for Project 2 LLM integration.

Input:
    train_data with labels.xlsx

Output:
    project2_llm_instruction_data.jsonl

This creates two instruction styles:
1. AI Analyst examples that explain the classification.
2. Writing Coach examples that provide revision guidance.

Run:
    python prepare_instruction_dataset.py --input "train_data with labels.xlsx"
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def find_text_and_label_columns(df: pd.DataFrame):
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = None
    label_col = None

    for candidate in ["text", "content", "document", "essay", "clean_text"]:
        if candidate in cols:
            text_col = cols[candidate]
            break

    for candidate in ["label", "labels", "target", "class"]:
        if candidate in cols:
            label_col = cols[candidate]
            break

    if text_col is None:
        text_col = df.select_dtypes(include="object").columns[0]
    if label_col is None:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError("Could not find a numeric label column.")
        label_col = numeric_cols[0]

    return text_col, label_col


def make_examples(text: str, label: int):
    label_name = "AI-written" if int(label) == 1 else "Human-written"
    opposite = "human-written" if int(label) == 1 else "AI-written"
    preview = text[:1500]

    analyst_prompt = (
        "You are the AI Analyst in an AI-vs-human text detection app. "
        "Use the classifier result and text preview to explain the decision without claiming certainty.\n\n"
        f"Classifier result: {label_name}\n"
        f"Text preview:\n{preview}"
    )
    analyst_response = (
        f"Executive Summary: The classifier result is {label_name}. This should be treated as a probability-based signal, not proof of authorship.\n\n"
        "Evidence: The decision should be explained using writing structure, vocabulary patterns, repetition, sentence rhythm, and model confidence. "
        f"The explanation should also mention that some text can contain both AI-like and {opposite} characteristics."
    )

    coach_prompt = (
        "You are the Writing Coach in an AI-vs-human text detection app. "
        "Give ethical revision guidance that improves clarity, specificity, and natural voice. Do not help the user cheat a detector.\n\n"
        f"Classifier result: {label_name}\n"
        f"Text preview:\n{preview}"
    )
    coach_response = (
        "Strengths: The text can be improved by preserving its main meaning and keeping the strongest supporting ideas.\n"
        "High-priority revisions: Add specific examples, vary sentence length, reduce generic transitions, and make the writer's purpose clearer.\n"
        "Medium-priority revisions: Replace repeated wording, simplify overly polished phrasing, and add more natural voice where appropriate."
    )

    return [
        {"instruction": analyst_prompt, "response": analyst_response, "task": "ai_analyst"},
        {"instruction": coach_prompt, "response": coach_response, "task": "writing_coach"},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Excel dataset")
    parser.add_argument("--output", default="project2_llm_instruction_data.jsonl")
    parser.add_argument("--max_rows", type=int, default=500)
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    text_col, label_col = find_text_and_label_columns(df)

    out_path = Path(args.output)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.head(args.max_rows).iterrows():
            text = str(row[text_col]).strip()
            if not text or text.lower() == "nan":
                continue
            try:
                label = int(row[label_col])
            except Exception:
                continue
            for ex in make_examples(text, label):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} instruction examples to {out_path}")
    print(f"Text column: {text_col}")
    print(f"Label column: {label_col}")


if __name__ == "__main__":
    main()
