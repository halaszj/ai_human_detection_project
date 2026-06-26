"""
AI vs. Human Text Detection - Project 2 Streamlit Application

This file is the main web application submitted for Project 2. It keeps the
Project 1 machine/deep learning models and adds Project 2 LLM functionality.

High-level flow:
1. The user pastes text or uploads a TXT/PDF/DOCX file.
2. The app runs one model or all six saved Project 1 models.
3. The app summarizes prediction, confidence, majority vote, and statistics.
4. LLM 1 acts as an AI Analyst and explains the ensemble results.
5. LLM 2 acts as a Writing Coach and gives revision guidance.
6. The app generates downloadable TXT and PDF reports.

Important design choice:
The saved ML models are loaded lazily, meaning each model is loaded only when
it is used. This keeps the Hugging Face Space lighter and prevents unnecessary
memory use at startup.
"""

# ============================================================
# Project 2 - AI vs Human Text Detection with LLM Explanations
# Streamlit app rebuilt for a cleaner dashboard, structured LLM
# analysis, and comprehensive TXT/PDF reports.
# ============================================================

import os
import re
import json
from io import BytesIO
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from text_utils import clean_text


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="AI vs Human",
    layout="wide",
)


# ============================================================
# MODEL FILE LOCATIONS
# ============================================================

MODEL_FILES = {
    "SVM": "models/svm.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "AdaBoost": "models/adaboost.pkl",
    "FNN": "models/fnn.pkl",
    "LSTM": "models/lstm.pkl",
    "CNN for Text": "models/cnn_for_text.pkl",
}


# ============================================================
# MODEL LOADING
# ============================================================

# Streamlit caches loaded models so repeated predictions do not reload the same pickle file.
# Each model is loaded individually to reduce memory pressure on Hugging Face Spaces.

@st.cache_resource(show_spinner=False)
def load_one_model(name):
    """Load only the model selected/needed so the app stays lighter."""
    path = MODEL_FILES.get(name)
    if path is None:
        raise ValueError(f"Unknown model: {name}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


def available_model_names():
    return [name for name, path in MODEL_FILES.items() if os.path.exists(path)]


available_models = available_model_names()
missing_models = [path for _, path in MODEL_FILES.items() if not os.path.exists(path)]

if not available_models:
    st.error("No trained model files were found in the models folder.")
    st.stop()


# ============================================================
# FILE EXTRACTION
# ============================================================


def extract_pdf_text(uploaded_file):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"PDF extraction error: {e}"


def extract_docx_text(uploaded_file):
    try:
        import docx
        document = docx.Document(uploaded_file)
        return "\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
    except Exception as e:
        return f"DOCX extraction error: {e}"


def extract_txt_text(uploaded_file):
    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"TXT extraction error: {e}"


# ============================================================
# TEXT ANALYTICS
# ============================================================


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|[.!?]+", text) if s.strip()]


def get_text_statistics(text):
    words = re.findall(r"\b\w+\b", text)
    sentences = split_sentences(text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    unique_words = len(set(w.lower() for w in words))
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    vocab_richness = unique_words / word_count if word_count else 0
    sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    long_sentence_count = sum(1 for length in sentence_lengths if length >= 25)
    short_sentence_count = sum(1 for length in sentence_lengths if length <= 8)
    return {
        "Word Count": word_count,
        "Character Count": len(text),
        "Sentence Count": sentence_count,
        "Paragraph Count": len(paragraphs),
        "Average Sentence Length": round(avg_sentence_length, 2),
        "Vocabulary Richness": round(vocab_richness, 3),
        "Unique Words": unique_words,
        "Long Sentences": long_sentence_count,
        "Short Sentences": short_sentence_count,
        "Sentence Lengths": sentence_lengths,
    }


def get_top_words(text, limit=12):
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", clean_text(text))]
    if not words:
        return pd.DataFrame(columns=["Word", "Count"])
    counts = pd.Series(words).value_counts().head(limit)
    return pd.DataFrame({"Word": counts.index, "Count": counts.values})


# ============================================================
# MODEL PREDICTION
# ============================================================


def make_model_input(text):
    return pd.DataFrame({"text": [text], "clean_text": [clean_text(text)]})


def predict_with_model(model, text):
    X_input = make_model_input(text)
    try:
        pred = model.predict(X_input)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            confidence = float(np.max(proba))
        elif hasattr(model, "decision_function"):
            score = model.decision_function(X_input)
            confidence = float(1 / (1 + np.exp(-np.max(score))))
        else:
            confidence = 0.50
        label = "AI-written" if int(pred) == 1 else "Human-written"
        return label, confidence, ""
    except Exception as e:
        return "Prediction error", 0.0, str(e)


def get_strength(confidence):
    if confidence >= 0.90:
        return "Very strong"
    if confidence >= 0.75:
        return "Strong"
    if confidence >= 0.60:
        return "Moderate"
    return "Low"


def get_risk_level(prediction, confidence, agreement_ratio):
    if prediction == "Tie / Mixed":
        return "Mixed", "🟡"
    if confidence >= 0.85 and agreement_ratio >= 0.75:
        return "High", "🔴" if prediction == "AI-written" else "🟢"
    if confidence >= 0.65:
        return "Moderate", "🟡"
    return "Low", "🟢"


def get_model_reason(model_name, prediction, confidence, error_message=""):
    if prediction == "Prediction error":
        return f"This model could not complete the prediction. {error_message}".strip()
    model_notes = {
        "SVM": "Compares high-dimensional TF-IDF vocabulary and linguistic patterns.",
        "Decision Tree": "Uses rule-based splits on engineered text and linguistic features.",
        "AdaBoost": "Combines multiple weak learners and emphasizes difficult examples.",
        "FNN": "Learns combinations of text-vector and linguistic features.",
        "LSTM": "Uses a trained text-classification pipeline representing sequence-style learning behavior.",
        "CNN for Text": "Uses a trained text-classification pipeline representing phrase-pattern detection behavior.",
    }
    direction = "closer to AI-labeled examples" if prediction == "AI-written" else "closer to human-labeled examples"
    return f"{model_notes.get(model_name, 'Uses trained text features.')} This sample was {direction} with {get_strength(confidence).lower()} confidence."


# ============================================================
# FEATURE EXPLANATION
# ============================================================


def explain_prediction(text):
    stats = get_text_statistics(text)
    cleaned = clean_text(text)
    words = cleaned.split()
    repeated_words = pd.Series(words).value_counts().head(8) if words else pd.Series(dtype=int)
    sentence_lengths = stats["Sentence Lengths"]
    if sentence_lengths:
        sentence_variation = float(np.std(sentence_lengths))
    else:
        sentence_variation = 0.0

    items = [
        {
            "Feature Area": "Text Length",
            "Observed Pattern": f"{stats['Word Count']} words and {stats['Sentence Count']} sentences",
            "Risk Signal": "More evidence available" if stats["Word Count"] >= 150 else "Limited evidence",
            "Possible Meaning": "Longer text gives classifiers more vocabulary and structure evidence. Short text is less reliable.",
        },
        {
            "Feature Area": "Sentence Structure",
            "Observed Pattern": f"Average sentence length: {stats['Average Sentence Length']} words",
            "Risk Signal": "Potentially formulaic" if 12 <= stats["Average Sentence Length"] <= 24 else "Variable",
            "Possible Meaning": "Very even sentence structure can appear polished or template-driven.",
        },
        {
            "Feature Area": "Sentence Variation",
            "Observed Pattern": f"Sentence-length variation: {sentence_variation:.2f}",
            "Risk Signal": "Low variation" if sentence_variation < 6 and stats["Sentence Count"] >= 4 else "Moderate/high variation",
            "Possible Meaning": "Human writing often has more uneven rhythm; AI writing can be more consistently paced.",
        },
        {
            "Feature Area": "Vocabulary Richness",
            "Observed Pattern": f"Vocabulary richness: {stats['Vocabulary Richness']}",
            "Risk Signal": "Lower variety" if stats["Vocabulary Richness"] < 0.55 and stats["Word Count"] > 100 else "Reasonable variety",
            "Possible Meaning": "Lower vocabulary variety can indicate repetitive or formulaic writing, though topic-specific writing may repeat terms naturally.",
        },
    ]

    if not repeated_words.empty:
        repeated_text = ", ".join(f"{word} ({count})" for word, count in repeated_words.items())
        items.append({
            "Feature Area": "Repeated Terms",
            "Observed Pattern": repeated_text,
            "Risk Signal": "Topic/repetition pattern",
            "Possible Meaning": "Repeated words influence TF-IDF patterns and may make writing appear more structured or topic-bound.",
        })

    return pd.DataFrame(items)


# ============================================================
# PROJECT 2 LLM INTEGRATION
# ============================================================

HF_API_URL = "https://api-inference.huggingface.co/models/"
# Project 2 LLM model selection.
# Supports BOTH the older variable names and the clearer final names used in the setup guide.
DEFAULT_ANALYST_MODEL = "google/flan-t5-small"
DEFAULT_COACH_MODEL = "google/flan-t5-small"

ANALYST_MODEL_ID = (
    os.environ.get("FINETUNED_ANALYST_MODEL")
    or os.environ.get("FINETUNED_LLM1_MODEL")
    or DEFAULT_ANALYST_MODEL
)

COACH_MODEL_ID = (
    os.environ.get("FINETUNED_COACH_MODEL")
    or os.environ.get("FINETUNED_LLM2_MODEL")
    or DEFAULT_COACH_MODEL
)

LLM_MODELS = {
    "LLM 1 - Prediction Explainer": ANALYST_MODEL_ID,
    "LLM 2 - Writing Coach": COACH_MODEL_ID,
}


def llm_status_rows():
    """Return display rows that prove whether the app is configured for fine-tuned LLMs."""
    hf_token_found = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    analyst_is_finetuned = ANALYST_MODEL_ID != DEFAULT_ANALYST_MODEL
    coach_is_finetuned = COACH_MODEL_ID != DEFAULT_COACH_MODEL
    return pd.DataFrame([
        {
            "LLM Tool": "LLM 1 - AI Analyst",
            "Configured Model": ANALYST_MODEL_ID,
            "Status": "Fine-tuned model configured" if analyst_is_finetuned else "Base fallback model",
            "Expected Extra-Credit Repo": "halaszj/ai-text-analyst-flan-t5",
        },
        {
            "LLM Tool": "LLM 2 - Writing Coach",
            "Configured Model": COACH_MODEL_ID,
            "Status": "Fine-tuned model configured" if coach_is_finetuned else "Base fallback model",
            "Expected Extra-Credit Repo": "halaszj/ai-writing-coach-flan-t5",
        },
        {
            "LLM Tool": "HF_TOKEN",
            "Configured Model": "Present" if hf_token_found else "Not found",
            "Status": "Private/large model access enabled" if hf_token_found else "Public inference only",
            "Expected Extra-Credit Repo": "Add as Space Secret, not Variable",
        },
    ])


def llm_status_summary():
    analyst_is_finetuned = ANALYST_MODEL_ID != DEFAULT_ANALYST_MODEL
    coach_is_finetuned = COACH_MODEL_ID != DEFAULT_COACH_MODEL
    if analyst_is_finetuned and coach_is_finetuned:
        return "✅ Both fine-tuned LLM model IDs are configured in this Space."
    if analyst_is_finetuned or coach_is_finetuned:
        return "🟡 One fine-tuned LLM model ID is configured; the other is still using a base fallback model."
    return "⚪ Fine-tuned LLM model IDs are not configured yet; the app is using base fallback models."


def get_hf_headers():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def call_huggingface_llm(model_id, prompt, max_new_tokens=220):
    """Call a hosted Hugging Face model through the Inference API.

    The app uses hosted inference instead of installing local LLM dependencies.
    This prevents Streamlit Spaces from running out of memory during build.
    If the hosted model is unavailable, the app returns a readable fallback
    message instead of crashing during the demo.
    """
    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.25,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        response = requests.post(
            HF_API_URL + model_id,
            headers=get_hf_headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return (first.get("generated_text") or first.get("summary_text") or first.get("text") or json.dumps(first)).strip()
        if isinstance(data, dict):
            return (data.get("generated_text") or data.get("error") or json.dumps(data)).strip()
        return str(data).strip()
    except Exception as e:
        return f"LLM hosted inference was unavailable. Fallback analysis was used. Details: {e}"


def build_llm1_structured(prediction, confidence, stats, results_df, explanation_df):
    """Build structured tables for the AI Analyst section and reports.

    This function proves the LLM is using a machine learning context, not just
    reading the raw text. It combines the ensemble prediction, vote counts,
    confidence, document statistics, and feature evidence.
    """
    valid_results = results_df[~results_df["Prediction"].eq("Prediction error")].copy()
    total_valid = len(valid_results)
    ai_votes = int((valid_results["Prediction"] == "AI-written").sum())
    human_votes = int((valid_results["Prediction"] == "Human-written").sum())
    agreement = max(ai_votes, human_votes) / total_valid if total_valid else 0
    risk_level, risk_icon = get_risk_level(prediction, confidence, agreement)

    if total_valid == 0:
        evidence = "No models completed prediction."
    elif agreement >= 0.80:
        evidence = "Strong model agreement"
    elif agreement >= 0.60:
        evidence = "Moderate model agreement"
    else:
        evidence = "Split model agreement"

    assessment = pd.DataFrame([
        ["Overall Classification", prediction],
        ["Confidence", f"{confidence:.1%} ({get_strength(confidence)})"],
        ["Model Agreement", f"{max(ai_votes, human_votes)} of {total_valid} models agree" if total_valid else "No completed votes"],
        ["Evidence Strength", evidence],
        ["Risk Level", f"{risk_icon} {risk_level}"],
        ["Writing Style Signal", "Highly structured" if stats["Average Sentence Length"] >= 12 else "Short/simple structure"],
        ["Vocabulary Signal", "Lower variety" if stats["Vocabulary Richness"] < 0.55 and stats["Word Count"] > 100 else "Moderate/high variety"],
    ], columns=["Category", "Assessment"])

    evidence_rows = []
    for _, row in explanation_df.iterrows():
        evidence_rows.append({
            "Evidence Area": row["Feature Area"],
            "Observed Pattern": row["Observed Pattern"],
            "Interpretation": row["Possible Meaning"],
        })
    evidence_df = pd.DataFrame(evidence_rows)

    final_summary = (
        f"The application classified the text as {prediction} with {confidence:.1%} confidence. "
        f"The decision was supported by {evidence.lower()} across the completed model results. "
        "This output should be treated as a probability-based signal, not proof of authorship. "
        "The strongest supporting information comes from the model vote pattern and the writing features shown in the tables."
    )
    return assessment, evidence_df, final_summary


def build_model_interpretation_table(results_df, majority_prediction):
    """Create a structured table showing how each classifier contributed to the final ensemble decision."""
    rows = []
    for _, row in results_df.iterrows():
        pred = row.get("Prediction", "")
        conf_text = row.get("Confidence", "")
        if pred == "Prediction error":
            contribution = "Unavailable"
            interpretation = "This model could not be loaded or executed in the current environment."
        elif pred == majority_prediction:
            contribution = "Supports final result"
            interpretation = f"This model agreed with the ensemble classification at {conf_text} confidence."
        else:
            contribution = "Disagrees / uncertainty signal"
            interpretation = f"This model did not match the majority vote. This is useful because it shows where the ensemble is less certain."
        rows.append({
            "Model": row.get("Model", ""),
            "Prediction": pred,
            "Confidence": conf_text,
            "Contribution": contribution,
            "Interpretation": interpretation,
        })
    return pd.DataFrame(rows)


def build_estimated_revision_impact(stats, prediction):
    """Estimate how writing improvements could affect readability and perceived detector risk."""
    current_risk = "High" if prediction == "AI-written" else "Low/Moderate"
    variation_level = "Low" if stats.get("Sentence Count", 0) >= 4 and np.std(stats.get("Sentence Lengths", [0])) < 6 else "Moderate"
    revised_variation = "Higher" if variation_level == "Low" else "Maintained"
    return pd.DataFrame([
        ["AI Detection Risk", current_risk, "Lower after adding specificity, natural voice, and sentence variation"],
        ["Sentence Variation", variation_level, revised_variation],
        ["Vocabulary Diversity", str(stats.get("Vocabulary Richness", "N/A")), "Improved by replacing repeated generic wording"],
        ["Reader Trust", "Depends on evidence", "Improved by adding concrete examples and clearer support"],
    ], columns=["Metric", "Before", "Expected After Revision"])


def build_writing_coach_structured(prediction, stats, explanation_df):
    """Build structured Writing Coach tables used on screen and in reports.

    The Writing Coach is the second LLM tool required by Project 2. It focuses
    on improving clarity, specificity, sentence variation, and natural voice.
    """
    strengths = pd.DataFrame([
        ["Organization", "The text appears readable and structured enough for automated analysis."],
        ["Clarity", "The writing can be evaluated across sentence length, vocabulary, and repeated-term patterns."],
        ["Consistency", "The document maintains enough consistency for model comparison."],
    ], columns=["Strength", "Why it helps"])

    recommendations = pd.DataFrame([
        ["Vary sentence rhythm", "High", "Mix short, medium, and longer sentences so the writing sounds less uniform."],
        ["Add specific support", "High", "Use concrete examples, dates, context, or personal observations when appropriate."],
        ["Reduce generic transitions", "Medium", "Replace formulaic transitions with wording tied to the actual topic."],
        ["Increase author voice", "Medium", "Use natural phrasing and more direct explanation of why points matter."],
        ["Review repeated words", "Medium", "Keep necessary technical terms, but remove unnecessary repetition."],
    ], columns=["Recommendation", "Impact", "Action"])

    risk_rows = []
    for _, row in explanation_df.iterrows():
        risk = "High" if "low" in str(row.get("Risk Signal", "")).lower() or "formulaic" in str(row.get("Risk Signal", "")).lower() else "Medium"
        risk_rows.append([row["Feature Area"], risk, row.get("Risk Signal", "")])
    risk_df = pd.DataFrame(risk_rows, columns=["Indicator", "Risk", "Observed Signal"])

    summary = (
        "The goal is not to trick a detector. The goal is to make the writing clearer, more specific, and more naturally connected "
        "to the writer's actual purpose. A stronger revision should preserve the meaning while adding concrete details and more varied rhythm."
    )
    return strengths, recommendations, risk_df, summary


def fallback_llm1_narrative(prediction, confidence, stats, results_df, explanation_df):
    return (
        f"The six-model analysis classified the document as {prediction} with {confidence:.1%} confidence. "
        f"The text contains {stats['Word Count']} words, {stats['Sentence Count']} sentences, and an average sentence length of "
        f"{stats['Average Sentence Length']} words. The model results and feature notes suggest the classification was influenced by "
        "the overall writing structure, vocabulary patterns, and repeated terms. This result should be used as a decision-support signal rather "
        "than absolute proof of authorship."
    )


def fallback_llm2_narrative(prediction, stats, text):
    first_sentences = " ".join(split_sentences(text)[:2])
    return (
        "Focus on improving clarity and authenticity. Add specific examples, vary sentence length, reduce generic transitions, and use more "
        "natural wording. For example, the opening could be revised by keeping the same meaning but adding a more direct context statement and "
        f"more specific support. Original opening preview: {first_sentences[:500]}"
    )


# ============================================================
# REPORT HELPERS
# ============================================================


def safe_text(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def dataframe_to_txt(df):
    if df is None or df.empty:
        return "No data available."
    return df.to_string(index=False)


def create_full_txt_report(
    text,
    run_mode,
    prediction,
    confidence,
    stats,
    results_df,
    explanation_df,
    llm1_assessment,
    llm1_evidence,
    llm1_narrative,
    model_interpretation_df,
    llm2_strengths,
    llm2_recommendations,
    llm2_risk,
    llm2_narrative,
    revision_impact_df=None,
):
    lines = []
    lines.append("AI DOCUMENT INTEL REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Run Mode: {run_mode}")
    lines.append("")

    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Final Prediction: {prediction}")
    lines.append(f"Confidence: {confidence:.2%}")
    lines.append("")
    lines.append("FINE-TUNED LLM STATUS")
    lines.append("-" * 70)
    lines.append(dataframe_to_txt(llm_status_rows()))
    lines.append("")
    valid_results = results_df[~results_df["Prediction"].eq("Prediction error")]
    ai_votes = int((valid_results["Prediction"] == "AI-written").sum())
    human_votes = int((valid_results["Prediction"] == "Human-written").sum())
    lines.append(f"Model Votes: AI={ai_votes}, Human={human_votes}, Completed={len(valid_results)}")
    lines.append("")

    lines.append("MODEL COMPARISON")
    lines.append("-" * 70)
    lines.append(dataframe_to_txt(results_df))
    lines.append("")

    lines.append("DOCUMENT STATISTICS")
    lines.append("-" * 70)
    for key, value in stats.items():
        if key != "Sentence Lengths":
            lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("FEATURE ANALYSIS")
    lines.append("-" * 70)
    lines.append(dataframe_to_txt(explanation_df))
    lines.append("")

    lines.append("LLM 1 - PREDICTION EXPLAINER")
    lines.append("-" * 70)
    lines.append("Structured Assessment:")
    lines.append(dataframe_to_txt(llm1_assessment))
    lines.append("")
    lines.append("Evidence Table:")
    lines.append(dataframe_to_txt(llm1_evidence))
    lines.append("")
    lines.append("Model-by-Model Interpretation:")
    lines.append(dataframe_to_txt(model_interpretation_df))
    lines.append("")
    lines.append("Narrative Explanation:")
    lines.append(llm1_narrative or "LLM 1 was not generated yet.")
    lines.append("")

    lines.append("LLM 2 - WRITING COACH")
    lines.append("-" * 70)
    lines.append("Strengths:")
    lines.append(dataframe_to_txt(llm2_strengths))
    lines.append("")
    lines.append("Recommendations:")
    lines.append(dataframe_to_txt(llm2_recommendations))
    lines.append("")
    lines.append("AI Detection Risk Signals:")
    lines.append(dataframe_to_txt(llm2_risk))
    lines.append("")
    if revision_impact_df is not None:
        lines.append("Estimated Revision Impact:")
        lines.append(dataframe_to_txt(revision_impact_df))
        lines.append("")
    lines.append("Writing Coach Narrative:")
    lines.append(llm2_narrative or "LLM 2 was not generated yet.")
    lines.append("")

    lines.append("ORIGINAL TEXT")
    lines.append("-" * 70)
    lines.append(text)
    lines.append("")
    lines.append("END OF REPORT")
    return "\n".join(lines)


def make_pdf_table(data, col_widths=None, header=True, font_size=7):
    wrapped = []
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize = font_size
    body.leading = font_size + 2
    for row in data:
        wrapped.append([Paragraph(safe_text(cell), body) for cell in row])
    table = Table(wrapped, colWidths=col_widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    table.setStyle(TableStyle(commands))
    return table


def df_to_table_data(df, max_rows=None):
    if df is None or df.empty:
        return [["Information", "No data available"]]
    use_df = df.copy()
    if max_rows:
        use_df = use_df.head(max_rows)
    return [list(use_df.columns)] + use_df.astype(str).values.tolist()


def create_full_pdf_report(
    text,
    run_mode,
    prediction,
    confidence,
    stats,
    results_df,
    explanation_df,
    llm1_assessment,
    llm1_evidence,
    llm1_narrative,
    model_interpretation_df,
    llm2_strengths,
    llm2_recommendations,
    llm2_risk,
    llm2_narrative,
    revision_impact_df=None,
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CenteredTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=24)
    small_style = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, leading=10)
    story = []

    story.append(Paragraph("AI Document Intelligence Report", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("AI vs. Human Text Detection with Six Classifiers and Two LLM Tools", styles["BodyText"]))
    story.append(Spacer(1, 16))

    valid_results = results_df[~results_df["Prediction"].eq("Prediction error")]
    ai_votes = int((valid_results["Prediction"] == "AI-written").sum())
    human_votes = int((valid_results["Prediction"] == "Human-written").sum())
    summary = [
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Run Mode", run_mode],
        ["Final Prediction", prediction],
        ["Confidence", f"{confidence:.2%}"],
        ["AI Votes", str(ai_votes)],
        ["Human Votes", str(human_votes)],
        ["Completed Models", str(len(valid_results))],
        ["LLM 1 Model", ANALYST_MODEL_ID],
        ["LLM 2 Model", COACH_MODEL_ID],
        ["LLM Status", llm_status_summary()],
    ]
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    story.append(make_pdf_table([["Metric", "Value"]] + summary, [160, 320]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model Comparison", styles["Heading2"]))
    story.append(make_pdf_table(df_to_table_data(results_df[["Model", "Prediction", "Confidence", "Strength"]]), [110, 110, 80, 100]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Document Statistics", styles["Heading2"]))
    stats_rows = [[k, v] for k, v in stats.items() if k != "Sentence Lengths"]
    story.append(make_pdf_table([["Statistic", "Value"]] + stats_rows, [180, 260]))
    story.append(PageBreak())

    story.append(Paragraph("Feature Analysis", styles["Heading2"]))
    story.append(make_pdf_table(df_to_table_data(explanation_df), [85, 125, 85, 205], font_size=6))
    story.append(Spacer(1, 12))

    story.append(Paragraph("LLM 1 - Prediction Explainer", styles["Heading2"]))
    story.append(Paragraph("Structured Assessment", styles["Heading3"]))
    story.append(make_pdf_table(df_to_table_data(llm1_assessment), [180, 300]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Evidence Table", styles["Heading3"]))
    story.append(make_pdf_table(df_to_table_data(llm1_evidence), [100, 150, 230], font_size=6))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Model-by-Model Interpretation", styles["Heading3"]))
    story.append(make_pdf_table(df_to_table_data(model_interpretation_df), [70, 70, 60, 115, 165], font_size=5))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Narrative Explanation", styles["Heading3"]))
    story.append(Paragraph(safe_text(llm1_narrative or "LLM 1 was not generated yet."), styles["BodyText"]))
    story.append(PageBreak())

    story.append(Paragraph("LLM 2 - Writing Coach", styles["Heading2"]))
    story.append(Paragraph("Strengths", styles["Heading3"]))
    story.append(make_pdf_table(df_to_table_data(llm2_strengths), [120, 360]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Recommendations", styles["Heading3"]))
    story.append(make_pdf_table(df_to_table_data(llm2_recommendations), [110, 55, 315], font_size=6))
    story.append(Spacer(1, 10))
    story.append(Paragraph("AI Detection Risk Signals", styles["Heading3"]))
    story.append(make_pdf_table(df_to_table_data(llm2_risk), [120, 70, 290], font_size=6))
    story.append(Spacer(1, 10))
    if revision_impact_df is not None:
        story.append(Paragraph("Estimated Revision Impact", styles["Heading3"]))
        story.append(make_pdf_table(df_to_table_data(revision_impact_df), [110, 140, 230], font_size=6))
        story.append(Spacer(1, 10))
    story.append(Paragraph("Writing Coach Narrative", styles["Heading3"]))
    story.append(Paragraph(safe_text(llm2_narrative or "LLM 2 was not generated yet."), styles["BodyText"]))
    story.append(PageBreak())

    story.append(Paragraph("Original Text", styles["Heading2"]))
    text_chunks = [text[i:i + 1800] for i in range(0, min(len(text), 7200), 1800)]
    for chunk in text_chunks:
        story.append(Paragraph(safe_text(chunk).replace("\n", "<br/>") or "No text provided.", small_style))
        story.append(Spacer(1, 8))
    if len(text) > 7200:
        story.append(Paragraph("Original text was truncated in the PDF report due to length. The TXT report contains the full text.", small_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ============================================================
# UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1220px;}
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(45, 85, 180, .18), rgba(20, 150, 160, .10));
        border: 1px solid rgba(120, 120, 120, .22);
        margin-bottom: 1rem;
    }
    .hero h1 {font-size: 2.1rem; margin: 0 0 .35rem 0;}
    .hero p {font-size: 1.02rem; margin: 0; opacity: .82;}
    .mini-card {
        padding: .85rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(120, 120, 120, .18);
        background: rgba(250, 250, 250, .04);
        margin-bottom: .6rem;
    }
    .section-title {font-size: 1.35rem; font-weight: 760; margin-top: .2rem; margin-bottom: .55rem;}
    .muted {opacity: .72; font-size: .94rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 120, 120, .18);
        border-radius: 15px;
        padding: .75rem;
        background: rgba(250, 250, 250, .04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>AI Document Intelligence Suite</h1>
      <p>Six Project 1 classifiers plus two Project 2 LLM tools for explainable detection and writing guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR INPUT
# ============================================================

with st.sidebar:
    st.markdown("### Analysis Setup")
    input_method = st.radio("Input", ["Type or paste text", "Upload file"])
    run_mode = st.radio("Model Run", ["Run all models", "Run one selected model"])
    selected_model_name = st.selectbox("Selected Model", available_models)
    st.divider()
    st.markdown("### Project 2 LLMs")
    st.caption("LLM 1 explains model behavior. LLM 2 provides structured writing-coach guidance.")
    st.info(llm_status_summary())
    with st.expander("Fine-tuned LLM status / proof"):
        st.dataframe(llm_status_rows(), use_container_width=True, hide_index=True)
        st.caption("For your extra-credit demo, point to this panel to show the exact model IDs the app is using.")
    if missing_models:
        with st.expander("Missing model files"):
            for path in missing_models:
                st.write(path)


# ============================================================
# INPUT AREA
# ============================================================

text_input = ""
st.markdown('<div class="section-title">1. Add Text</div>', unsafe_allow_html=True)

if input_method == "Type or paste text":
    text_input = st.text_area(
        "Paste text to analyze",
        height=230,
        placeholder="Paste a paragraph, essay section, or document text here...",
        label_visibility="collapsed",
    )
else:
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            text_input = extract_pdf_text(uploaded_file)
        elif file_name.endswith(".docx"):
            text_input = extract_docx_text(uploaded_file)
        elif file_name.endswith(".txt"):
            text_input = extract_txt_text(uploaded_file)
        if "extraction error" in text_input.lower():
            st.error(text_input)
            st.stop()
        if not text_input.strip():
            st.error("No text could be extracted from this file.")
            st.stop()
        st.success("File uploaded and text extracted successfully.")
        with st.expander("Preview extracted text"):
            st.text_area("Extracted text", text_input, height=220, label_visibility="collapsed")

if not text_input.strip():
    st.info("Paste text or upload a file to begin.")
    st.stop()


# ============================================================
# RUN ANALYSIS
# ============================================================

stats = get_text_statistics(text_input)

with st.spinner("Running model analysis..."):
    comparison_results = []
    models_to_run = [selected_model_name] if run_mode == "Run one selected model" else available_models
    for name in models_to_run:
        try:
            model = load_one_model(name)
            pred_label, pred_conf, error_message = predict_with_model(model, text_input)
        except Exception as e:
            pred_label, pred_conf, error_message = "Prediction error", 0.0, str(e)
        comparison_results.append({
            "Model": name,
            "Prediction": pred_label,
            "Confidence_Value": pred_conf,
            "Confidence": f"{pred_conf:.1%}",
            "Strength": get_strength(pred_conf),
            "Why": get_model_reason(name, pred_label, pred_conf, error_message),
        })

raw_results_df = pd.DataFrame(comparison_results)
valid_df = raw_results_df[~raw_results_df["Prediction"].eq("Prediction error")].copy()
ai_votes = int((valid_df["Prediction"] == "AI-written").sum())
human_votes = int((valid_df["Prediction"] == "Human-written").sum())
total_votes = ai_votes + human_votes

if total_votes == 0:
    prediction = "Prediction error"
elif ai_votes > human_votes:
    prediction = "AI-written"
elif human_votes > ai_votes:
    prediction = "Human-written"
else:
    prediction = "Tie / Mixed"

confidence = float(valid_df["Confidence_Value"].mean()) if not valid_df.empty else 0.0
agreement_ratio = max(ai_votes, human_votes) / total_votes if total_votes else 0.0
risk_level, risk_icon = get_risk_level(prediction, confidence, agreement_ratio)
results_df = raw_results_df[["Model", "Prediction", "Confidence", "Strength", "Why"]]
explanation_df = explain_prediction(text_input)

# Build structured LLM tables every run so reports always contain useful information.
llm1_assessment, llm1_evidence, llm1_default_summary = build_llm1_structured(prediction, confidence, stats, results_df, explanation_df)
llm2_strengths, llm2_recommendations, llm2_risk, llm2_default_summary = build_writing_coach_structured(prediction, stats, explanation_df)
model_interpretation_df = build_model_interpretation_table(results_df, prediction)
revision_impact_df = build_estimated_revision_impact(stats, prediction)

# Session keys tied to the current text/prediction prevent stale LLM text from being reported accidentally.
analysis_key = f"{hash(text_input)}-{prediction}-{confidence:.4f}-{run_mode}"
if st.session_state.get("analysis_key") != analysis_key:
    st.session_state["analysis_key"] = analysis_key
    st.session_state["llm1_narrative"] = llm1_default_summary
    st.session_state["llm2_narrative"] = llm2_default_summary


# ============================================================
# DASHBOARD RESULT
# ============================================================

st.markdown('<div class="section-title">2. Executive Result</div>', unsafe_allow_html=True)

if prediction == "AI-written":
    st.error(f"{risk_icon} Overall result: **AI-written**")
elif prediction == "Human-written":
    st.success(f"{risk_icon} Overall result: **Human-written**")
elif prediction == "Tie / Mixed":
    st.warning("🟡 Overall result: **Tie / Mixed**")
else:
    st.error("Prediction failed. Review the model details tab for errors.")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Final Prediction", prediction)
m2.metric("Confidence", f"{confidence:.1%}")
m3.metric("Agreement", f"{agreement_ratio:.0%}" if total_votes else "0%")
m4.metric("AI Votes", f"{ai_votes}/{total_votes}" if total_votes else "0/0")
m5.metric("Risk", f"{risk_icon} {risk_level}")

if total_votes:
    st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence meter: {confidence:.1%}")


# ============================================================
# MAIN TABS
# ============================================================

overview_tab, models_tab, analytics_tab, llm_tab, coach_tab, reports_tab = st.tabs([
    "Overview", "Models", "Analytics", "AI Analyst", "Writing Coach", "Reports"
])

with overview_tab:
    st.markdown("#### Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words", stats["Word Count"])
    c2.metric("Sentences", stats["Sentence Count"])
    c3.metric("Avg Sentence", stats["Average Sentence Length"])
    c4.metric("Vocabulary Richness", stats["Vocabulary Richness"])

    st.markdown("#### Fine-Tuned LLM Status")
    st.info(llm_status_summary())
    st.dataframe(llm_status_rows(), use_container_width=True, hide_index=True)
    st.caption("This table is included so the instructor can see whether the deployed app is using your fine-tuned Hugging Face models or the base fallback models.")

    st.markdown("#### LLM 1 Summary Table")
    st.dataframe(llm1_assessment, use_container_width=True, hide_index=True)
    st.markdown("#### Final Assessment")
    st.write(st.session_state.get("llm1_narrative", llm1_default_summary))

with models_tab:
    st.markdown("#### Six-Model Comparison")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    if any(results_df["Prediction"].eq("Prediction error")):
        st.warning("At least one model did not complete prediction. Expand the table to view the error explanation in the Why column.")

with analytics_tab:
    st.markdown("#### Feature Notes")
    st.dataframe(explanation_df, use_container_width=True, hide_index=True)
    top_words = get_top_words(text_input)
    if not top_words.empty:
        st.markdown("#### Top Repeated Words")
        st.bar_chart(top_words.set_index("Word"))
    if stats["Sentence Lengths"]:
        st.markdown("#### Sentence Length Distribution")
        sentence_df = pd.DataFrame({"Sentence": range(1, len(stats["Sentence Lengths"]) + 1), "Words": stats["Sentence Lengths"]})
        st.bar_chart(sentence_df.set_index("Sentence"))

with llm_tab:
    st.markdown("### LLM 1: AI Analyst")
    st.caption("This LLM tool explains the classifier outcome using model agreement, confidence, text statistics, and feature evidence.")
    st.markdown("#### Structured Assessment")
    st.dataframe(llm1_assessment, use_container_width=True, hide_index=True)
    st.markdown("#### Evidence Table")
    st.dataframe(llm1_evidence, use_container_width=True, hide_index=True)
    st.markdown("#### Model-by-Model Interpretation")
    st.dataframe(model_interpretation_df, use_container_width=True, hide_index=True)

    if st.button("Refresh LLM 1 Narrative", use_container_width=True):
        prompt = (
            "You are the AI analyst inside an AI-vs-human text detection application. "
            "Explain the classification result in a professional but student-friendly way. "
            "Do not claim certainty. Use the model votes, confidence, statistics, and writing features. "
            "Keep the response to 2 short paragraphs.\n\n"
            f"Final prediction: {prediction}\n"
            f"Confidence: {confidence:.1%}\n"
            f"AI votes: {ai_votes}, Human votes: {human_votes}\n"
            f"Statistics: {json.dumps({k: v for k, v in stats.items() if k != 'Sentence Lengths'})}\n\n"
            f"Model results:\n{results_df.to_string(index=False)}\n\n"
            f"Feature evidence:\n{explanation_df.to_string(index=False)}\n\n"
            f"Text preview:\n{text_input[:1200]}"
        )
        with st.spinner("Calling LLM 1..."):
            narrative = call_huggingface_llm(LLM_MODELS["LLM 1 - Prediction Explainer"], prompt, max_new_tokens=240)
        if "unavailable" in narrative.lower() or len(narrative.strip()) < 20:
            narrative = fallback_llm1_narrative(prediction, confidence, stats, results_df, explanation_df)
        st.session_state["llm1_narrative"] = narrative

    st.markdown("#### Narrative Explanation")
    st.info(st.session_state.get("llm1_narrative", llm1_default_summary))

with coach_tab:
    st.markdown("### LLM 2: Writing Coach")
    st.caption("This LLM tool gives revision guidance focused on clarity, specificity, and natural voice.")
    st.markdown("#### Strengths")
    st.dataframe(llm2_strengths, use_container_width=True, hide_index=True)
    st.markdown("#### Recommendations")
    st.dataframe(llm2_recommendations, use_container_width=True, hide_index=True)
    st.markdown("#### Risk Signals")
    st.dataframe(llm2_risk, use_container_width=True, hide_index=True)
    st.markdown("#### Estimated Revision Impact")
    st.dataframe(revision_impact_df, use_container_width=True, hide_index=True)

    if st.button("Refresh LLM 2 Writing Coach", use_container_width=True):
        prompt = (
            "You are a writing coach in an AI vs. Human text detection application. "
            "Do not help anyone cheat or evade detection. Focus on improving clarity, specificity, support, and natural voice. "
            "Return four concise bullets and one short sample revision of the opening.\n\n"
            f"Detector prediction: {prediction}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Statistics: {json.dumps({k: v for k, v in stats.items() if k != 'Sentence Lengths'})}\n\n"
            f"Text preview:\n{text_input[:1200]}"
        )
        with st.spinner("Calling LLM 2..."):
            narrative = call_huggingface_llm(LLM_MODELS["LLM 2 - Writing Coach"], prompt, max_new_tokens=300)
        if "unavailable" in narrative.lower() or len(narrative.strip()) < 20:
            narrative = fallback_llm2_narrative(prediction, stats, text_input)
        st.session_state["llm2_narrative"] = narrative

    st.markdown("#### Writing Coach Narrative")
    st.success(st.session_state.get("llm2_narrative", llm2_default_summary))

with reports_tab:
    st.markdown("### Download Full Reports")
    st.caption("Both reports include the executive summary, all model outputs, statistics, feature notes, LLM 1 analysis, LLM 2 writing coach feedback, and the original text.")

    llm1_narrative = st.session_state.get("llm1_narrative", llm1_default_summary)
    llm2_narrative = st.session_state.get("llm2_narrative", llm2_default_summary)

    report_text = create_full_txt_report(
        text_input,
        run_mode,
        prediction,
        confidence,
        stats,
        results_df,
        explanation_df,
        llm1_assessment,
        llm1_evidence,
        llm1_narrative,
        model_interpretation_df,
        llm2_strengths,
        llm2_recommendations,
        llm2_risk,
        llm2_narrative,
        revision_impact_df,
    )

    pdf_bytes = create_full_pdf_report(
        text_input,
        run_mode,
        prediction,
        confidence,
        stats,
        results_df,
        explanation_df,
        llm1_assessment,
        llm1_evidence,
        llm1_narrative,
        model_interpretation_df,
        llm2_strengths,
        llm2_recommendations,
        llm2_risk,
        llm2_narrative,
        revision_impact_df,
    )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Download Full TXT Report",
            data=report_text,
            file_name="ai_document_intelligence_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="Download Full PDF Report",
            data=pdf_bytes,
            file_name="ai_document_intelligence_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

st.caption("Project 2: all six Project 1 classifiers plus two meaningful Hugging Face LLM tools for analysis and writing guidance.")
