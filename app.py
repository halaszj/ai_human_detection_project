# ============================================================
# AI vs Human Text Detection Streamlit App
# Project 1 - Streamlit Web Application
#
# App Features:
# - Type/paste text input
# - PDF, DOCX, and TXT upload
# - Single-model or all-model prediction
# - Detailed model comparison table
# - Human-readable feature explanation
# - Text statistics
# - TXT and PDF report downloads
# ============================================================

import os
import re
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from text_utils import clean_text, LinguisticFeatureExtractor


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Project 1 - AI vs Human Text Detector",
    layout="wide"
)

st.markdown("# Project 1 - AI vs Human Text Detection App")
st.write("Upload or paste text to classify it as human-written or AI-generated.")


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

@st.cache_resource
def load_models():
    models = {}
    missing = []

    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            missing.append(path)

    return models, missing


models, missing_models = load_models()

if not models:
    st.error("No trained models were found in the models folder.")
    st.stop()

if missing_models:
    with st.expander("Missing model files"):
        for path in missing_models:
            st.write(path)


# ============================================================
# FILE EXTRACTION FUNCTIONS
# ============================================================

def extract_pdf_text(uploaded_file):
    """Extract text from a PDF file."""
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
    """Extract text from a DOCX file."""
    try:
        import docx

        document = docx.Document(uploaded_file)

        return "\n".join(
            p.text.strip()
            for p in document.paragraphs
            if p.text.strip()
        )

    except Exception as e:
        return f"DOCX extraction error: {e}"


def extract_txt_text(uploaded_file):
    """Extract text from a TXT file."""
    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"TXT extraction error: {e}"


# ============================================================
# TEXT STATISTICS
# ============================================================

def get_text_statistics(text):
    """Calculate basic writing statistics."""
    words = text.split()

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)
    unique_words = len(set(w.lower() for w in words))

    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    vocab_richness = unique_words / word_count if word_count else 0

    sentence_lengths = [len(s.split()) for s in sentences]

    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Average Sentence Length": round(avg_sentence_length, 2),
        "Vocabulary Richness": round(vocab_richness, 3),
        "Sentence Lengths": sentence_lengths,
    }


# ============================================================
# MODEL PREDICTION FUNCTIONS
# ============================================================

def make_model_input(text):
    """Create the same DataFrame structure used during model training."""
    return pd.DataFrame({
        "text": [text],
        "clean_text": [clean_text(text)]
    })


def predict_with_model(model, text):
    """Run one trained model and return prediction plus confidence."""
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
        return label, confidence

    except Exception as e:
        return f"Prediction error: {e}", 0.0


def get_strength(confidence):
    """Convert confidence into a readable strength label."""
    if confidence >= 0.90:
        return "Very strong"
    if confidence >= 0.75:
        return "Strong"
    if confidence >= 0.60:
        return "Moderate"
    return "Low"


def get_model_reason(model_name, prediction, confidence):
    """Generate model-specific explanation text."""
    if "Prediction error" in prediction:
        return "This model could not complete the prediction."

    model_notes = {
        "SVM": "SVM compares high-dimensional TF-IDF vocabulary patterns.",
        "Decision Tree": "Decision Tree uses rule-based splits on text and linguistic features.",
        "AdaBoost": "AdaBoost combines several weak learners and focuses on difficult examples.",
        "FNN": "FNN learns combinations of TF-IDF and linguistic features.",
        "LSTM": "This app version uses a lightweight neural-style model trained on engineered text features.",
        "CNN for Text": "This app version uses a lightweight neural-style model trained to approximate phrase-pattern detection.",
    }

    direction = (
        "The text is closer to AI-labeled examples in the training data."
        if prediction == "AI-written"
        else "The text is closer to human-labeled examples in the training data."
    )

    return f"{model_notes.get(model_name, 'This model uses trained text features.')} {direction}"


# ============================================================
# HUMAN-READABLE FEATURE EXPLANATION
# ============================================================

def explain_prediction(text):
    """
    Create a readable feature explanation.

    Instead of repeating the model comparison table, this section explains
    writing characteristics that likely influenced the models.
    """
    stats = get_text_statistics(text)
    cleaned = clean_text(text)
    words = cleaned.split()

    long_words = [w for w in words if len(w) >= 8]
    repeated_words = (
        pd.Series(words).value_counts().head(8)
        if words
        else pd.Series(dtype=int)
    )

    explanation_items = []

    explanation_items.append({
        "Feature Area": "Text Length",
        "Observed Pattern": f"{stats['Word Count']} total words",
        "Possible Meaning": (
            "Longer samples give the models more evidence. Very short passages "
            "can be harder to classify reliably."
        )
    })

    explanation_items.append({
        "Feature Area": "Sentence Structure",
        "Observed Pattern": (
            f"Average sentence length is "
            f"{stats['Average Sentence Length']} words"
        ),
        "Possible Meaning": (
            "Very even, polished, or repetitive sentence patterns can sometimes "
            "look more like AI-generated writing."
        )
    })

    explanation_items.append({
        "Feature Area": "Vocabulary Richness",
        "Observed Pattern": (
            f"Vocabulary richness score is "
            f"{stats['Vocabulary Richness']}"
        ),
        "Possible Meaning": (
            "Higher vocabulary variety may suggest natural writing, while lower "
            "variety can suggest repetition or formulaic structure."
        )
    })

    if long_words:
        explanation_items.append({
            "Feature Area": "Advanced Vocabulary",
            "Observed Pattern": ", ".join(long_words[:10]),
            "Possible Meaning": (
                "Frequent formal or abstract vocabulary may push some models "
                "toward an AI-written prediction."
            )
        })

    if not repeated_words.empty:
        repeated_text = ", ".join(
            f"{word} ({count})"
            for word, count in repeated_words.items()
        )

        explanation_items.append({
            "Feature Area": "Repeated Terms",
            "Observed Pattern": repeated_text,
            "Possible Meaning": (
                "Repeated words influence TF-IDF patterns and may affect whether "
                "the text appears formulaic or natural."
            )
        })

    return pd.DataFrame(explanation_items)


# ============================================================
# REPORT GENERATION
# ============================================================

def create_report(text, run_mode, prediction, confidence, stats, results_df, explanation_df):
    """Create a plain-text report for TXT download."""
    report = []

    report.append("AI vs Human Text Detection Report")
    report.append("=" * 45)
    report.append("")
    report.append(f"Run Mode: {run_mode}")
    report.append(f"Final Prediction: {prediction}")
    report.append(f"Confidence: {confidence:.2%}")
    report.append("")

    report.append("Text Statistics")
    report.append("-" * 45)

    for key, value in stats.items():
        if key != "Sentence Lengths":
            report.append(f"{key}: {value}")

    report.append("")
    report.append("Model Results")
    report.append("-" * 45)
    report.append(results_df.to_string(index=False))

    report.append("")
    report.append("Feature Explanation")
    report.append("-" * 45)
    report.append(explanation_df.to_string(index=False))

    report.append("")
    report.append("Input Text Preview")
    report.append("-" * 45)
    report.append(text[:3000])

    return "\n".join(report)


def clean_html_text(value):
    """Clean text before inserting into PDF paragraphs."""
    value = str(value)
    value = value.replace("&", "&amp;")
    value = value.replace("<", "&lt;")
    value = value.replace(">", "&gt;")
    return value


def create_pdf_report(run_mode, prediction, confidence, stats, results_df, explanation_df, text):
    """Create a clean PDF report using ReportLab."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI vs Human Text Detection Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Prediction Summary", styles["Heading2"]))

    summary_table = Table([
        ["Run Mode", run_mode],
        ["Final Prediction", prediction],
        ["Confidence", f"{confidence:.2%}"],
    ], colWidths=[150, 330])

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Text Statistics", styles["Heading2"]))

    stats_data = [["Metric", "Value"]]

    for key, value in stats.items():
        if key != "Sentence Lengths":
            stats_data.append([key, str(value)])

    stats_table = Table(stats_data, colWidths=[220, 260])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    story.append(stats_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Model Results", styles["Heading2"]))

    model_data = [["Model", "Prediction", "Confidence", "Strength"]]

    for _, row in results_df.iterrows():
        model_data.append([
            str(row.get("Model", "")),
            str(row.get("Prediction", "")),
            str(row.get("Confidence", "")),
            str(row.get("Strength", "")),
        ])

    model_table = Table(model_data, colWidths=[120, 120, 90, 100])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    story.append(model_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Model Explanation", styles["Heading2"]))

    if "Why" in results_df.columns:
        for _, row in results_df.iterrows():
            story.append(
                Paragraph(
                    f"<b>{clean_html_text(row['Model'])}:</b> "
                    f"{clean_html_text(row['Why'])}",
                    styles["BodyText"]
                )
            )
            story.append(Spacer(1, 6))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Feature Explanation", styles["Heading2"]))

    for _, row in explanation_df.iterrows():
        story.append(
            Paragraph(
                f"<b>{clean_html_text(row['Feature Area'])}</b>: "
                f"{clean_html_text(row['Observed Pattern'])}<br/>"
                f"{clean_html_text(row['Possible Meaning'])}",
                styles["BodyText"]
            )
        )
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Input Text Preview", styles["Heading2"]))

    preview = clean_html_text(text[:1500]).replace("\n", "<br/>")
    story.append(Paragraph(preview, styles["BodyText"]))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# ============================================================
# 1. ANALYSIS SETUP
# ============================================================

st.markdown("## 1. Analysis Setup")

setup_col1, setup_col2, setup_col3 = st.columns(3)

with setup_col1:
    input_method = st.radio(
        "Input Method",
        ["Type or paste text", "Upload file"]
    )

with setup_col2:
    run_mode = st.radio(
        "Analysis Mode",
        ["Run one selected model", "Run all models"]
    )

with setup_col3:
    selected_model_name = st.selectbox(
        "Model",
        list(models.keys())
    )


# ============================================================
# 2. TEXT INPUT
# ============================================================

st.markdown("## 2. Text Input")

text_input = ""

if input_method == "Type or paste text":
    text_input = st.text_area(
        "Enter text to analyze",
        height=280,
        placeholder="Paste or type text here..."
    )

else:
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"]
    )

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

        with st.expander("View extracted text"):
            st.text_area("Extracted text", text_input, height=250)

if not text_input.strip():
    st.info("Enter text or upload a file to begin.")
    st.stop()


# ============================================================
# 3. PREDICTION RESULTS
# ============================================================

st.markdown("## 3. Prediction Results")

if run_mode == "Run one selected model":
    selected_model = models[selected_model_name]
    prediction, confidence = predict_with_model(selected_model, text_input)

    strength = get_strength(confidence)
    reason = get_model_reason(selected_model_name, prediction, confidence)

    results_df = pd.DataFrame([{
        "Model": selected_model_name,
        "Prediction": prediction,
        "Confidence": f"{confidence:.1%}",
        "Strength": strength,
        "Why": reason
    }])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", selected_model_name)
    c2.metric("Prediction", prediction)
    c3.metric("Confidence", f"{confidence:.1%}")
    c4.metric("Strength", strength)

    if prediction == "AI-written":
        st.warning(f"The selected model predicts this text is **AI-written**.")
    elif prediction == "Human-written":
        st.success(f"The selected model predicts this text is **Human-written**.")
    else:
        st.error(prediction)

else:
    comparison_results = []

    for name, model in models.items():
        pred_label, pred_conf = predict_with_model(model, text_input)

        comparison_results.append({
            "Model": name,
            "Prediction": pred_label,
            "Confidence_Value": pred_conf,
            "Confidence": f"{pred_conf:.1%}",
            "Strength": get_strength(pred_conf),
            "Why": get_model_reason(name, pred_label, pred_conf)
        })

    raw_results_df = pd.DataFrame(comparison_results)

    ai_votes = int((raw_results_df["Prediction"] == "AI-written").sum())
    human_votes = int((raw_results_df["Prediction"] == "Human-written").sum())
    total_votes = ai_votes + human_votes

    ai_percent = ai_votes / total_votes if total_votes else 0
    human_percent = human_votes / total_votes if total_votes else 0

    if ai_votes > human_votes:
        prediction = "AI-written"
    elif human_votes > ai_votes:
        prediction = "Human-written"
    else:
        prediction = "Tie / Mixed"

    confidence = raw_results_df["Confidence_Value"].mean()

    summary_df = pd.DataFrame([
        {"Score": "Final Prediction", "Result": prediction},
        {"Score": "AI Votes", "Result": f"{ai_votes}/{total_votes} ({ai_percent:.1%})"},
        {"Score": "Human Votes", "Result": f"{human_votes}/{total_votes} ({human_percent:.1%})"},
        {"Score": "Average Confidence", "Result": f"{confidence:.1%}"},
    ])

    st.subheader("Scoring Summary")
    st.table(summary_df)

    if ai_percent >= 0.80:
        st.error(f"Strong AI consensus: {ai_percent:.0%} of models agree.")
    elif human_percent >= 0.80:
        st.success(f"Strong human consensus: {human_percent:.0%} of models agree.")
    else:
        st.warning("Mixed results. The models do not strongly agree.")

    results_df = raw_results_df[
        ["Model", "Prediction", "Confidence", "Strength", "Why"]
    ]


# ============================================================
# 4. MODEL COMPARISON TABLE
# ============================================================

st.markdown("## 4. Model Comparison Table")
st.write("Detailed model results with confidence and model-specific explanation.")

st.dataframe(results_df, width="stretch")


# ============================================================
# 5. TEXT STATISTICS
# ============================================================

st.markdown("## 5. Text Statistics")

stats = get_text_statistics(text_input)

stats_df = pd.DataFrame([
    {"Metric": "Word Count", "Value": stats["Word Count"]},
    {"Metric": "Sentence Count", "Value": stats["Sentence Count"]},
    {"Metric": "Average Sentence Length", "Value": stats["Average Sentence Length"]},
    {"Metric": "Vocabulary Richness", "Value": stats["Vocabulary Richness"]},
])

st.table(stats_df)

if stats["Sentence Lengths"]:
    sentence_df = pd.DataFrame({
        "Sentence": range(1, len(stats["Sentence Lengths"]) + 1),
        "Words": stats["Sentence Lengths"]
    })

    st.subheader("Sentence Length Distribution")
    st.bar_chart(sentence_df.set_index("Sentence"))


# ============================================================
# 6. FEATURE EXPLANATION
# ============================================================

st.markdown("## 6. Feature Explanation")

explanation_df = explain_prediction(text_input)

st.write(
    "This section explains writing patterns that may have influenced the prediction."
)

st.dataframe(explanation_df, width="stretch")


# ============================================================
# 7. REPORT DOWNLOAD
# ============================================================

st.markdown("## 7. Report Download")

report_text = create_report(
    text=text_input,
    run_mode=run_mode,
    prediction=prediction,
    confidence=confidence,
    stats=stats,
    results_df=results_df,
    explanation_df=explanation_df
)

pdf_bytes = create_pdf_report(
    run_mode=run_mode,
    prediction=prediction,
    confidence=confidence,
    stats=stats,
    results_df=results_df,
    explanation_df=explanation_df,
    text=text_input
)

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.download_button(
        label="📄 Download TXT Report",
        data=report_text,
        file_name="ai_text_detection_report.txt",
        mime="text/plain"
    )

with download_col2:
    st.download_button(
        label="📑 Download PDF Report",
        data=pdf_bytes,
        file_name="ai_text_detection_report.pdf",
        mime="application/pdf"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Project 1 Streamlit App: text/file input, single-model or all-model analysis, "
    "prediction, model comparison, feature explanation, text statistics, and reports."
)
