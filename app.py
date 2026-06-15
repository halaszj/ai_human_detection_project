import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from fpdf import FPDF
from text_utils import clean_text, LinguisticFeatureExtractor


st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}
.result-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
    background-color: #fafafa;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 AI vs Human Text Detection App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload or paste text to classify it as human-written or AI-generated.</div>',
    unsafe_allow_html=True
)


MODEL_FILES = {
    "SVM": "models/svm.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "AdaBoost": "models/adaboost.pkl",
    "FNN": "models/fnn.pkl",
    "LSTM": "models/lstm.pkl",
    "CNN for Text": "models/cnn_for_text.pkl",
}


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
        return "\n".join(
            p.text.strip()
            for p in document.paragraphs
            if p.text.strip()
        )

    except Exception as e:
        return f"DOCX extraction error: {e}"


def extract_txt_text(uploaded_file):
    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"TXT extraction error: {e}"


def get_text_statistics(text):
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


def make_model_input(text):
    return pd.DataFrame({
        "text": [text],
        "clean_text": [clean_text(text)]
    })


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
        return label, confidence

    except Exception as e:
        return f"Prediction error: {e}", 0.0


def explain_prediction(model, text, top_n=12):
    cleaned = clean_text(text)
    words = cleaned.split()

    if not words:
        return pd.DataFrame(columns=["Feature", "Influence", "Explanation"])

    try:
        if hasattr(model, "named_steps"):
            feature_step = model.named_steps.get("features", None)
            classifier = model.named_steps.get("classifier", None)

            if (
                feature_step is not None
                and classifier is not None
                and hasattr(classifier, "coef_")
            ):
                tfidf = feature_step.named_transformers_.get("tfidf", None)

                if tfidf is not None:
                    feature_names = tfidf.get_feature_names_out()
                    vectorized = tfidf.transform([cleaned])
                    active_indices = vectorized.nonzero()[1]

                    scores = []

                    for idx in active_indices:
                        word = feature_names[idx]
                        coef = float(classifier.coef_[0][idx])
                        direction = "AI influence" if coef > 0 else "Human influence"

                        scores.append({
                            "Feature": word,
                            "Influence": round(coef, 4),
                            "Explanation": direction
                        })

                    scores = sorted(
                        scores,
                        key=lambda x: abs(x["Influence"]),
                        reverse=True
                    )

                    return pd.DataFrame(scores[:top_n])

    except Exception:
        pass

    common_words = pd.Series(words).value_counts().head(top_n)

    return pd.DataFrame({
        "Feature": common_words.index,
        "Influence": common_words.values,
        "Explanation": "Most frequent cleaned words"
    })


def create_pdf_report(report_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    for line in report_text.split("\n"):
        safe_line = (
            line.replace("–", "-")
                .replace("—", "-")
                .replace("“", '"')
                .replace("”", '"')
                .replace("’", "'")
        )
        pdf.multi_cell(0, 5, safe_line)

    pdf_bytes = pdf.output(dest="S")

    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")

    return pdf_bytes


def create_report(text, run_mode, prediction, confidence, stats, results_df, explanation_df):
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


st.markdown("## 1. Analysis Setup")

setup_col1, setup_col2, setup_col3 = st.columns(3)

with setup_col1:
    input_method = st.radio(
        "Input Method",
        ["Type or paste text", "Upload file"],
        horizontal=False
    )

with setup_col2:
    run_mode = st.radio(
        "Analysis Mode",
        ["Run one selected model", "Run all models"],
        horizontal=False
    )

with setup_col3:
    selected_model_name = st.selectbox(
        "Model",
        list(models.keys()),
        help="Used for single-model prediction and feature explanation."
    )


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


st.markdown("## 3. Prediction Results")

if run_mode == "Run one selected model":
    selected_model = models[selected_model_name]
    prediction, confidence = predict_with_model(selected_model, text_input)

    results_df = pd.DataFrame([{
        "Model": selected_model_name,
        "Prediction": prediction,
        "Confidence": f"{confidence:.1%}"
    }])

    c1, c2, c3 = st.columns(3)

    c1.metric("Model", selected_model_name)
    c2.metric("Prediction", prediction)
    c3.metric("Confidence", f"{confidence:.1%}")

    if prediction == "AI-written":
        st.warning(
            f"The selected model predicts this text is **AI-written** "
            f"with **{confidence:.1%} confidence**."
        )
    elif prediction == "Human-written":
        st.success(
            f"The selected model predicts this text is **Human-written** "
            f"with **{confidence:.1%} confidence**."
        )
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
            "Confidence": f"{pred_conf:.1%}"
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

    results_df = raw_results_df[["Model", "Prediction", "Confidence"]]


st.markdown("## 4. Model Comparison")

if run_mode == "Run all models":
    st.write("Side-by-side comparison of all six models:")

    model_cols = st.columns(len(results_df))

    for i, row in results_df.iterrows():
        with model_cols[i]:
            st.markdown(
                f"""
                <div class="result-card">
                    <h4>{row['Model']}</h4>
                    <p><strong>{row['Prediction']}</strong></p>
                    <p>Confidence: {row['Confidence']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

else:
    st.write("Single selected model result:")

st.dataframe(results_df, width="stretch")


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


st.markdown("## 6. Feature Explanation")

explanation_model = models[selected_model_name]
explanation_df = explain_prediction(explanation_model, text_input)

st.write(
    f"Feature explanation for **{selected_model_name}**. "
    "Positive influence values lean toward AI-written text; negative values lean toward human-written text when available. "
    "For models without direct coefficients, the app displays the most frequent cleaned words."
)

st.dataframe(explanation_df, width="stretch")


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

pdf_bytes = create_pdf_report(report_text)

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


st.caption(
    "Project 1 Streamlit App: text/file input, single-model or all-model analysis, "
    "prediction, model comparison, feature explanation, text statistics, and downloadable report."
)
