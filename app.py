import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from text_utils import clean_text, LinguisticFeatureExtractor


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🤖",
    layout="wide"
)

st.title("AI vs Human Text Detection App")
st.write(
    "Upload a document or paste text below. The app predicts whether the writing appears human-written or AI-generated."
)


# =========================
# Model loading
# =========================
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
    st.write("Expected files:")
    for path in MODEL_FILES.values():
        st.code(path)
    st.stop()

if missing_models:
    with st.expander("Some model files are missing"):
        for path in missing_models:
            st.write(path)


# =========================
# File extraction helpers
# =========================
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
        paragraphs = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text.strip())

        return "\n".join(paragraphs).strip()

    except Exception as e:
        return f"DOCX extraction error: {e}"


def extract_txt_text(uploaded_file):
    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"TXT extraction error: {e}"


# =========================
# Text statistics
# =========================
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


# =========================
# Prediction helpers
# =========================
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


def explain_prediction(model, text, top_n=10):
    cleaned = clean_text(text)
    words = cleaned.split()

    if not words:
        return pd.DataFrame(columns=["Feature / Word", "Influence"])

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
                        coef = classifier.coef_[0][idx]
                        scores.append((word, float(coef)))

                    scores = sorted(scores, key=lambda x: abs(x[1]), reverse=True)
                    return pd.DataFrame(
                        scores[:top_n],
                        columns=["Feature / Word", "Influence"]
                    )

    except Exception:
        pass

    common_words = pd.Series(words).value_counts().head(top_n)

    return pd.DataFrame({
        "Feature / Word": common_words.index,
        "Influence": common_words.values
    })


def create_report(text, selected_model, prediction, confidence, stats, comparison_df):
    report = []
    report.append("AI vs Human Text Detection Report")
    report.append("=" * 45)
    report.append("")
    report.append(f"Selected Model: {selected_model}")
    report.append(f"Prediction: {prediction}")
    report.append(f"Confidence: {confidence:.2%}")
    report.append("")
    report.append("Text Statistics")
    report.append("-" * 45)

    for key, value in stats.items():
        if key != "Sentence Lengths":
            report.append(f"{key}: {value}")

    report.append("")
    report.append("Model Comparison")
    report.append("-" * 45)
    report.append(comparison_df.to_string(index=False))
    report.append("")
    report.append("Input Text Preview")
    report.append("-" * 45)
    report.append(text[:3000])

    return "\n".join(report)


# =========================
# User input
# =========================
st.sidebar.header("Input Options")

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Type or paste text", "Upload file"]
)

text_input = ""

if input_method == "Type or paste text":
    text_input = st.text_area(
        "Enter text to analyze:",
        height=300,
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

        st.subheader("Extracted Text Preview")
        st.text_area("Extracted text", text_input, height=300)

        if "extraction error" in text_input.lower():
            st.error(text_input)
            st.stop()

        if not text_input.strip():
            st.error(
                "No text could be extracted from this file. "
                "If this is a scanned PDF, try a DOCX or TXT file instead."
            )
            st.stop()


if not text_input.strip():
    st.info("Enter text or upload a file to begin.")
    st.stop()


# =========================
# Model selector
# =========================
st.sidebar.header("Model Options")

selected_model_name = st.sidebar.selectbox(
    "Choose model:",
    list(models.keys())
)

selected_model = models[selected_model_name]


# =========================
# Prediction display
# =========================
st.header("Prediction Display")

prediction, confidence = predict_with_model(selected_model, text_input)

col1, col2 = st.columns(2)

with col1:
    st.metric("Prediction", prediction)

with col2:
    st.metric("Confidence", f"{confidence:.2%}")


# =========================
# Text statistics
# =========================
st.header("Text Statistics")

stats = get_text_statistics(text_input)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Word Count", stats["Word Count"])
c2.metric("Sentence Count", stats["Sentence Count"])
c3.metric("Avg Sentence Length", stats["Average Sentence Length"])
c4.metric("Vocabulary Richness", stats["Vocabulary Richness"])

if stats["Sentence Lengths"]:
    sentence_df = pd.DataFrame({
        "Sentence": range(1, len(stats["Sentence Lengths"]) + 1),
        "Words": stats["Sentence Lengths"]
    })

    st.subheader("Sentence Length Distribution")
    st.bar_chart(sentence_df.set_index("Sentence"))


# =========================
# Explanation section
# =========================
st.header("Explanation Section")

st.write(
    "This section shows the strongest available words/features for the selected model. "
    "For models that do not expose feature weights, it shows the most frequent cleaned words."
)

explanation_df = explain_prediction(selected_model, text_input)
st.dataframe(explanation_df, use_container_width=True)


# =========================
# Model comparison
# =========================
st.header("Model Comparison View")

comparison_results = []

for name, model in models.items():
    pred_label, pred_conf = predict_with_model(model, text_input)

    comparison_results.append({
        "Model": name,
        "Prediction": pred_label,
        "Confidence": round(pred_conf, 4)
    })

comparison_df = pd.DataFrame(comparison_results)

st.dataframe(comparison_df, use_container_width=True)


# =========================
# Report download
# =========================
st.header("Report Download")

report_text = create_report(
    text=text_input,
    selected_model=selected_model_name,
    prediction=prediction,
    confidence=confidence,
    stats=stats,
    comparison_df=comparison_df
)

st.download_button(
    label="Download Text Report",
    data=report_text,
    file_name="ai_text_detection_report.txt",
    mime="text/plain"
)


st.caption(
    "Project 1 Streamlit App: text/file input, model selector, prediction, explanation, text statistics, model comparison, and report download."
)
