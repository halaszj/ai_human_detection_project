import os
import re
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None


# ---------------------------------------------------------
# Required custom class for loading saved pickle models
# ---------------------------------------------------------
class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for text in X:
            text = str(text)
            words = text.split()
            sentences = re.split(r"[.!?]+", text)
            sentences = [s for s in sentences if s.strip()]

            word_count = len(words)
            char_count = len(text)
            sentence_count = len(sentences)
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            avg_sentence_length = word_count / sentence_count if sentence_count else 0
            vocab_richness = len(set(words)) / word_count if word_count else 0

            features.append([
                word_count,
                char_count,
                sentence_count,
                avg_word_length,
                avg_sentence_length,
                vocab_richness
            ])

        return np.array(features)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def extract_pdf_text(uploaded_file):
    if PyPDF2 is None:
        return "PyPDF2 is not installed."

    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_docx_text(uploaded_file):
    if docx is None:
        return "python-docx is not installed."

    document = docx.Document(uploaded_file)
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs).strip()


def get_text_statistics(text):
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)
    unique_words = len(set([w.lower() for w in words]))

    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    vocab_richness = unique_words / word_count if word_count else 0

    sentence_lengths = [len(s.split()) for s in sentences]

    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Average Sentence Length": round(avg_sentence_length, 2),
        "Vocabulary Richness": round(vocab_richness, 3),
        "Sentence Lengths": sentence_lengths
    }


def predict_with_model(model, text):
    cleaned = clean_text(text)

    try:
        pred = model.predict([cleaned])[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([cleaned])[0]
            confidence = float(np.max(proba))
        elif hasattr(model, "decision_function"):
            score = model.decision_function([cleaned])
            confidence = float(1 / (1 + np.exp(-np.max(score))))
        else:
            confidence = 0.50

        label = "AI-written" if int(pred) == 1 else "Human-written"

        return label, confidence

    except Exception as e:
        return f"Prediction error: {e}", 0.0


def explain_prediction(model, text, top_n=10):
    cleaned = clean_text(text)

    try:
        if hasattr(model, "named_steps"):
            vectorizer = None
            classifier = None

            for step_name, step_obj in model.named_steps.items():
                if hasattr(step_obj, "get_feature_names_out"):
                    vectorizer = step_obj
                if hasattr(step_obj, "coef_"):
                    classifier = step_obj

            if vectorizer is not None and classifier is not None:
                feature_names = vectorizer.get_feature_names_out()
                vectorized_text = vectorizer.transform([cleaned])

                coefs = classifier.coef_[0]
                active_indices = vectorized_text.nonzero()[1]

                word_scores = []
                for idx in active_indices:
                    word_scores.append((feature_names[idx], coefs[idx]))

                word_scores = sorted(
                    word_scores,
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                return word_scores[:top_n]

    except Exception:
        pass

    words = cleaned.split()
    common_words = pd.Series(words).value_counts().head(top_n)

    return list(common_words.items())


def create_report(text, selected_model, prediction, confidence, stats, comparison_df):
    report = []
    report.append("AI vs Human Text Detection Report")
    report.append("=" * 40)
    report.append("")
    report.append(f"Selected Model: {selected_model}")
    report.append(f"Prediction: {prediction}")
    report.append(f"Confidence: {confidence:.2%}")
    report.append("")
    report.append("Text Statistics")
    report.append("-" * 40)

    for key, value in stats.items():
        if key != "Sentence Lengths":
            report.append(f"{key}: {value}")

    report.append("")
    report.append("Model Comparison")
    report.append("-" * 40)

    if comparison_df is not None and not comparison_df.empty:
        report.append(comparison_df.to_string(index=False))

    report.append("")
    report.append("Input Text Preview")
    report.append("-" * 40)
    report.append(text[:2000])

    return "\n".join(report)


# ---------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🤖",
    layout="wide"
)

st.title("AI vs Human Text Detection App")
st.write(
    "Upload a document or paste text, choose a model, and predict whether the text was written by a human or generated by AI."
)

MODEL_PATH = "models/all_models.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model bundle not found. Run `python train_app_models.py` first to create `models/all_models.pkl`."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)

if isinstance(bundle, dict) and "models" in bundle:
    models = bundle["models"]
else:
    models = bundle

if not isinstance(models, dict):
    st.error("The loaded model bundle is not in the expected format.")
    st.stop()


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------
st.sidebar.header("Input Options")

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Type or paste text", "Upload file"]
)

text_input = ""

if input_method == "Type or paste text":
    text_input = st.text_area(
        "Enter text to analyze:",
        height=250,
        placeholder="Paste or type text here..."
    )

else:
    uploaded_file = st.file_uploader(
        "Upload a .pdf, .docx, or .txt file",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".pdf"):
            text_input = extract_pdf_text(uploaded_file)

        elif file_name.endswith(".docx"):
            text_input = extract_docx_text(uploaded_file)

        elif file_name.endswith(".txt"):
            text_input = uploaded_file.read().decode("utf-8", errors="ignore")

        st.text_area("Extracted Text", text_input, height=250)


if not text_input.strip():
    st.info("Enter text or upload a file to begin.")
    st.stop()


# ---------------------------------------------------------
# Model selector
# ---------------------------------------------------------
model_names = list(models.keys())

selected_model_name = st.sidebar.selectbox(
    "Choose trained model:",
    model_names
)

selected_model = models[selected_model_name]


# ---------------------------------------------------------
# Main prediction
# ---------------------------------------------------------
st.header("Prediction Result")

prediction, confidence = predict_with_model(selected_model, text_input)

col1, col2 = st.columns(2)

with col1:
    st.metric("Prediction", prediction)

with col2:
    st.metric("Confidence", f"{confidence:.2%}")


# ---------------------------------------------------------
# Text statistics
# ---------------------------------------------------------
st.header("Text Statistics")

stats = get_text_statistics(text_input)

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

stat_col1.metric("Word Count", stats["Word Count"])
stat_col2.metric("Sentence Count", stats["Sentence Count"])
stat_col3.metric("Avg. Sentence Length", stats["Average Sentence Length"])
stat_col4.metric("Vocabulary Richness", stats["Vocabulary Richness"])

if stats["Sentence Lengths"]:
    st.subheader("Sentence Length Distribution")
    sentence_df = pd.DataFrame({
        "Sentence Number": range(1, len(stats["Sentence Lengths"]) + 1),
        "Word Count": stats["Sentence Lengths"]
    })
    st.bar_chart(sentence_df.set_index("Sentence Number"))


# ---------------------------------------------------------
# Explanation section
# ---------------------------------------------------------
st.header("Explanation Section")

st.write(
    "The following words or features had the strongest influence on the selected model prediction when available. "
    "For models where direct feature weights are unavailable, the app shows the most frequent cleaned words."
)

explanation = explain_prediction(selected_model, text_input)

if explanation:
    explanation_df = pd.DataFrame(
        explanation,
        columns=["Feature / Word", "Influence or Frequency"]
    )
    st.dataframe(explanation_df, use_container_width=True)
else:
    st.write("No explanation available for this model.")


# ---------------------------------------------------------
# Model comparison
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Report download
# ---------------------------------------------------------
st.header("Download Report")

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


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.caption(
    "Project 1: AI vs Human Text Detection — includes text input/upload, model selection, prediction, explanation, statistics, model comparison, and report download."
)
