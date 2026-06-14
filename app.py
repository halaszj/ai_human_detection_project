import io
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from text_utils import clean_text, text_statistics, sentence_lengths

# Optional file readers
try:
    from docx import Document
except Exception:
    Document = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None
    letter = None

st.set_page_config(page_title='AI vs. Human Text Detector', layout='wide')
st.title('AI vs. Human Text Detector')
st.write('Upload a document or paste text, choose a model, and review prediction results, text statistics, feature explanations, and model comparisons.')

MODEL_PATH = os.path.join('models', 'all_models.pkl')

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

bundle = load_models()

if bundle is None:
    st.error('Model bundle not found. Run `python train_app_models.py` first to create `models/all_models.pkl`.')
    st.stop()

models = bundle['models']
metrics_df = bundle.get('metrics', pd.DataFrame())
label_map = bundle.get('label_map', {0: 'Human-written', 1: 'AI-written'})

# ---------- File reading helpers ----------
def extract_docx(uploaded_file):
    if Document is None:
        st.warning('DOCX support requires python-docx. Install it with: pip install python-docx')
        return ''
    doc = Document(uploaded_file)
    return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_pdf(uploaded_file):
    if PdfReader is None:
        st.warning('PDF support requires pypdf. Install it with: pip install pypdf')
        return ''
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or '')
    return '\n'.join(pages)

def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ''
    name = uploaded_file.name.lower()
    if name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8', errors='ignore')
    if name.endswith('.docx'):
        return extract_docx(uploaded_file)
    if name.endswith('.pdf'):
        return extract_pdf(uploaded_file)
    return ''

# ---------- Prediction helpers ----------
def predict_with_model(model, text):
    row = pd.DataFrame({'text': [text], 'clean_text': [clean_text(text)]})
    pred = int(model.predict(row)[0])
    if hasattr(model, 'predict_proba'):
        ai_prob = float(model.predict_proba(row)[0][1])
    elif hasattr(model, 'decision_function'):
        score = float(model.decision_function(row)[0])
        ai_prob = 1 / (1 + np.exp(-score))
    else:
        ai_prob = 0.5
    confidence = ai_prob if pred == 1 else 1 - ai_prob
    return pred, ai_prob, confidence

def influential_terms(text, model, top_n=10):
    """Simple explanation: show important TF-IDF words present in this document when available."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    if not tokens:
        return pd.DataFrame({'Feature': [], 'Reason': []})

    try:
        features = model.named_steps['features']
        tfidf = features.named_transformers_['tfidf']
        vocab = set(tfidf.get_feature_names_out())
        present = [t for t in tokens if t in vocab]
        term_counts = pd.Series(present).value_counts().head(top_n)
        return pd.DataFrame({
            'Feature': term_counts.index,
            'Count in text': term_counts.values,
            'Reason': ['High-frequency model vocabulary term in this document'] * len(term_counts)
        })
    except Exception:
        return pd.DataFrame({
            'Feature': pd.Series(tokens).value_counts().head(top_n).index,
            'Count in text': pd.Series(tokens).value_counts().head(top_n).values,
            'Reason': ['Frequently used cleaned word'] * min(top_n, len(set(tokens)))
        })

def build_text_report(text, selected_model_name, selected_result, comparison_df, stats):
    pred_label, ai_prob, confidence = selected_result
    lines = []
    lines.append('AI vs. Human Text Detection Report')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    lines.append(f'Selected model: {selected_model_name}')
    lines.append(f'Prediction: {pred_label}')
    lines.append(f'AI probability: {ai_prob:.2%}')
    lines.append(f'Confidence: {confidence:.2%}')
    lines.append('')
    lines.append('Text Statistics')
    for k, v in stats.items():
        lines.append(f'- {k}: {v}')
    lines.append('')
    lines.append('Model Comparison')
    lines.append(comparison_df.to_string(index=False))
    lines.append('')
    lines.append('Input Text Preview')
    lines.append(text[:1500])
    return '\n'.join(lines)

def build_pdf_report(report_text):
    if canvas is None:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in report_text.split('\n'):
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line[:110])
        y -= 14
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ---------- Input section ----------
st.sidebar.header('Input Options')
input_mode = st.sidebar.radio('Choose input type', ['Paste text', 'Upload file'])

text_input = ''
if input_mode == 'Paste text':
    text_input = st.text_area('Paste or type text here:', height=260)
else:
    uploaded = st.file_uploader('Upload .pdf, .docx, or .txt file', type=['pdf', 'docx', 'txt'])
    if uploaded:
        text_input = read_uploaded_file(uploaded)
        st.text_area('Extracted text preview:', text_input[:5000], height=220)

selected_model_name = st.sidebar.selectbox('Choose trained model', list(models.keys()))
selected_model = models[selected_model_name]

analyze = st.button('Analyze Text', type='primary')

if analyze:
    if not text_input.strip():
        st.warning('Please enter text or upload a supported file first.')
        st.stop()

    stats = text_statistics(text_input)
    pred, ai_prob, confidence = predict_with_model(selected_model, text_input)
    selected_label = label_map.get(pred, str(pred))

    col1, col2, col3 = st.columns(3)
    col1.metric('Prediction', selected_label)
    col2.metric('AI Probability', f'{ai_prob:.2%}')
    col3.metric('Confidence', f'{confidence:.2%}')
    st.progress(ai_prob)

    st.subheader('Text Statistics')
    stat_df = pd.DataFrame(stats.items(), columns=['Statistic', 'Value'])
    st.dataframe(stat_df, use_container_width=True)

    lengths = sentence_lengths(text_input)
    st.write('Sentence length distribution')
    st.bar_chart(pd.DataFrame({'Sentence Length': lengths}))

    st.subheader('Explanation: Influential Features')
    st.write('This section gives an interpretable approximation by showing document terms and linguistic patterns that the model can use. For tree/boosting/neural models, exact internal reasoning is less transparent, so this should be treated as an explanation aid rather than a perfect proof.')
    explanation_df = influential_terms(text_input, selected_model)
    st.dataframe(explanation_df, use_container_width=True)

    st.write('Linguistic indicators used by the model include word count, average sentence length, vocabulary richness, punctuation count, and uppercase ratio.')

    st.subheader('Model Comparison View')
    comparison_rows = []
    for name, model in models.items():
        p, prob, conf = predict_with_model(model, text_input)
        comparison_rows.append({
            'Model': name,
            'Prediction': label_map.get(p, str(p)),
            'AI Probability': f'{prob:.2%}',
            'Confidence': f'{conf:.2%}'
        })
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True)

    if not metrics_df.empty:
        st.subheader('Saved Model Evaluation Metrics')
        st.dataframe(metrics_df, use_container_width=True)

    st.subheader('Report Download')
    report_text = build_text_report(
        text_input,
        selected_model_name,
        (selected_label, ai_prob, confidence),
        comparison_df,
        stats
    )
    st.download_button(
        'Download Text Report',
        data=report_text,
        file_name='ai_text_detection_report.txt',
        mime='text/plain'
    )

    pdf_bytes = build_pdf_report(report_text)
    if pdf_bytes:
        st.download_button(
            'Download PDF Report',
            data=pdf_bytes,
            file_name='ai_text_detection_report.pdf',
            mime='application/pdf'
        )
    else:
        st.info('PDF download requires reportlab. Install it with: pip install reportlab')
