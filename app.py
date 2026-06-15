import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler

models = {
    "SVM": joblib.load("models/svm.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "AdaBoost": joblib.load("models/adaboost.pkl"),
    "FNN": joblib.load("models/fnn.pkl"),
    "LSTM": joblib.load("models/lstm.pkl"),
    "CNN for Text": joblib.load("models/cnn_for_text.pkl"),
}

class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].astype(str).tolist()
        elif isinstance(X, pd.Series):
            texts = X.astype(str).tolist()
        else:
            texts = [str(x) for x in X]
        rows = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            unique_words = set(words)
            word_count = len(words)
            char_count = len(text)
            sentence_count = max(len(sentences), 1)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            avg_sentence_len = word_count / sentence_count
            type_token_ratio = len(unique_words) / word_count if word_count else 0
            punctuation_count = sum(1 for c in text if c in string.punctuation)
            comma_count = text.count(',')
            exclamation_count = text.count('!')
            question_count = text.count('?')
            uppercase_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
            rows.append([char_count, word_count, sentence_count, avg_word_len,
                         avg_sentence_len, type_token_ratio, punctuation_count,
                         comma_count, exclamation_count, question_count, uppercase_ratio])
        return np.array(rows)

stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

st.title('AI vs. Human Text Detector')
st.write('Enter text below. The model predicts whether it is more likely human-written or AI-written.')

try:
    model = joblib.load('best_ai_text_detector.pkl')
except FileNotFoundError:
    st.error('Model file not found. Run the notebook or train_best_model.py first.')
    st.stop()

user_text = st.text_area('Paste text here:', height=250)

if st.button('Analyze Text'):
    if not user_text.strip():
        st.warning('Please enter text first.')
    else:
        input_df = pd.DataFrame({'text': [user_text], 'clean_text': [clean_text(user_text)]})
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        label = 'AI-written' if prediction == 1 else 'Human-written'
        st.subheader(f'Prediction: {label}')
        st.write(f'AI probability: {probability:.2%}')
        st.progress(float(probability))