import re
import string
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)
LINGUISTIC_FEATURE_NAMES = [
    'char_count', 'word_count', 'sentence_count', 'avg_word_length',
    'avg_sentence_length', 'vocab_richness', 'punctuation_count',
    'comma_count', 'exclamation_count', 'question_count', 'uppercase_ratio'
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 1]
    return ' '.join(tokens)

def sentence_lengths(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', str(text)) if s.strip()]
    lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    return lengths if lengths else [0]

def text_statistics(text):
    text = str(text)
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = set(words)
    lengths = sentence_lengths(text)
    return {
        'Word Count': len(words),
        'Sentence Count': len(lengths),
        'Average Sentence Length': round(float(np.mean(lengths)), 2) if lengths else 0,
        'Shortest Sentence': int(np.min(lengths)) if lengths else 0,
        'Longest Sentence': int(np.max(lengths)) if lengths else 0,
        'Vocabulary Richness': round(len(unique_words) / len(words), 3) if words else 0,
        'Character Count': len(text),
        'Punctuation Count': sum(1 for c in text if c in string.punctuation)
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
            words = re.findall(r'\b\w+\b', text.lower())
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            unique_words = set(words)
            word_count = len(words)
            char_count = len(text)
            sentence_count = max(len(sentences), 1)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            avg_sentence_len = word_count / sentence_count
            vocab_richness = len(unique_words) / word_count if word_count else 0
            punctuation_count = sum(1 for c in text if c in string.punctuation)
            comma_count = text.count(',')
            exclamation_count = text.count('!')
            question_count = text.count('?')
            uppercase_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
            rows.append([
                char_count, word_count, sentence_count, avg_word_len,
                avg_sentence_len, vocab_richness, punctuation_count,
                comma_count, exclamation_count, question_count, uppercase_ratio
            ])
        return np.array(rows)
