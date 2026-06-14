import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for item in X:
            if isinstance(item, (list, tuple, np.ndarray)):
                text = str(item[0])
            else:
                text = str(item)

            words = text.split()
            sentences = re.split(r"[.!?]+", text)
            sentences = [s for s in sentences if s.strip()]

            word_count = len(words)
            char_count = len(text)
            sentence_count = len(sentences)
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            avg_sentence_length = word_count / sentence_count if sentence_count else 0
            vocab_richness = len(set(words)) / word_count if word_count else 0
            punctuation_count = len(re.findall(r"[^\w\s]", text))
            uppercase_count = sum(1 for c in text if c.isupper())
            uppercase_ratio = uppercase_count / char_count if char_count else 0
            digit_count = len(re.findall(r"\d", text))
            stopword_count = sum(1 for w in words if w.lower() in ENGLISH_STOP_WORDS)
            stopword_ratio = stopword_count / word_count if word_count else 0

            features.append([
                word_count,
                char_count,
                sentence_count,
                avg_word_length,
                avg_sentence_length,
                vocab_richness,
                punctuation_count,
                uppercase_count,
                uppercase_ratio,
                digit_count,
                stopword_ratio
            ])

        return np.array(features)
