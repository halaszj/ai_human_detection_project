import re
import string
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

df = pd.read_excel('train_data with labels(3).xlsx')
df = df.dropna(subset=['text', 'label']).drop_duplicates(subset=['text']).copy()
df['label'] = df['label'].astype(int)
df['clean_text'] = df['text'].apply(clean_text)

X = df[['text', 'clean_text']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

features = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95), 'clean_text'),
        ('ling', Pipeline([
            ('features', LinguisticFeatureExtractor()),
            ('scale', StandardScaler())
        ]), ['text'])
    ]
)

pipeline = Pipeline([
    ('features', features),
    ('svm', SVC(probability=True, random_state=42))
])

params = {
    'svm__C': [0.5, 1, 2],
    'svm__kernel': ['linear', 'rbf']
}

grid = GridSearchCV(pipeline, params, scoring='f1', cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

preds = grid.predict(X_test)
print('Best parameters:', grid.best_params_)
print('Accuracy:', accuracy_score(y_test, preds))
print(classification_report(y_test, preds, target_names=['Human', 'AI']))

joblib.dump(grid.best_estimator_, 'best_ai_text_detector.pkl')
print('Saved model to best_ai_text_detector.pkl')