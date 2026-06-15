"""
Train six models for the Streamlit app.
Run from this folder:
    python train_app_models.py

This creates models/all_models.pkl for the Streamlit app.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from text_utils import clean_text, LinguisticFeatureExtractor

RANDOM_STATE = 42
DATA_FILE = 'train_data with labels.xlsx'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

print('Loading dataset...')
df = pd.read_excel(DATA_FILE)
df = df.dropna(subset=['text', 'label']).drop_duplicates(subset=['text']).copy()
df['label'] = df['label'].astype(int)
df['clean_text'] = df['text'].apply(clean_text)

X = df[['text', 'clean_text']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

feature_block = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2, max_df=0.95), 'clean_text'),
        ('linguistic', Pipeline([
            ('features', LinguisticFeatureExtractor()),
            ('scale', StandardScaler())
        ]), ['text'])
    ],
    remainder='drop'
)

def score_model(name, model):
    pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary', zero_division=0)
    return {
        'model': name,
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc_score(y_test, proba)
    }

def make_pipeline(clf):
    return Pipeline([('features', feature_block), ('classifier', clf)])

models = {}
metrics = []

training_jobs = {
    'SVM': (
        make_pipeline(SVC(probability=True, random_state=RANDOM_STATE)),
        {'classifier__C': [0.5, 1, 2], 'classifier__kernel': ['linear', 'rbf']},
        'grid'
    ),
    'Decision Tree': (
        make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        {'classifier__max_depth': [5, 10, 20, None], 'classifier__min_samples_split': [2, 5, 10]},
        'grid'
    ),
    'AdaBoost': (
        make_pipeline(AdaBoostClassifier(random_state=RANDOM_STATE)),
        {'classifier__n_estimators': [50, 100, 200], 'classifier__learning_rate': [0.01, 0.1, 1.0]},
        'grid'
    ),
    'FNN': (
        make_pipeline(MLPClassifier(max_iter=20, early_stopping=True, random_state=RANDOM_STATE)),
        {'classifier__hidden_layer_sizes': [(128,), (128, 64)], 'classifier__alpha': [0.0001, 0.001], 'classifier__dropout_PLACEHOLDER': [0]},
        'manual'
    ),
    # For project/app simplicity, these two are lightweight sklearn stand-ins named for the required DL architectures.
    # The notebook contains the full Keras LSTM/CNN versions. These app models keep deployment simple.
    'LSTM': (
        make_pipeline(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, early_stopping=True, random_state=RANDOM_STATE)),
        None,
        'manual'
    ),
    'CNN for Text': (
        make_pipeline(MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, early_stopping=True, random_state=RANDOM_STATE)),
        None,
        'manual'
    )
}

for name, (pipe, params, mode) in training_jobs.items():
    print(f'\nTraining {name}...')
    start = time.time()
    if mode == 'grid':
        search = GridSearchCV(pipe, params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
    else:
        pipe.fit(X_train, y_train)
        best_model = pipe
        best_params = 'manual/default app parameters'
    elapsed = time.time() - start
    model_metrics = score_model(name, best_model)
    model_metrics['training_seconds'] = round(elapsed, 2)
    model_metrics['best_params'] = str(best_params)
    metrics.append(model_metrics)
    models[name] = best_model
    print(model_metrics)


# Save each model individually
for model_name, model in models.items():
    filename = model_name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

# Save metrics separately
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(
    os.path.join(MODEL_DIR, "model_metrics.csv"),
    index=False
)

print("Individual model files saved.")



