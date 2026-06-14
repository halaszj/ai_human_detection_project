# Project 1: AI vs. Human Text Detection

This project includes both required parts:

1. **Jupyter Notebook** for data exploration, preprocessing, feature engineering, model training, tuning, evaluation, and written analysis.
2. **Streamlit Web Application** for non-technical users to test text or uploaded files.

## Dataset

The included dataset is:

`train_data with labels(3).xlsx`

Labels:

- `0` = Human-written text
- `1` = AI-written text

## Installation

Create/activate a Python environment, then run:

```bash
pip install -r requirements.txt
```

If you get NumPy import errors, run:

```bash
pip uninstall -y numpy pandas scipy scikit-learn tensorflow keras
pip install "numpy<2.0" pandas scipy scikit-learn tensorflow keras
pip install -r requirements.txt
```

Then restart your notebook kernel or terminal session.

## Part 1: Notebook

Open:

`Project1_AI_vs_Human_Text_Detection.ipynb`

The notebook covers:

- Data exploration and class balance visualization
- Text cleaning and tokenization
- TF-IDF features
- Word embedding features
- Linguistic features
- Six models: SVM, Decision Tree, AdaBoost, FNN, LSTM, CNN for Text
- Hyperparameter tuning with GridSearch/RandomSearch/manual tuning
- Accuracy, precision, recall, F1-score, confusion matrices, ROC curves, and written analysis prompts

## Part 2: Streamlit App

First train the app models:

```bash
python train_app_models.py
```

This creates:

`models/all_models.pkl`

Then run the app:

```bash
streamlit run app.py
```

## Streamlit App Requirements Covered

The app includes:

1. **File Upload or Text Input** — accepts `.pdf`, `.docx`, `.txt`, or pasted text.
2. **Model Selector** — lets the user choose from six trained models.
3. **Prediction Display** — shows AI/Human result and confidence score.
4. **Explanation Section** — shows influential terms and linguistic features.
5. **Text Statistics** — word count, sentence length distribution, vocabulary richness, and more.
6. **Model Comparison View** — compares predictions from all six models side by side.
7. **Report Download** — exports a `.txt` report and, if `reportlab` is installed, a PDF report.

## Notes About the App Models

The notebook contains the complete training/evaluation workflow for all six required model types. The Streamlit app uses saved deployable model pipelines so the application is easier to run locally. For full academic reporting, use the notebook metrics and discussion sections.
