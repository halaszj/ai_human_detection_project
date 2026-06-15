# AI vs. Human Text Detection

This project trains and runs an **AI vs. Human Text Detector**. It includes a Jupyter Notebook for the full assignment workflow and a Streamlit app for making predictions from saved model files.

Labels used by the dataset:

- `0` = Human-written text
- `1` = AI-written text

---

## Quick Start: Run the Streamlit App in Under 10 Minutes

### 1. Put all files in one project folder

Your folder should look like this:

```text
AI-vs-Human-Text-Detection/
│
├── app.py
├── Project1_AI_vs_Human_Text_Detection.ipynb
├── README.md
├── requirements.txt
├── text_utils.py
├── train_app_models.py
├── all_models.pkl
├── train_data with labels.xlsx
│
└── models/
    ├── svm.pkl
    ├── decision_tree.pkl
    ├── adaboost.pkl
    ├── fnn.pkl
    ├── lstm.pkl
    └── cnn_for_text.pkl
```

> Important: `app.py` expects the individual model `.pkl` files to be inside the `models/` folder. 

If your Excel file has a different name, such as `train_data with labels(4).xlsx` or `train_data with labels(3)(1).xlsx`, rename it to:

```text
train_data with labels.xlsx
```

This avoids file-not-found errors when running the training scripts or notebook.

---

## 2. Create and activate a virtual environment

### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install the required packages

```bash
pip install -r requirements.txt
```

If TensorFlow gives you issues, update pip first:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Run the Streamlit app

```bash
streamlit run app.py
```

A browser window should open automatically. If it does not, copy the local URL from the terminal, usually:

```text
http://localhost:8501
```

Paste text into the box and click **Analyze Text**. The app will display whether the text is more likely **Human-written** or **AI-written**, along with the AI probability.

---

## How to Run the Jupyter Notebook

Use this when you want to reproduce the full assignment, including data exploration, preprocessing, feature engineering, model training, tuning, evaluation charts, confusion matrices, and written analysis.

```bash
jupyter notebook Project1_AI_vs_Human_Text_Detection.ipynb
```

Then run the notebook from top to bottom.

The notebook covers:

1. Data Exploration & Preprocessing  
2. Feature Engineering  
3. Model Training & Tuning  
4. Evaluation & Comparison  
5. Written Analysis  
6. Saving the best deployable model  

---

## How to Retrain All App Models

To recreate the model files inside the `models/` folder, run:

```bash
python train_app_models.py
```

This script trains and saves individual model files such as:

```text
models/svm.pkl
models/decision_tree.pkl
models/adaboost.pkl
models/fnn.pkl
models/lstm.pkl
models/cnn_for_text.pkl
```

It also saves model metrics to:

```text
models/model_metrics.csv
```

> Note: `train_app_models.py` expects the dataset to be named `train_data with labels.xlsx`.

---

## Project Files

| File/Folder | Purpose |
|---|---|
| `app.py` | Streamlit web app used to classify text as Human-written or AI-written. |
| `Project1_AI_vs_Human_Text_Detection.ipynb` | Main Jupyter Notebook with exploration, preprocessing, feature engineering, model training, tuning, and evaluation. |
| `requirements.txt` | Python packages needed to run the notebook and app. |
| `text_utils.py` | Shared text-cleaning and linguistic-feature extraction helper code. |
| `train_best_model.py` | Trains the best deployable SVM model and saves `best_ai_text_detector.pkl`. |
| `train_app_models.py` | Trains multiple app models and saves them into the `models/` folder. |
| `model_comparison_results.csv` | Saved model comparison metrics. |
| `best_ai_text_detector.pkl` | Main saved model used by the Streamlit app. |
| `models/` | Folder containing individual model `.pkl` files. |
| `train_data with labels.xlsx` | Training dataset with `text` and `label` columns. |

---

## Model Results Summary

The strongest model in the saved results was the **SVM using TF-IDF + Linguistic features**.

| Model             | Feature Set               |   Accuracy |     F1 |    AUC |   Training Time Seconds |
|:------------------|:--------------------------|-----------:|-------:|-------:|------------------------:|
| SVM               | TF-IDF + Linguistic       |     0.9657 | 0.9655 | 0.9947 |                   436.6 |
| CNN for Text      | Tokenized Sequences       |     0.9492 | 0.9476 | 0.9928 |                   127.5 |
| SVM with Word2Vec | Word2Vec Embeddings       |     0.94   | 0.9394 | 0.9879 |                    66.6 |
| AdaBoost          | TF-IDF + Linguistic       |     0.9327 | 0.9337 | 0.9846 |                   146.4 |
| FNN               | TF-IDF/SVD Dense Features |     0.9296 | 0.9298 | 0.9827 |                    14.6 |
| Decision Tree     | TF-IDF + Linguistic       |     0.8727 | 0.8727 | 0.8522 |                   168.6 |
| LSTM              | Tokenized Sequences       |     0.8158 | 0.8413 | 0.8973 |                   310.4 |

---

## Common Errors and Fixes

### `FileNotFoundError: models/svm.pkl`

The app loads individual models from the `models/` folder when it starts. Make sure this folder exists and contains the `.pkl` files:

```text
models/svm.pkl
models/decision_tree.pkl
models/adaboost.pkl
models/fnn.pkl
models/lstm.pkl
models/cnn_for_text.pkl
```

If the files are missing, run:

```bash
python train_app_models.py
```

---

### Excel dataset file not found

Make sure your dataset filename matches what the script expects.

Recommended dataset name:

```text
train_data with labels.xlsx
```

The dataset must contain these columns:

```text
text
label
```

---

### TensorFlow install problems

Try:

```bash
python -m pip install --upgrade pip
pip install tensorflow
```

Then rerun:

```bash
pip install -r requirements.txt
```

---

### Streamlit command not found

Try:

```bash
python -m streamlit run app.py
```

---

## Recommended Demo Workflow

For a quick presentation or grading demo:

1. Open the project folder.
2. Run `pip install -r requirements.txt`.
3. Run `streamlit run app.py`.
4. Paste a sample paragraph into the app.
5. Click **Analyze Text**.
6. Show the prediction and AI probability.
7. Open the notebook to show the model training and evaluation work.

---

## Notes

- The Streamlit app is intended as a support tool, not final proof that a text was written by AI.
- The best model achieved strong performance, but false positives and false negatives are still possible.
- For classroom use, the prediction should be combined with instructor judgment, student writing history, and assignment context.
