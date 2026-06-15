# AI vs. Human Text Detection Project

This package contains a complete project for Project 1: AI vs. Human Text Detection.

## Files

- `Project1_AI_vs_Human_Text_Detection.ipynb` — full notebook with all required sections:
  - Data Exploration & Preprocessing
  - Feature Engineering: TF-IDF, Word2Vec embeddings, linguistic features
  - Model Training & Tuning for six required models
  - Evaluation & Comparison with metrics, confusion matrices, ROC curves, and written analysis
- `app.py` — Streamlit app for using the saved model
- `train_best_model.py` — quick script to train a deployable SVM model
- `requirements.txt` — required Python packages
- `train_data with labels(3).xlsx` — dataset

## How to Run the Notebook

```bash
pip install -r requirements.txt
jupyter notebook Project1_AI_vs_Human_Text_Detection.ipynb
```

Run the notebook from top to bottom. Some deep learning cells may take longer depending on your computer.

## How to Run the Streamlit App

First run the notebook or training script so `best_ai_text_detector.pkl` is created.

```bash
python train_best_model.py
streamlit run app.py
```

## Notes

The notebook includes written-analysis placeholders near the end. After running the models, replace the bracketed sections with your actual model results.