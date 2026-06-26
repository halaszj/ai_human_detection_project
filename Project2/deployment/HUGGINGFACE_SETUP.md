# Hugging Face Spaces Setup

## Space Type

Choose **Streamlit**.

## Required Runtime Files

Upload these files/folders to the Space repository:

```
app.py
text_utils.py
requirements.txt
runtime.txt
README.md
model_comparison_results.csv
MODEL_MANIFEST.txt
models/
```

## Variables

Add these under **Settings → Variables and secrets → Variables**:

```
FINETUNED_ANALYST_MODEL=halaszj/ai-text-analyst-flan-t5
FINETUNED_COACH_MODEL=halaszj/ai-writing-coach-flan-t5
```

## Secrets

Add this under **Settings → Variables and secrets → Secrets**:

```replace with your token
HF_TOKEN=your_hugging_face_token
```

## Restart

After adding variables/secrets, restart the Space.

## Verification

Open the app and confirm that the Fine-Tuned LLM Status panel shows the configured model IDs.
