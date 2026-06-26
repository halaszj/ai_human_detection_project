# Project 2 Fine-Tuning Package — Self-Contained Colab Notebook

This package fixes/replaces the earlier broken fine-tuning notebooks. I decided this would be the best approach based on the many errors I encountered with Google Colab and the free edition.

## What makes this different?

This is self-contained:

- No `colab_requirements.txt` file needed
- No external `data/` folder needed
- No JSONL files to upload
- No separate training data files
- No broken f-strings
- No deprecated `tokenizer=` argument in `Seq2SeqTrainer`

The notebook generates the training data internally, fine-tunes two FLAN-T5 models, and pushes them to my Hugging Face account.

## Output model repos

The notebook is configured for my Hugging Face username `halaszj`.

It pushes:

1. `halaszj/ai-text-analyst-flan-t5`
2. `halaszj/ai-writing-coach-flan-t5`

## How to run

1. Go to Google Colab.
2. Upload/open `Project2_Self_Contained_FineTune.ipynb`.
3. Select `Runtime > Change runtime type > T4 GPU`.
4. Run all cells from top to bottom.
5. When prompted, paste your Hugging Face write token.
6. After training finishes, verify the two model repos exist at `https://huggingface.co/halaszj`.
7. In your Streamlit Space, set Variables:
   - `FINETUNED_ANALYST_MODEL = halaszj/ai-text-analyst-flan-t5`
   - `FINETUNED_COACH_MODEL = halaszj/ai-writing-coach-flan-t5`
8. Restart your Space.