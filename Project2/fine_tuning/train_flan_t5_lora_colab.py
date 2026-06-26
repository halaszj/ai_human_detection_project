"""
Fine-tune FLAN-T5 with LoRA for Project 2 extra credit.

Recommended environment:
    Google Colab GPU, TTU HPCC, or another GPU runtime.

Before running:
    pip install -U transformers datasets peft accelerate bitsandbytes sentencepiece huggingface_hub

Input:
    project2_llm_instruction_data.jsonl

After training:
    upload the model to Hugging Face and set Streamlit Space secrets:
        FINETUNED_LLM1_MODEL=your-username/project2-ai-analyst-flan-t5-lora
        FINETUNED_LLM2_MODEL=your-username/project2-writing-coach-flan-t5-lora
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "google/flan-t5-small"
DATA_FILE = "project2_llm_instruction_data.jsonl"
OUTPUT_DIR = "project2-flan-t5-lora"


def format_example(example):
    task_prefix = "Project 2 AI text detection assistant:\n"
    prompt = task_prefix + example["instruction"]
    target = example["response"]
    model_inputs = tokenizer(prompt, max_length=768, truncation=True)
    labels = tokenizer(target, max_length=384, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, peft_config)

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    tokenized = dataset.map(format_example, remove_columns=dataset.column_names)

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
        push_to_hub=False,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned LoRA adapter to {OUTPUT_DIR}")
