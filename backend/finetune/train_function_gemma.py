# backend/finetune/train_function_gemma.py
"""
LoRA fine-tune FunctionGemma 270M on response quality classification.
Run on RTX 6000: python finetune/train_function_gemma.py
Output: adapters/function_gemma/
"""
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

BASE_MODEL = "google/functiongemma-270m-it"
DATA_PATH = "finetune/data/function_gemma_train.jsonl"
OUTPUT_DIR = "adapters/function_gemma"


def load_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            text = (
                f"<context>{json.dumps(r['input'])}</context>\n"
                f"<action>{r['output']}</action>"
            )
            records.append({"text": text})
    return Dataset.from_list(records)


def tokenize(batch, tokenizer, max_length=256):
    return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_data(DATA_PATH)
    tokenized = dataset.map(lambda b: tokenize(b, tokenizer), batched=True)
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        warmup_steps=10,
        save_strategy="no",
        logging_steps=10,
        fp16=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"FunctionGemma adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
