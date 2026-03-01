# backend/finetune/train_gemma12b.py
"""
LoRA fine-tune Gemma 3 12B on math tutoring dialogues + worksheet reading.
Run on RTX 6000: python finetune/train_gemma12b.py
Output: adapters/gemma12b/
"""
import json, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

BASE_MODEL = "google/gemma-3-12b-it"
DATA_PATH = "finetune/data/gemma12b_train.jsonl"
OUTPUT_DIR = "adapters/gemma12b"


def load_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "worksheet_read":
                text = (
                    f"<start_of_turn>user\n"
                    f"Read this worksheet image and identify wrong answers.\n"
                    f"Image shows: {r['image_description']}<end_of_turn>\n"
                    f"<start_of_turn>model\n{r['output']}<end_of_turn>"
                )
            elif r.get("type") == "face_analysis":
                text = (
                    f"<start_of_turn>user\n"
                    f"Analyze this student's face: {r['image_description']}<end_of_turn>\n"
                    f"<start_of_turn>model\n{r['output']}<end_of_turn>"
                )
            else:
                text = (
                    f"<start_of_turn>user\n"
                    f"Math tutoring context: {json.dumps(r.get('context', {}))}<end_of_turn>\n"
                    f"<start_of_turn>model\n{json.dumps(r.get('output', {}))}<end_of_turn>"
                )
            records.append({"text": text})
    return Dataset.from_list(records)


def tokenize(batch, tokenizer, max_length=512):
    out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")
    # Gemma 3 is multimodal and requires token_type_ids (0=text, 1=image).
    # All tokens are text in our training data.
    out["token_type_ids"] = [[0] * len(ids) for ids in out["input_ids"]]
    return out


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_data(DATA_PATH)
    tokenized = dataset.map(lambda b: tokenize(b, tokenizer), batched=True)
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=20,
        save_strategy="no",
        logging_steps=10,
        fp16=True,
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Gemma 3 12B adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
