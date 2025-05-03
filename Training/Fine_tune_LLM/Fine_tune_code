import json
from datasets import Dataset
import evaluate  # Importing the new evaluate library
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# 1. Load the saved train and test data from JSON files
with open("train_data.json", "r") as f:
    train_data = json.load(f)

with open("test_data.json", "r") as f:
    test_data = json.load(f)

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# 2. Model and tokenizer
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"  # replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # ensure padding token exists
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()}  # <- Explicit GPU device assignment
)

model = prepare_model_for_kbit_training(model)

# 3. Add LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # may vary per model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Tokenization function
def tokenize_function(examples):
    texts = ["###GDPR related case\n" + inp + "\n###Judgement\n" + out for inp, out in zip(examples["input"], examples["output"])]
    model_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=3072*2,
        return_tensors="pt"
    )
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["input", "output"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["input", "output"])

# 5. Load BLEU metric from the evaluate library
metric = evaluate.load("bleu")

# 6. Compute BLEU function
def compute_metrics(p):
    predictions, labels = p
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Remove possible padding and trailing spaces
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,  # You can set bf16=True instead if your GPU supports it better
)

# 8. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Add the BLEU evaluation here
)

# 9. Training
trainer.train()

# 10. Save model and tokenizer
model.save_pretrained("./fine_tuned_model_prompt")
tokenizer.save_pretrained("./fine_tuned_model_prompt")

# 11. Evaluation
# eval_results = trainer.evaluate()
# print(eval_results)
