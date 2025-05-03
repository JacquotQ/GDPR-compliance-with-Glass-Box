import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 1. Load input file with a single object containing "input" field
with open("input.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Assuming input.json contains an object like {"input": "some text"}
input_text = data.get("input", None)

if input_text is None:
    print("No 'input' field found in input.json.")
    exit(1)

# 2. Load model and tokenizer
model_path = "./fine_tuned_model_prompt"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Inference
model.eval()
with torch.no_grad():
    inputs = tokenizer(f"###GDPR related case\n {input_text} \n###Judgement\n", return_tensors="pt", truncation=True, max_length=3072*2, padding=True).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=3072)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# 4. Print result
# print(f"\nInput:\n{input_text}\n\nOutput:\n{output_text}")
print(f"Output:\n{output_text}")
