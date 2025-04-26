from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

# Wczytanie modelu i tokenizera
model_path = "./data/zephyr-7b-sft-qlora"
# Load environment variables from .env file
load_dotenv()

# Retrieve the token from the environment
hf_token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_path, token=hf_token, torch_dtype=torch.bfloat16, device_map="auto")

# Definicja historii rozmowy zgodnie z chat_template
messages = [
    {"role": "system", "content": "Jesteś pomocnym asystentem."},
    {"role": "user", "content": "Obywatele powinni mieć łatwiejszy dostęp do broni palnej. Nie zaczynaj od Panie Marszałku"}
]

# Formatowanie prompta zgodnie z chat_template
prompt = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# Generowanie odpowiedzi
output_ids = model.generate(
    prompt,
    max_length=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# Dekodowanie odpowiedzi
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
