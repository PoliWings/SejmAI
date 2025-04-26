import io
import json
import os
import requests

assert "LLM_USERNAME" in os.environ, f'Environment variable LLM_USERNAME must be set'
assert "LLM_PASSWORD" in os.environ, f'Environment variable LLM_PASSWORD must be set'

base_url = "https://153.19.239.239/api/train"
auth = (os.getenv("LLM_USERNAME"), os.getenv("LLM_PASSWORD"))
auth_kwargs = {
    'auth': auth,
    'verify': False,
}

def create_record(question: str, answer: str):
    msg = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return {
        "messages": msg,
    }


dataset_filename = 'example_dataset.json'
dataset = {
    "train": [
        create_record('Ile to jest 2+2?', '4'),
        create_record('Ile to jest 4+4?', '8')
    ],
    "validation": [
        create_record('Ile to jest 3+3?', '6'),
        create_record('Ile to jest 1+1?', '2')
    ]
}

response = requests.post(
    f"{base_url}/dataset/sft",
    headers={
        'Accept': 'application/json',
    },
    files={"file": (dataset_filename, io.BytesIO(json.dumps(dataset).encode('utf-8')))},
    **auth_kwargs,
)
print(response.text)
response.raise_for_status()

assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY environment variable is not set."
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb_project = 'huggingface'  #configure your wandb project

registered_name = "grupa5__lora_module"  # this must be unique; please use your group no. (project no.) in the name
payload = {
    "registered_name": registered_name,
    "model": "speakleash/Bielik-11B-v2.2-Instruct",
    "chat_template": None,  # Use existing chat template of Bielik 
    "max_length": 16,  # set to 512 for real dataset. 16 here is to ensure that 1 full batch can be created
    "dataset": dataset_filename,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1, # set to 8 for real dataset. 1 here is to ensure that 1 full batch can be created
    "per_device_eval_batch_size": 1, # set to 8 for real dataset. 1 here is to ensure that 1 full batch can be created
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.0,
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "eval_steps": 1,
    "save_steps": 1,
    "logging_steps": 1,
    "wandb_token": wandb_api_key,
    "wandb_project": wandb_project
}

response = requests.post(
    f"{base_url}/train/sft",
    headers={
        "Content-Type": "application/json",
        'Accept': 'application/json'
    },
    json=payload,
    **auth_kwargs,
)
print(response.json())
response.raise_for_status()

response_json = response.json()
training_id = response_json['training_id']
print(f'Training started; training id is: {training_id}')

# wait for the training to finish

from time import sleep

is_running = True
while is_running:
    sleep(30)

    response = requests.get(
        f"{base_url}/train/status/{training_id}",
        headers={
            'Accept': 'application/json'
        },
        **auth_kwargs,
    )
    response.raise_for_status()

    response_json = response.json()
    is_running = response_json['is_running']
    last_logs = response_json['logs']
    print(f'Training status: {"running" if is_running else "finished"}; last logs:')
    for log in last_logs:
        print(log)

registered_lora_adapter_name = response_json['registered_lora_adapter_name']
registered_lora_adapter_version = response_json['registered_lora_adapter_version']
print(registered_lora_adapter_name)
print(registered_lora_adapter_version)

# Download full training logs to a file (if needed you can download logs while the training process is running)
response = requests.get(
    f"{base_url}/train/logs/{training_id}",
    **auth_kwargs,
)
print(response.text)
response.raise_for_status()

with open('training.log', 'w') as tmp_file:
    tmp_file.write(response.text)

# Cancel training, e.g. to start new one with different params
response = requests.delete(
    f"{base_url}/train/cancel/{training_id}",
    headers={
        'Accept': 'application/json'
    },
    **auth_kwargs,
)
print(response.json())
response.raise_for_status()

# send requests to LLM Service to load the LoRA adapter
bielik_url = "https://153.19.239.239/api/llm"

lora_data = {
    'lora_adapter': registered_lora_adapter_name,
    'lora_adapter_version': registered_lora_adapter_version,
}
response = requests.post(
        f'{bielik_url}/lora',
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        json=lora_data,
        **auth_kwargs,
    )
print(response.json())
response.raise_for_status()

# prompt the model with the trained LoRA adapter loaded:

system_prompt = "Odpowiadaj krótko i zwięźle"

def send_chat_prompt_with_lora(prompt):
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_length": 16,  # adjust as needed
        "temperature": 0.7,
        "lora_adapter": registered_lora_adapter_name,
    }

    response = requests.put(
        f'{bielik_url}/prompt/chat',
        json=data,
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        **auth_kwargs,
    )
    response.raise_for_status()

    response_json = response.json()
    print(response_json)


send_chat_prompt_with_lora('Ile to jest 2+2?')

# OR you can download the trained LoRA adapter and use it on your own machine using HuggingFace PEFT models: https://huggingface.co/docs/peft/en/tutorial/peft_model_config#peft-models

response = requests.get(
    f"{base_url}/train/model/{training_id}",
    **auth_kwargs,
)
response.raise_for_status()

with open(f'{registered_lora_adapter_name}_{registered_lora_adapter_version}.zip', 'wb') as tmp_file:
    tmp_file.write(response.content)