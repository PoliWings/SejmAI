import os
import requests
from time import sleep
import argparse
import urllib3
from dotenv import load_dotenv

# ===================== Load .env =====================
load_dotenv("../.env")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===================== Argument Parsing =====================
parser = argparse.ArgumentParser(description="Fine-tuning management script")
parser.add_argument(
    "--start",
    choices=["left", "right"],
    help="Start fine-tuning using the specified dataset side",
)
parser.add_argument("--status", action="store_true", help="Check fine-tuning status")
parser.add_argument("--cancel", action="store_true", help="Cancel fine-tuning")
parser.add_argument("--load-lora", choices=["left", "right"], help="Load LoRA adapter into model")
parser.add_argument("--unload-lora", choices=["left", "right"], help="Unload LoRA adapter from model")
parser.add_argument("--download-lora", choices=["left", "right"], help="Download trained LoRA adapter")
parser.add_argument("--version", type=str, help="Specify LoRA adapter version to load/delete/download")
args = parser.parse_args()

# ===================== Env Variables =====================
required_vars = ["LLM_URL", "LLM_USERNAME", "LLM_PASSWORD"]
for var in required_vars:
    assert var in os.environ, f"Environment variable {var} must be set in .env"

train_url = f"{os.getenv('LLM_URL')}/train"
llm_url = f"{os.getenv('LLM_URL')}/llm"
auth = (os.getenv("LLM_USERNAME"), os.getenv("LLM_PASSWORD"))
auth_kwargs = {
    "auth": auth,
    "verify": False,
}

# ===================== Start =====================
if args.start in ["left", "right"]:
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY must be set in .env"
    side = args.start
    dataset_filename = f"{side}_model_sft.json"
    dataset_path = os.path.join("sft", dataset_filename)

    assert os.path.isfile(dataset_path), f"Dataset file not found: {dataset_path}"

    with open(dataset_path, "rb") as f:
        response = requests.post(
            f"{train_url}/dataset/sft",
            headers={"Accept": "application/json"},
            files={"file": (dataset_filename, f)},
            **auth_kwargs,
        )
    response.raise_for_status()

    project_name = f"opposing_views__{side}_lora_module"

    payload = {
        "registered_name": project_name,
        "model": "speakleash/Bielik-11B-v2.2-Instruct",
        "chat_template": None,
        "max_length": 2048,
        "dataset": dataset_filename,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "eval_steps": 1000,
        "save_steps": 1000,
        "logging_steps": 10,
        "wandb_token": os.getenv("WANDB_API_KEY"),
        "wandb_project": project_name,
    }

    response = requests.post(
        f"{train_url}/train/sft",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        json=payload,
        **auth_kwargs,
    )
    response.raise_for_status()
    training_id = response.json()["training_id"]
    print(f"Training started; training id is: {training_id}")

# ===================== Status =====================
elif args.status:
    assert "TRAINING_ID" in os.environ, "TRAINING_ID must be set in .env"
    training_id = os.getenv("TRAINING_ID")

    is_running = True
    while is_running:
        response = requests.get(
            f"{train_url}/train/status/{training_id}",
            headers={"Accept": "application/json"},
            **auth_kwargs,
        )
        response.raise_for_status()
        response_json = response.json()
        is_running = response_json["is_running"]
        print(f"Training status: {'running' if is_running else 'finished'}; last logs:")
        for log in response_json["logs"]:
            print(log)
        sleep(3)
        os.system("clear")

    registered_lora_adapter_name = response_json["registered_lora_adapter_name"]
    registered_lora_adapter_version = response_json["registered_lora_adapter_version"]
    print(
        f"Training finished; registered LoRA adapter name: {registered_lora_adapter_name}, version: {registered_lora_adapter_version}"
    )

    response = requests.get(f"{train_url}/train/logs/{training_id}", **auth_kwargs)
    response.raise_for_status()
    with open(f"training_{training_id}.log", "w") as f:
        f.write(response.text)

# ===================== Cancel =====================
elif args.cancel:
    assert "TRAINING_ID" in os.environ, "TRAINING_ID must be set in .env"
    training_id = os.getenv("TRAINING_ID")

    response = requests.delete(
        f"{train_url}/train/cancel/{training_id}",
        headers={"Accept": "application/json"},
        **auth_kwargs,
    )
    print(response.json())
    response.raise_for_status()

# ===================== Load LoRA =====================
elif args.load_lora in ["left", "right"]:
    side = args.load_lora
    adapter_name = f"opposing_views__{side}_lora_module"
    adapter_version = args.version

    assert adapter_version, "use --version to set adapter version"

    lora_data = {
        "lora_adapter": adapter_name,
        "lora_adapter_version": adapter_version,
    }

    response = requests.post(
        f"{llm_url}/lora",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json=lora_data,
        **auth_kwargs,
    )
    print(response.json())
    response.raise_for_status()

# ===================== Unload LoRA =====================
elif args.unload_lora in ["left", "right"]:
    side = args.unload_lora
    adapter_name = f"opposing_views__{side}_lora_module"

    lora_data = {"lora_adapter": adapter_name}
    response = requests.delete(
        f"{llm_url}/lora",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json=lora_data,
        **auth_kwargs,
    )
    print(response.json())
    response.raise_for_status()

# ===================== Download LoRA =====================
elif args.download_lora in ["left", "right"]:
    side = args.download_lora
    adapter_name = f"opposing_views__{side}_lora_module"
    adapter_version = args.version
    training_id = os.getenv("TRAINING_ID")

    assert training_id, "TRAINING_ID must be set in .env"
    assert adapter_version, "use --version to set adapter version"

    response = requests.get(
        f"{train_url}/train/model/{training_id}",
        **auth_kwargs,
    )
    response.raise_for_status()

    filename = f"{adapter_name}_{adapter_version}.zip"
    with open(filename, "wb") as tmp_file:
        tmp_file.write(response.content)
    print(f"Adapter saved to {filename}")

# ===================== Default =====================
else:
    print("No action specified. Use one of: --start, --status, --cancel, --load-lora, --delete-lora, --download-lora.")
