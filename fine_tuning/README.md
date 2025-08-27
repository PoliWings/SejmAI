# Fine Tuning
This directory contains the code and configuration files for fine-tuning the model on the Sejm dataset.

## Instructions

1. **Check CUDA version**: Ensure that you have the correct version of CUDA installed. You can check your CUDA version by running:
   ```bash
   nvcc --version
   ```
   If you don't have CUDA installed, you can download it using this command:
   ```bash
   sudo apt install nvidia-cuda-toolkit
   ```
2. **Create conda environment**: Create a new conda environment:
   ```bash
   conda create -n sejm python=3.10 -y && conda activate sejm
   ```
3. **Install torch**: Install the appropriate version of PyTorch for your CUDA version. For example, if you have CUDA 11.8, you can install PyTorch with:
   ```bash
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
   ```
4. **Install other dependencies**: Install the required dependencies:
   ```bash
   pip install bitsandbytes datasets peft sentencepiece transformers ipykernel protobuf wandb huggingface_hub[cli] dotenv trl
   ```
5. **Login to Hugging Face**: Log in to your Hugging Face account:
   ```bash
   hf auth login
   ```
6. **Run the training script**: Use the provided training script to fine-tune the model:
   ```bash
   python train_local.py --data-path <path_to_dataset> --base-model <base_model>
   ```

## File Manager
Before using the script, ensure that the `.env` file containing the `API_URL` is created.

To learn more about available options and usage, run:

```bash
./file_manager.sh --help
```

## Useful commands
- **Run script in background**: To run a script in the background, you can use the `nohup` command:
  ```bash
  nohup ./start.sh --base-model <model_name> &> out.log &
  ```
- **Check GPU & RAM usage**: To check the GPU & RAM usage, you can use the `nvidia-smi` and `free` command:
  ```bash
  watch -n 0.5 -d "nvidia-smi && free -h"
  ```
- **Check logs**: To check the logs of a running script, you can use the `tail` command:
  ```bash
  tail -f out.log
  ```
- **Kill a process**: To kill a process, you can use the `kill` command:
  ```bash
  kill -9 <pid>
  ```
  You can find the PID of a process using the `ps` command:
  ```bash
  ps -ef | grep python
  ```
  or if you want to kill all processes with a specific name:
  ```bash
  pkill -f <process_name>
  ```
