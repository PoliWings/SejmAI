# Installation and Run Guide

## Install Required Packages

``` bash
sudo apt install cmake nvidia-cuda-toolkit
```

## Check CUDA Version

``` bash
nvcc --version

nvidia-smi
```

## Install Python

``` bash
conda create -n sejm python=<version> -y
```

## Install PyTorch with CUDA

### For CUDA 12.4

``` bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### For CUDA 11.8
``` bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

## Local Project Installation

``` bash
python -m pip install .
```

## Install `flash-attn` (only RTX 30xx and newer)

``` bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### For CUDA 12.4 & Python 3.13 & Torch 2.6.0

``` bash
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
```

### For CUDA 11.8 & Python 3.10 & Torch 2.3.1

``` bash
pip install flash_attn-2.7.3+cu11torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Login to huggingface.co

``` bash
huggingface-cli login
```

## Start in background

``` bash
nohup ./start.sh &> out.log &
```

## Monitor and Manage

Check running Python processes:

``` bash
ps -ef | grep python
```

Kill Python processes:

``` bash
pkill -f "python"
```

Monitor RAM usage:

``` bash
watch -n 0.5 -d free -h
```

Monitor GPU usage:

``` bash
watch -n 0.5 -d nvidia-smi
```
