# Installation Guide

## System Requirements

- Python >= 3.10
- CUDA-capable GPU (recommended, but CPU mode is supported)
- CUDA toolkit (for GPU support)
- At least 50GB free disk space (for model storage)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/your-org/cmw-vllm.git
cd cmw-vllm
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

This will install:
- vLLM
- HuggingFace Hub
- Click (for CLI)
- Other dependencies

### 4. Verify Installation

```bash
cmw-vllm setup
```

This command checks:
- vLLM installation
- GPU availability
- Required dependencies

## Troubleshooting

### vLLM Installation Fails

If vLLM installation fails, you may need to install CUDA toolkit first:

```bash
# Check CUDA version
nvidia-smi

# Install vLLM with specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### GPU Not Detected

If GPU is not detected:
- Ensure NVIDIA drivers are installed
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU: `nvidia-smi`

### Disk Space Issues

Models can be large (30GB+). Ensure sufficient disk space:
- Check space: `df -h`
- Use `--local-dir` to download to a different location
- Set `HF_HOME` environment variable to use different cache location
