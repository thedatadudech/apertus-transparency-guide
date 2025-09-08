# Apertus Transparency Guide - Installation Instructions

## 🚀 Quick Start Installation

### Prerequisites

Before installing, ensure you have:

- **Python 3.8+** (3.9 or 3.10 recommended)
- **Git** for cloning the repository
- **CUDA-capable GPU** (recommended but not required)
- **16GB+ RAM** for basic usage, 32GB+ for full transparency analysis

### Hardware Requirements

| Use Case | GPU | RAM | Storage | Expected Performance |
|----------|-----|-----|---------|---------------------|
| Basic Chat | RTX 3060 12GB | 16GB | 20GB | Good |
| Transparency Analysis | RTX 4090 24GB | 32GB | 50GB | Excellent |
| Full Development | A100 40GB | 64GB | 100GB | Optimal |
| CPU Only | N/A | 32GB+ | 20GB | Slow but functional |

---

## 📦 Installation Methods

### Method 1: Clone and Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/apertus-transparency-guide.git
cd apertus-transparency-guide

# Create virtual environment
python -m venv apertus_env
source apertus_env/bin/activate  # On Windows: apertus_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Test installation
python examples/basic_chat.py
```

### Method 2: Direct pip install

```bash
# Install directly from repository
pip install git+https://github.com/yourusername/apertus-transparency-guide.git

# Or install from PyPI (when published)
pip install apertus-transparency-guide
```

### Method 3: Docker Installation

```bash
# Build Docker image
docker build -t apertus-transparency .

# Run interactive container
docker run -it --gpus all -p 8501:8501 apertus-transparency

# Run dashboard
docker run -p 8501:8501 apertus-transparency streamlit run dashboards/streamlit_transparency.py
```

---

## 🔧 Platform-Specific Instructions

### Windows Installation

```powershell
# Install Python 3.9+ from python.org
# Install Git from git-scm.com

# Clone repository
git clone https://github.com/yourusername/apertus-transparency-guide.git
cd apertus-transparency-guide

# Create virtual environment
python -m venv apertus_env
apertus_env\Scripts\activate

# Install PyTorch with CUDA (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Test installation
python examples\basic_chat.py
```

### macOS Installation

```bash
# Install Python via Homebrew
brew install python@3.10

# Install dependencies
export PATH="/opt/homebrew/bin:$PATH"  # For Apple Silicon Macs

# Clone and install
git clone https://github.com/yourusername/apertus-transparency-guide.git
cd apertus-transparency-guide

# Create virtual environment
python3 -m venv apertus_env
source apertus_env/bin/activate

# Install dependencies (CPU version for Apple Silicon)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Test installation
python examples/basic_chat.py
```

### Linux (Ubuntu/Debian) Installation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and Git
sudo apt install python3.10 python3.10-venv python3-pip git -y

# Clone repository
git clone https://github.com/yourusername/apertus-transparency-guide.git
cd apertus-transparency-guide

# Create virtual environment
python3 -m venv apertus_env
source apertus_env/bin/activate

# Install PyTorch with CUDA (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Test installation
python examples/basic_chat.py
```

---

## 🎯 GPU Setup and Optimization

### NVIDIA GPU Setup

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Install CUDA-optimized PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For older GPUs, use CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Verify GPU setup
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```

### AMD GPU Setup (ROCm)

```bash
# Install ROCm PyTorch (Linux only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Verify ROCm setup
python -c "
import torch
print(f'ROCm available: {torch.cuda.is_available()}')  # ROCm uses CUDA API
"
```

### Apple Silicon (M1/M2) Optimization

```bash
# Install MPS-optimized PyTorch
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
```

---

## 🔐 Configuration and Environment Setup

### Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env  # or your preferred editor
```

Key configuration options:

```bash
# Model configuration
DEFAULT_MODEL_NAME=swiss-ai/apertus-7b-instruct
MODEL_CACHE_DIR=./model_cache
DEVICE_MAP=auto
TORCH_DTYPE=float16

# Performance tuning
MAX_MEMORY_GB=16
ENABLE_MEMORY_MAPPING=true
GPU_MEMORY_FRACTION=0.9

# Swiss localization
DEFAULT_LANGUAGE=de
SUPPORTED_LANGUAGES=de,fr,it,en,rm
```

### Hugging Face Token Setup

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face (optional, for private models)
huggingface-cli login

# Or set token in environment
export HUGGINGFACE_TOKEN=your_token_here
```

---

## 🧪 Verification and Testing

### Quick Test Suite

```bash
# Test basic functionality
python -c "
from src.apertus_core import ApertusCore
print('✅ Core module imported successfully')

try:
    apertus = ApertusCore()
    response = apertus.chat('Hello, test!')
    print('✅ Basic chat functionality working')
except Exception as e:
    print(f'❌ Error: {e}')
"

# Test transparency features
python -c "
from src.transparency_analyzer import ApertusTransparencyAnalyzer
analyzer = ApertusTransparencyAnalyzer()
architecture = analyzer.analyze_model_architecture()
print('✅ Transparency analysis working')
"

# Test multilingual features
python examples/multilingual_demo.py

# Test pharmaceutical analysis
python examples/pharma_analysis.py
```

### Dashboard Testing

```bash
# Test Streamlit dashboard
streamlit run dashboards/streamlit_transparency.py

# Should open browser at http://localhost:8501
# If not, manually navigate to the URL shown in terminal
```

### Performance Benchmarking

```bash
# Run performance test
python -c "
import time
import torch
from src.apertus_core import ApertusCore

print('Running performance benchmark...')
apertus = ApertusCore()

# Warmup
apertus.chat('Warmup message')

# Benchmark
start_time = time.time()
for i in range(5):
    response = apertus.chat(f'Test message {i}')
end_time = time.time()

avg_time = (end_time - start_time) / 5
print(f'Average response time: {avg_time:.2f} seconds')

if torch.cuda.is_available():
    memory_used = torch.cuda.memory_allocated() / 1024**3
    print(f'GPU memory used: {memory_used:.2f} GB')
"
```

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Issue: "CUDA out of memory"

```bash
# Solution 1: Use smaller model or quantization
export TORCH_DTYPE=float16
export USE_QUANTIZATION=true

# Solution 2: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Solution 3: Reduce batch size or context length
export MAX_CONTEXT_LENGTH=2048
```

#### Issue: "Model not found"

```bash
# Check Hugging Face connectivity
pip install huggingface_hub
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"

# Clear model cache and redownload
rm -rf ~/.cache/huggingface/transformers/
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('swiss-ai/apertus-7b-instruct')"
```

#### Issue: "Import errors"

```bash
# Reinstall dependencies
pip uninstall apertus-transparency-guide -y
pip install -r requirements.txt
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### Issue: "Slow performance"

```bash
# Enable optimizations
export TORCH_COMPILE=true
export USE_FLASH_ATTENTION=true

# For CPU-only systems
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

#### Issue: "Streamlit dashboard not working"

```bash
# Update Streamlit
pip install --upgrade streamlit

# Check port availability
lsof -i :8501  # Kill process if needed

# Run with different port
streamlit run dashboards/streamlit_transparency.py --server.port 8502
```

---

## 📈 Performance Optimization Tips

### Memory Optimization

```python
# In your code, use these optimizations:

# 1. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 2. Use mixed precision
import torch
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(**inputs)

# 3. Clear cache regularly
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

### Speed Optimization

```python
# 1. Compile model (PyTorch 2.0+)
import torch
model = torch.compile(model)

# 2. Use optimized attention
# Set in environment: PYTORCH_ENABLE_MPS_FALLBACK=1

# 3. Batch processing
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

---

## 🔄 Updating and Maintenance

### Updating the Installation

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall package
pip install -e . --force-reinstall

# Clear model cache (if needed)
rm -rf ~/.cache/huggingface/transformers/models--swiss-ai--apertus*
```

### Maintenance Tasks

```bash
# Clean up cache files
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Clearing caches...')
torch.cuda.empty_cache() if torch.cuda.is_available() else None
"

# Update model cache
python -c "
from transformers import AutoModelForCausalLM
print('Updating model cache...')
AutoModelForCausalLM.from_pretrained('swiss-ai/apertus-7b-instruct', force_download=True)
"

# Run health check
python examples/basic_chat.py --health-check
```

---

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look in `./logs/apertus.log` for detailed error messages
2. **GitHub Issues**: [Create an issue](https://github.com/yourusername/apertus-transparency-guide/issues)
3. **Discord Community**: Join the [Swiss AI Discord](discord-link)
4. **Documentation**: Visit the [full documentation](docs-link)

### Diagnostic Information

When reporting issues, include this diagnostic information:

```bash
python -c "
import sys, torch, transformers, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"
```

---

**Installation complete! 🎉**

You're now ready to explore Apertus's transparency features. Start with:

```bash
python examples/basic_chat.py
```

or launch the interactive dashboard:

```bash
streamlit run dashboards/streamlit_transparency.py
```
