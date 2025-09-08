# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Apertus Swiss AI Transparency Guide - a comprehensive Python library and example collection for working with Switzerland's transparent open AI model. The project demonstrates advanced transparency analysis, multilingual capabilities, and pharmaceutical document analysis using the Apertus Swiss AI model.

## Architecture

The codebase follows a modular architecture:

- **Core Layer** (`src/apertus_core.py`): Main wrapper for model loading and basic operations
- **Analysis Layer** (`src/transparency_analyzer.py`): Advanced introspection tools for attention, hidden states, and weight analysis  
- **Application Layer** (`src/multilingual_assistant.py`, `src/pharma_analyzer.py`): Specialized assistants for different use cases
- **Interface Layer** (`examples/`, `dashboards/`): Ready-to-run examples and interactive interfaces

## Development Commands

### Installation and Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install compatible NumPy first (important for PyTorch compatibility)
pip install 'numpy>=1.24.0,<2.0.0'

# Install core dependencies
pip install torch transformers accelerate

# Install remaining dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Authenticate with Hugging Face (required for model access)
huggingface-cli login

# Basic functionality test
python examples/basic_chat.py
```

### Important Installation Notes
- **NumPy Compatibility**: Use `numpy<2.0.0` to avoid PyTorch compatibility issues
- **Virtual Environment**: Always use `.venv` to isolate dependencies
- **Model Access**: The actual model is `swiss-ai/Apertus-8B-Instruct-2509` which requires registration
- **Hugging Face Auth**: Must provide name, country, and affiliation to access the model, then login with `huggingface-cli login`

### Testing and Validation

**Prerequisites**: Must have approved access to `swiss-ai/Apertus-8B-Instruct-2509` and be logged in via `huggingface-cli login`

```bash
# Run basic functionality test
python examples/basic_chat.py

# Test multilingual capabilities  
python examples/multilingual_demo.py

# Launch interactive transparency dashboard
streamlit run dashboards/streamlit_transparency.py
```

### Package Management
```bash
# Install with console scripts
pip install -e .

# Access via console commands (after installation):
apertus-chat              # Basic chat interface
apertus-multilingual      # Multilingual demo
apertus-dashboard         # Transparency dashboard
```

## Key Components

### ApertusCore (`src/apertus_core.py`)
The main wrapper class that handles:
- Model loading with transparency options enabled
- Basic text generation with Swiss instruction format
- Conversation history management
- Multilingual capability testing
- Hardware/memory optimization

### ApertusTransparencyAnalyzer (`src/transparency_analyzer.py`)  
Advanced analysis tools providing:
- Complete model architecture analysis with parameter breakdown
- Attention pattern visualization with heatmaps
- Hidden state evolution tracking across layers
- Step-by-step token prediction analysis with probability distributions
- Weight matrix analysis and visualization
- Layer-by-layer neural network introspection

### Model Configuration
- **Actual model**: `swiss-ai/Apertus-8B-Instruct-2509`
- **Access requirement**: Must provide name, country, and affiliation on Hugging Face to access
- **Authentication**: Login with `huggingface-cli login` after getting approval
- Transparency features: `output_attentions=True`, `output_hidden_states=True`
- Optimized for: float16 precision, auto device mapping
- Memory requirements: 16GB+ RAM, CUDA GPU recommended

### Swiss Instruction Format
The model uses a specific instruction template:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
{system_message}

### Instruction:  
{prompt}

### Response:
```

## Common Tasks

### Basic Model Usage
```python
from src.apertus_core import ApertusCore
apertus = ApertusCore()
response = apertus.chat("Your message here")
```

### Transparency Analysis
```python
from src.transparency_analyzer import ApertusTransparencyAnalyzer
analyzer = ApertusTransparencyAnalyzer()

# Analyze model architecture
architecture = analyzer.analyze_model_architecture()

# Visualize attention patterns
attention_matrix, tokens = analyzer.visualize_attention_patterns("Your text")

# Track hidden state evolution
evolution = analyzer.trace_hidden_states("Your text")
```

### Multilingual Support
The model supports German, French, Italian, English, and Romansh. Language detection and switching happens automatically based on input.

## Dependencies

Core dependencies include:
- `torch>=2.0.0` - PyTorch for model operations
- `transformers>=4.30.0` - Hugging Face transformers
- `streamlit>=1.25.0` - Interactive dashboards
- `matplotlib>=3.6.0`, `seaborn>=0.12.0` - Visualization
- `numpy>=1.24.0`, `pandas>=2.0.0` - Data processing

## Performance Considerations

- **Memory**: Models require 14-26GB GPU memory depending on size
- **Optimization**: Enable gradient checkpointing and mixed precision for memory efficiency
- **Caching**: Models cache locally in `~/.cache/huggingface/`
- **GPU**: CUDA recommended, supports CPU fallback with slower performance

## File Structure Notes

- All core functionality in `src/` directory
- Examples are self-contained in `examples/` 
- Interactive dashboards in `dashboards/`
- Documentation in `docs/` with detailed guides
- Package configuration via `setup.py` with console script entry points