# transformer-surgeon

Transformer models library with compression options

## Overview

**transformer-surgeon** is a PyTorch-based library for building, compressing, and experimenting with transformer models.  
It supports structured pruning, low-rank decomposition, and flexible configuration for both vision and text transformer architectures.

## Features

- **Modular transformer blocks** for vision and text
- **Structured pruning** (layer or block-wise) with magnitude-based neuron removal
- **Low-rank decomposition (LRD)** for linear layers with configurable ranks
- **Generic compression manager** that works across different model architectures
- **Custom configuration classes** for compression with automatic dimension calculation
- **Support for multi-modal models** (images, videos, text)
- **LinearLRD layers** with integrated pruning mask support
- **Reversible compression** (soft application) or permanent compression (hard application)
- **QKV concatenated layer support** for attention mechanisms
- **Easy integration** with HuggingFace Transformers

## New in this Version

### 🔧 **Generic Compression Manager**
- `CompressionSchemesManager`: Universal compression manager that works with any transformer architecture
- Model-specific configurations through simple config dictionaries
- Support for multiple block types (vision, text) in the same model

### 🧮 **LinearLRD Layer**
- Drop-in replacement for `nn.Linear` with built-in low-rank decomposition
- Configurable rank: integer values or "full" for no decomposition
- Integrated pruning mask support
- Maintains same interface as standard linear layers

### ⚙️ **Advanced Configuration**
- Automatic dimension calculation based on pruning ratios
- Support for skip connection management
- Flexible key mapping system for different layer types
- Validation and error checking for configuration parameters

### 🔄 **Reversible Operations**
- **Soft application**: Compression can be reversed (restores original weights)
- **Hard application**: Permanent compression (reduces model size)
- `restore_all()` method to undo all compression operations

## Installation

```bash
git clone https://github.com/CESOIA/transformer-surgeon.git
cd transformer-surgeon
pip install -e .
```


## Quick Start

### Basic Usage

```python
from transformersurgeon.qwen2_vl_c import (
    Qwen2VLConfigCompress, 
    Qwen2VLForConditionalGenerationCompress,
    Qwen2VLCompressionSchemesManager
)

# Create compressed model configuration
config = Qwen2VLConfigCompress(
    # Pruning ratios for different layer types
    pruning_ratio_lists={
        "sa_qkv": [0.2, 0.3, 0.4],  # Attention layers
        "mlp_up": [0.1, 0.2, 0.3],  # MLP layers
    },
    # Low-rank decomposition ranks
    lrd_rank_lists={
        "sa_out": [64, 32, 16],     # Output projection ranks
        "mlp_down": ["full", 128, 64], # "full" means no decomposition
    },
    # Skip connection pruning
    pruning_ratio_skip_connections=0.1,
)

# Load model with compression
model = Qwen2VLForConditionalGenerationCompress(config)

# Create compression manager
manager = Qwen2VLCompressionSchemesManager(config, model)

# Apply compression (soft - reversible)
manager.apply_all(hard=False)

# View compression schemes
print(manager)
```

## Compression Options

### **Low-Rank Decomposition (LRD)**
- **Configurable ranks**: Integer values or "full" for no decomposition
- **SVD-based**: Uses singular value decomposition for optimal approximation
- **Layer-wise control**: Different ranks for different layers
- **Integrated with LinearLRD**: Seamless integration with custom linear layers

### **Advanced Features**
- **Flexible compression configuration**: You can customize the compression of each layer and block independently
- **QKV concatenated support**: Special handling for attention layer concatenated projections
- **Multi-block architectures**: Support for vision + text models

## Architecture Support

Currently supported models:
- **Qwen2.5-VL**: Vision-language model with compression support

**Adding new models**: The generic `CompressionSchemesManager` can be easily extended to support new architectures by providing a model-specific configuration dictionary.

## Hard and soft application

When using `manager.apply_all()` function from the class `CompressionSchemesManager`, one can choose to perform a **soft** application (default behavior) or to perform a **hard** application (`manager.apply_all(hard=True)`).
- Soft application: the performed compression is reversible and does not influence the structure of the model. Use this option when fine-tuning the model with STE or iterative approaches.
    - When performing pruning, a mask is applied, leaving the original weights unaltered.
    - When performing LRD, the original weight matrix is stored to allow retrieval.
- Hard application: the performed compression is not reversible and influences the structure of the model. Use this option for the final release of the compressed model.
    - When performing pruning, rows of the matrix are removed permanently.
    - When performing LRD, the original matrix is not stored to save memory.

When using `manager.apply_all()` a second time:
- If hard=False, nothing is done. This way, if weights have been fine-tuned, they won't be initialized.
- If hard=True, soft compression is converted in a hard one. Pruning mask are used to permanently remove weights and the stored original matrix is deleted.

To reinitialize compression, first restore the model with `manager.restore_all()`, then use `apply_all()` again.

## Roadmap

### ✅ **Completed**
- Generic compression manager architecture
- Low Rank Decomposition integration
- Qwen2.5-VL model compression support
- Reversible compression operations

### 🚧 **In progress/planned**
- Support for more transformer models (e.g., BLIP)
- Structured pruning integration
- Vanishing contribution model compression integration
- Quantization integration

## License

MIT License

---

**Maintainer:** Luciano Prono  
**Contact:** [luciano.prono@polito.it](mailto:luciano.prono@polito.it)  
**Institution:** Politecnico di Torino & King Abdullah University of Science and Technology (KAUST)
