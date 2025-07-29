# transformer-surgeon

Transformer models library with compression options

## Overview

**transformer-surgeon** is a PyTorch-based library for building, compressing, and experimenting with transformer models.  
It supports structured pruning, low-rank decomposition, and flexible configuration for both vision and text transformer architectures.

## Features

- Modular transformer blocks for vision and text
- Structured pruning (layer or block-wise)
- Low-rank decomposition for linear layers
- Custom configuration classes for compression
- Support for multi-modal models (images, videos, text)
- Easy integration with HuggingFace Transformers

## Installation

```bash
git clone https://github.com/yourusername/transformer-surgeon.git
cd transformer-surgeon
pip install -e .
```

## Usage

```python
from transformer_surgeon.qwen2_vl_c import Qwen2VLModel, Qwen2VLConfigCompress

config = Qwen2VLConfigCompress(
    pruning_ratio_lists={...},
    lrd_rank_lists={...},
    pruning_ratio_skip_connections=0.2,
)
model = Qwen2VLModel(config)
```

## Compression Options

- **Pruning:** Remove neurons, attention heads, or blocks based on configurable ratios.
- **Low-Rank Decomposition:** Replace linear layers with low-rank approximations for efficiency.
- **Skip Connection Management:** Ensure compatibility when pruning layers connected by skip connections.

## Documentation

- See [docs/](docs/) for API details and examples.
- Example configs and scripts are provided in the `examples/` folder.

## License

MIT License

---

**Maintainer:** Luciano Prono
**Contact:** [luciano.prono@polito.it](mailto:luciano.prono@polito.it)

## What to do next

- Tools to convert full-size model to compressed model
- Debug and test compressed models