# DeepSeek-OCR-2 Backend for MinerU

This document describes the integration of [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) as a new backend for MinerU.

## Overview

DeepSeek-OCR-2 is a 3B parameter Vision-Language Model specialized for document OCR and parsing. It features:

- **Visual Causal Flow** architecture for human-like visual encoding
- Direct Markdown output with layout preservation
- Dynamic resolution support: (0-6)×768×768 + 1×1024×1024
- Apache 2.0 license

## Installation

### Prerequisites

```bash
# Install DeepSeek dependencies
pip install torch>=2.6.0 transformers>=4.46.3 einops addict easydict

# Optional: Flash Attention for faster inference
pip install flash-attn>=2.7.3
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA 11.8+
- **VRAM**: ≥8GB recommended
- **Python**: 3.10+

## Usage

### Command Line

```bash
# Use DeepSeek backend
mineru -p your_document.pdf -o output_dir -b deepseek-auto-engine

# With specific model path (optional)
mineru -p your_document.pdf -o output_dir -b deepseek-auto-engine --deepseek-model-path /path/to/model
```

### Python API

```python
from mineru.cli.common import do_parse, read_fn

pdf_bytes = read_fn("your_document.pdf")

do_parse(
    output_dir="output",
    pdf_file_names=["document"],
    pdf_bytes_list=[pdf_bytes],
    p_lang_list=["en"],
    backend="deepseek-auto-engine",
)
```

## Comparison with Other Backends

| Feature | pipeline | hybrid | vlm | deepseek |
|---------|----------|--------|-----|----------|
| Parameters | Various | 1.2B | 1.2B | 3B |
| VRAM | 6GB | 8-10GB | 10GB | 8GB |
| CPU Only | ✓ | ✗ | ✗ | ✗ |
| Accuracy | 82+ | 90+ | 90+ | 90+ |
| Layout | ✓ | ✓ | ✓ | ✓ |
| Tables | ✓ | ✓ | ✓ | ✓ |
| Formulas | ✓ | ✓ | ✓ | ✓ |

## Architecture

```
mineru/backend/deepseek/
├── __init__.py
├── deepseek_analyze.py           # Core analysis logic
├── model_output_to_middle_json.py # Convert output to MinerU format
└── deepseek_middle_json_mkcontent.py # Generate markdown
```

## Notes

1. **First Run**: The model will be downloaded from HuggingFace (~6GB)
2. **Model Caching**: Model is cached as singleton to avoid repeated loading
3. **Batch Processing**: Currently processes pages sequentially

## References

- [DeepSeek-OCR-2 on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [MinerU Documentation](https://opendatalab.github.io/MinerU/)
