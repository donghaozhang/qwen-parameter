# Qwen2.5-VL Model Analysis

This repository contains tools for analyzing the architecture and parameters of the Qwen2.5-VL vision-language model.

## Overview

The Qwen2.5-VL model is a multimodal model that combines vision and language capabilities. This project provides utilities to:

1. Analyze the model's architecture and parameter distribution
2. Visualize the vision-language merger components
3. Run inference with the model
4. Explore the model's structure in detail

## Files

- `para.py`: Analyzes the vision-language merger architecture
- `para_analyse.py`: Provides detailed parameter analysis of different model components
- `para_visual.py`: Prints the model structure to understand its components
- `para_visual.md`: Contains output from model structure analysis
- `qwen2_5_run.py`: Example script for running inference with the model

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- prettytable

## Usage

To analyze the model architecture:

```bash
python para.py
```

To run detailed parameter analysis:

```bash
python para_analyse.py
```

To visualize the model structure:

```bash
python para_visual.py
```

To run inference with the model:

```bash
python qwen2_5_run.py
```

## License

This project is provided for educational and research purposes.

## Acknowledgements

This project uses the Qwen2.5-VL model from Alibaba Cloud. 