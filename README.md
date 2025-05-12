# Qwen2.5-VL Model Analysis

This repository contains tools for analyzing the architecture and parameters of the Qwen2.5-VL vision-language model.

## Overview

The Qwen2.5-VL model is a multimodal model that combines vision and language capabilities. This project provides utilities to:

1. Analyze the model's architecture and parameter distribution
2. Visualize the vision-language merger components
3. Run inference with the model
4. Explore the model's structure in detail

## Repository Structure

```
qwen/
├── .cursor/rules/              # Cursor IDE navigation rules
├── archive/                    # Archived individual analysis scripts
│   ├── para_analyse.py         # Parameter analysis script
│   ├── para.py                 # Merger architecture analysis script
│   ├── para_visual.py          # Structure visualization script
│   ├── qwen_vl_utils.py        # Qwen VL utilities
│   └── utils.py                # General utilities
├── results/                    # Analysis results and documentation
│   ├── para_abalyse.md         # Parameter analysis results
│   ├── para.md                 # Merger architecture analysis results
│   ├── para_visual.md          # Structure visualization results
│   └── qwen_parameter_analysis.md  # Comprehensive analysis document
├── qwen_parameter_analyzer.py  # Combined analysis tool
├── qwen2_5_run.py              # Model inference script
└── requirements.txt            # Project dependencies
```

## Main Files

- `qwen_parameter_analyzer.py`: Comprehensive analyzer for model parameters, architecture, and merger components
- `qwen2_5_run.py`: Example script for running inference with the model
- `results/qwen_parameter_analysis.md`: Detailed analysis of the model architecture and parameters

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- prettytable

## Usage

### Comprehensive Analysis (Recommended)

Use the combined script for all analyses:

```bash
# Run all analyses
python qwen_parameter_analyzer.py

# Run specific analysis type
python qwen_parameter_analyzer.py --analysis structure
python qwen_parameter_analyzer.py --analysis components
python qwen_parameter_analyzer.py --analysis merger

# Specify model or device
python qwen_parameter_analyzer.py --model "Qwen/Qwen2.5-VL-7B-Instruct" --device cuda
```

### Model Inference

To run inference with the model:

```bash
python qwen2_5_run.py
```

## Analysis Results

The analysis results are stored in the `results/` directory:

- `qwen_parameter_analysis.md`: Comprehensive analysis of the model architecture
- `para_visual.md`: Detailed output of the model structure
- `para.md`: Analysis of the vision-language merger architecture
- `para_abalyse.md`: Detailed parameter distribution analysis

## License

This project is provided for educational and research purposes.

## Acknowledgements

This project uses the Qwen2.5-VL model from Alibaba Cloud. 