import torch
import numpy as np
from prettytable import PrettyTable

def format_param_count(count):
    """Format parameter count in a readable way."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    else:
        return f"{count}"

def count_parameters(module):
    """Count the parameters of a module."""
    return sum(p.numel() for p in module.parameters())

def count_trainable_parameters(model):
    """Count the trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_parameter_table(components_dict):
    """Create a pretty table for displaying parameter counts."""
    table = PrettyTable()
    table.field_names = ["Component", "Parameters", "% of Total"]
    
    total_params = sum(count_parameters(module) for module in components_dict.values())
    
    for name, module in components_dict.items():
        params = count_parameters(module)
        percent = (params / total_params) * 100
        table.add_row([name, format_param_count(params), f"{percent:.2f}%"])
    
    table.add_row(["Total", format_param_count(total_params), "100.00%"])
    
    return table

def process_vision_info(messages):
    """Process vision information from messages for Qwen2.5-VL model.
    
    This is a placeholder function. In a real implementation, it would extract
    image and video data from the messages.
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    # In a real implementation, this would load the image
                    image_inputs.append(content["image"])
    
    return image_inputs, video_inputs

def visualize_merger_flow():
    """Create a text-based visualization of the merger flow."""
    return """
    ┌───────────────┐      ┌──────────────┐
    │ Vision Model  │      │ Text Tokens  │
    └───────┬───────┘      └──────┬───────┘
            │                     │
            ▼                     ▼
    ┌───────────────┐      ┌──────────────┐
    │Vision Features│      │Token Embeddings│
    └───────┬───────┘      └──────┬───────┘
            │                     │
            ▼                     │
    ┌───────────────┐             │
    │Vision Projection│            │
    └───────┬───────┘             │
            │                     │
            ▼                     ▼
    ┌─────────────────────────────────┐
    │     Merged Representation       │
    └─────────────────┬───────────────┘
                      │
                      ▼
    ┌─────────────────────────────────┐
    │      Language Model Layers      │
    └─────────────────┬───────────────┘
                      │
                      ▼
    ┌─────────────────────────────────┐
    │          Output Tokens          │
    └─────────────────────────────────┘
    """ 