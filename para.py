import os
import torch
import sys
import subprocess

# Set Hugging Face cache directory to /workspace/qwen

# Install required packages if they're missing
try:
    from prettytable import PrettyTable
except ImportError:
    print("Installing prettytable...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
    from prettytable import PrettyTable

from transformers import Qwen2_5_VLForConditionalGeneration
import numpy as np

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

def analyze_merger_architecture(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Analyze the vision-language merger architecture in detail."""
    print(f"Loading {model_name} for merger analysis...")
    
    # Load model with CPU to save memory during analysis
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # Create a table for displaying results
    table = PrettyTable()
    table.field_names = ["Component", "Type", "Parameters", "Shape"]
    
    # Identify and analyze merger components
    merger_components = {}
    
    # Common merger component names
    merger_names = [
        'vision_projection', 'multi_modal_projector', 'vision_embed_tokens',
        'perceiver', 'vision_tower', 'mm_projector', 'vision_resampler',
        'visual.merger', 'visual_projection'
    ]
    
    # Find all potential merger components
    for name, module in model.named_modules():
        if any(merger_name in name for merger_name in merger_names):
            merger_components[name] = module
    
    # Analyze each merger component
    for name, module in merger_components.items():
        # Skip non-parameter modules
        if count_parameters(module) == 0:
            continue
            
        # Get parameter shapes
        param_shapes = []
        for param_name, param in module.named_parameters():
            param_shapes.append(f"{param_name}: {list(param.shape)}")
        
        # Add to table
        table.add_row([
            name,
            module.__class__.__name__,
            format_param_count(count_parameters(module)),
            "\n".join(param_shapes[:3]) + ("\n..." if len(param_shapes) > 3 else "")
        ])
    
    print(table)
    
    # Analyze the flow of information
    print("\nVision-Language Information Flow Analysis:")
    
    # Trace the vision embedding dimensions
    vision_embed_dim = None
    language_embed_dim = None
    
    # Find vision embedding dimension
    if hasattr(model, 'visual'):
        if hasattr(model.visual, 'config'):
            vision_embed_dim = getattr(model.visual.config, 'hidden_size', None)
        elif hasattr(model.visual, 'embed_dim'):
            vision_embed_dim = model.visual.embed_dim
    elif hasattr(model, 'vision_tower'):
        if hasattr(model.vision_tower, 'config'):
            vision_embed_dim = getattr(model.vision_tower.config, 'hidden_size', None)
    
    # Find language embedding dimension
    if hasattr(model, 'language_model'):
        if hasattr(model.language_model, 'config'):
            language_embed_dim = getattr(model.language_model.config, 'hidden_size', None)
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'config'):
            language_embed_dim = getattr(model.model.config, 'hidden_size', None)
    
    print(f"Vision Embedding Dimension: {vision_embed_dim}")
    print(f"Language Embedding Dimension: {language_embed_dim}")
    
    # Analyze projection dimensions
    if hasattr(model, 'vision_projection'):
        in_features = getattr(model.vision_projection, 'in_features', 'Unknown')
        out_features = getattr(model.vision_projection, 'out_features', 'Unknown')
        print(f"\nVision Projection: {in_features} → {out_features}")
        print(f"This transforms vision features to match language model dimensions")
    elif hasattr(model, 'mm_projector'):
        if hasattr(model.mm_projector, 'weight'):
            in_features = model.mm_projector.weight.shape[1]
            out_features = model.mm_projector.weight.shape[0]
            print(f"\nVision Projection (mm_projector): {in_features} → {out_features}")
            print(f"This transforms vision features to match language model dimensions")
    
    return model, merger_components

def visualize_merger_flow():
    """Create a text-based visualization of the merger flow."""
    print("\nVision-Language Merger Flow:")
    print("""
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
    """)

if __name__ == "__main__":
    model, merger_components = analyze_merger_architecture()
    visualize_merger_flow()
    
    # Additional analysis of model architecture
    print("\nOverall Model Architecture:")
    for i, (name, module) in enumerate(model.named_children()):
        print(f"{i+1}. {name}: {module.__class__.__name__} ({format_param_count(count_parameters(module))} params)")
        
    print("\nAnalysis complete!") 