import os
import torch
import sys
import subprocess


# Install required packages if they're missing
try:
    from prettytable import PrettyTable
except ImportError:
    print("Installing prettytable...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
    from prettytable import PrettyTable

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import numpy as np

def count_parameters(model):
    """Count the parameters of a model."""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """Count the trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def analyze_model_components(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Analyze the parameters of different components in the Qwen2.5-VL model."""
    print(f"Loading {model_name} for parameter analysis...")
    
    # Load model with CPU to save memory during analysis
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # Create a table for displaying results
    table = PrettyTable()
    table.field_names = ["Component", "Parameters", "% of Total"]
    
    # Get total parameters
    total_params = count_parameters(model)
    
    # Analyze vision encoder - in Qwen2.5-VL, it's model.visual not vision_model
    vision_params = 0
    if hasattr(model, 'visual'):
        vision_params = count_parameters(model.visual)
    elif hasattr(model, 'vision_tower'):
        vision_params = count_parameters(model.vision_tower)
    vision_percent = (vision_params / total_params) * 100
    
    # Analyze language model
    language_params = 0
    if hasattr(model, 'language_model'):
        language_params = count_parameters(model.language_model)
    elif hasattr(model, 'model'):
        language_params = count_parameters(model.model)
    language_percent = (language_params / total_params) * 100
    
    # Analyze vision-language connector (merger)
    connector_params = 0
    if hasattr(model, 'vision_projection'):
        connector_params += count_parameters(model.vision_projection)
    if hasattr(model, 'multi_modal_projector'):
        connector_params += count_parameters(model.multi_modal_projector)
    if hasattr(model, 'mm_projector'):
        connector_params += count_parameters(model.mm_projector)
    connector_percent = (connector_params / total_params) * 100
    
    # Other parameters (embeddings, etc.)
    other_params = total_params - vision_params - language_params - connector_params
    other_percent = (other_params / total_params) * 100
    
    # Add rows to the table
    table.add_row(["Vision Encoder", format_param_count(vision_params), f"{vision_percent:.2f}%"])
    table.add_row(["Language Model", format_param_count(language_params), f"{language_percent:.2f}%"])
    table.add_row(["Vision-Language Connector", format_param_count(connector_params), f"{connector_percent:.2f}%"])
    table.add_row(["Other Components", format_param_count(other_params), f"{other_percent:.2f}%"])
    table.add_row(["Total", format_param_count(total_params), "100.00%"])
    
    print(table)
    
    # Detailed analysis of vision components
    print("\nDetailed Vision Component Analysis:")
    analyze_vision_component(model)
    
    # Detailed analysis of language model
    print("\nDetailed Language Model Analysis:")
    analyze_language_model(model)
    
    return model

def analyze_vision_component(model):
    """Analyze the components of the vision encoder."""
    vision_table = PrettyTable()
    vision_table.field_names = ["Component", "Parameters"]
    
    # Find the vision component (different models may name it differently)
    vision_component = None
    if hasattr(model, 'visual'):
        vision_component = model.visual
    elif hasattr(model, 'vision_tower'):
        vision_component = model.vision_tower
    
    if vision_component is None:
        print("Could not find vision component in model")
        return
    
    # Patch embedding
    if hasattr(vision_component, 'patch_embed'):
        embed_params = count_parameters(vision_component.patch_embed)
        vision_table.add_row(["Patch Embedding", format_param_count(embed_params)])
    
    # Transformer blocks
    if hasattr(vision_component, 'blocks'):
        blocks_params = count_parameters(vision_component.blocks)
        num_blocks = len(vision_component.blocks)
        params_per_block = blocks_params / num_blocks
        vision_table.add_row(["Transformer Blocks", f"{format_param_count(blocks_params)} ({num_blocks} blocks, {format_param_count(params_per_block)} per block)"])
    
    # Normalization layers
    if hasattr(vision_component, 'norm'):
        norm_params = count_parameters(vision_component.norm)
        vision_table.add_row(["Normalization", format_param_count(norm_params)])
    
    print(vision_table)
    
    # Print structure of the vision component
    print("\nVision Component Structure:")
    for name, child in vision_component.named_children():
        print(f"- {name}: {child.__class__.__name__} ({format_param_count(count_parameters(child))} params)")

def analyze_language_model(model):
    """Analyze the components of the language model."""
    language_table = PrettyTable()
    language_table.field_names = ["Component", "Parameters"]
    
    # Find the language model component
    language_component = None
    if hasattr(model, 'language_model'):
        language_component = model.language_model
    elif hasattr(model, 'model'):
        language_component = model.model
    
    if language_component is None:
        print("Could not find language model component")
        return
    
    # Embeddings
    if hasattr(language_component, 'embed_tokens'):
        embed_params = count_parameters(language_component.embed_tokens)
        language_table.add_row(["Token Embeddings", format_param_count(embed_params)])
    
    # Layers
    if hasattr(language_component, 'layers'):
        layers_params = count_parameters(language_component.layers)
        num_layers = len(language_component.layers)
        params_per_layer = layers_params / num_layers
        language_table.add_row(["Transformer Layers", f"{format_param_count(layers_params)} ({num_layers} layers, {format_param_count(params_per_layer)} per layer)"])
    
    # LM Head
    if hasattr(model, 'lm_head'):
        lm_head_params = count_parameters(model.lm_head)
        language_table.add_row(["LM Head", format_param_count(lm_head_params)])
    
    # Norm
    if hasattr(language_component, 'norm'):
        norm_params = count_parameters(language_component.norm)
        language_table.add_row(["Final Norm", format_param_count(norm_params)])
    
    print(language_table)

if __name__ == "__main__":
    print("Analyzing Qwen2.5-VL model parameters...")
    model = analyze_model_components()
    
    print("\nModel architecture overview:")
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")
        
    print("\nAnalysis complete!") 