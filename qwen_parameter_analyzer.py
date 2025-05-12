import os
import torch
import sys
import subprocess
import argparse

# Install required packages if they're missing
try:
    from prettytable import PrettyTable
except ImportError:
    print("Installing prettytable...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "prettytable"])
    from prettytable import PrettyTable

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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

def count_trainable_parameters(model):
    """Count the trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_structure(model):
    """Print the structure of the model to understand its components."""
    
    # Print all modules with "visual" or "vision" in the name
    print("\nVision-related modules:")
    for name, module in model.named_modules():
        if "visual" in name or "vision" in name:
            param_count = sum(p.numel() for p in module.parameters())
            print(f"- {name}: {type(module).__name__} ({format_param_count(param_count)} parameters)")
    
    # Print model config
    print("\nModel config:")
    for key, value in model.config.__dict__.items():
        if not key.startswith("_"):
            print(f"- {key}: {value}")

def analyze_model_components(model):
    """Analyze the parameters of different components in the model."""
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

def analyze_merger_architecture(model):
    """Analyze the vision-language merger architecture in detail."""
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
    
    return merger_components

def visualize_merger_flow():
    """Create a text-based visualization of the merger flow."""
    print("\nVision-Language Merger Flow:")
    print("""
    ┌───────────────┐      ┌──────────────┐
    │ Vision Model  │      │ Text Tokens  │
    └───────┬───────┘      └──────┬───────┘
            │                     │
            ▼                     ▼
    ┌───────────────┐      ┌──────────────  ┐
    │Vision Features│      │Token Embeddings│
    └───────┬───────┘      └──────┬───────  ┘
            │                     │
            ▼                     │
    ┌───────────────┐             │
    │Vision Projection│           │
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

def load_model(model_name, device="cpu"):
    """Load the model with the specified device."""
    print(f"Loading {model_name}...")
    
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map=device
    )
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Analyze Qwen2.5-VL model parameters')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help='Model name or path')
    parser.add_argument('--device', type=str, default="cpu", 
                        help='Device to load model on (cpu, cuda, auto)')
    parser.add_argument('--analysis', type=str, default="all", 
                        choices=["all", "structure", "components", "merger"],
                        help='Type of analysis to perform')
    
    args = parser.parse_args()
    
    visualize_merger_flow()
    
    # Load model
    model = load_model(args.model, args.device)

    
    if args.analysis == "all" or args.analysis == "components":
        print("\n=== MODEL COMPONENTS ANALYSIS ===")
        analyze_model_components(model)
    
    if args.analysis == "all" or args.analysis == "merger":
        print("\n=== VISION-LANGUAGE MERGER ANALYSIS ===")
        merger_components = analyze_merger_architecture(model)
    
    # Print overall architecture summary
    print("\nOverall Model Architecture:")
    for i, (name, module) in enumerate(model.named_children()):
        print(f"{i+1}. {name}: {module.__class__.__name__} ({format_param_count(count_parameters(module))} params)")
    

    if args.analysis == "all" or args.analysis == "structure":
        print("\n=== MODEL STRUCTURE ANALYSIS ===")
        print_model_structure(model)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 