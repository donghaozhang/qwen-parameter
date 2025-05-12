import os
import torch
import sys
from transformers import Qwen2_5_VLForConditionalGeneration

# Set Hugging Face cache directory to /workspace/qwen


def print_model_structure(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Print the structure of the model to understand its components."""
    print(f"Loading {model_name} to inspect structure...")
    
    # Load model with CPU to save memory
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # Print top-level attributes
    print("\nTop-level attributes:")
    for attr_name in dir(model):
        if not attr_name.startswith("_") and not callable(getattr(model, attr_name)):
            try:
                attr = getattr(model, attr_name)
                if isinstance(attr, torch.nn.Module):
                    param_count = sum(p.numel() for p in attr.parameters())
                    print(f"- {attr_name}: {type(attr).__name__} ({param_count} parameters)")
                else:
                    print(f"- {attr_name}: {type(attr).__name__}")
            except Exception as e:
                print(f"- {attr_name}: Error accessing ({str(e)})")
    
    # Print top-level modules
    print("\nTop-level modules (named_children):")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"- {name}: {type(module).__name__} ({param_count} parameters)")
    
    # Print all modules with "visual" or "vision" in the name
    print("\nVision-related modules:")
    for name, module in model.named_modules():
        if "visual" in name or "vision" in name:
            param_count = sum(p.numel() for p in module.parameters())
            print(f"- {name}: {type(module).__name__} ({param_count} parameters)")
    
    # Print model config
    print("\nModel config:")
    for key, value in model.config.__dict__.items():
        if not key.startswith("_"):
            print(f"- {key}: {value}")

if __name__ == "__main__":
    print_model_structure() 