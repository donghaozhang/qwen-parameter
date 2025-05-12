import os
import requests
from PIL import Image
from io import BytesIO

def process_vision_info(messages):
    """Process vision information from messages for Qwen2.5-VL model.
    
    Args:
        messages: List of message dictionaries containing text and image content
        
    Returns:
        tuple: (image_inputs, video_inputs) where:
            - image_inputs: List of loaded images
            - video_inputs: List of loaded videos (empty in this implementation)
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    # Handle image content
                    image_source = content["image"]
                    
                    # Handle image from URL
                    if isinstance(image_source, str) and (image_source.startswith("http://") or image_source.startswith("https://")):
                        try:
                            response = requests.get(image_source)
                            image = Image.open(BytesIO(response.content))
                            image_inputs.append(image)
                        except Exception as e:
                            print(f"Error loading image from URL: {e}")
                    
                    # Handle local file path
                    elif isinstance(image_source, str) and os.path.isfile(image_source):
                        try:
                            image = Image.open(image_source)
                            image_inputs.append(image)
                        except Exception as e:
                            print(f"Error loading image from file: {e}")
                    
                    # Handle PIL Image object
                    elif hasattr(image_source, "convert"):  # Check if it's a PIL Image
                        image_inputs.append(image_source)
                    
                    else:
                        print(f"Unsupported image source: {type(image_source)}")
                
                elif content["type"] == "video":
                    # In this implementation, we don't handle video processing
                    # This would require additional video processing libraries
                    print("Video processing is not implemented in this utility")
    
    return image_inputs, video_inputs 