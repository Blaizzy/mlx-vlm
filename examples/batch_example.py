import mlx_vlm
from mlx_vlm.utils import load_image, process_image
import sys
import traceback

def test_model(model_name, images, prompt):
    """Test batch generation with a specific model."""
    try:
        print(f"\nTesting {model_name}...")
        print("=" * 50)
        
        # Load model and processor
        print(f"Loading model {model_name}...")
        model, processor = mlx_vlm.load(model_name)
        
        # Create batch of prompts
        prompts = [prompt] * len(images)
        
        # Validate images exist
        for img_path in images:
            try:
                # Test loading each image
                _ = load_image(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                return
        
        print("Generating responses...")
        # Generate responses
        responses = mlx_vlm.batch_generate(
            model,
            processor,
            prompts=prompts,
            images=images,
            max_tokens=500,
            temperature=0.7,
            verbose=True
        )
        
        print("\nGenerated Responses:")
        for i, response in enumerate(responses):
            print(f"\nImage {i+1}:")
            print(response)
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError testing {model_name}:")
        print(traceback.format_exc())
        print("=" * 50)

def main():
    # Test images
    images = [
        "examples/images/cats.jpg",
        "examples/images/desktop_setup.png"
    ]
    
    # Test prompt
    prompt = "Describe what you see in this image in detail."
    
    # Test with different models
    models = [
        "mlx-community/llava-interleave-qwen-0.5b-bf16",  # LLaVA model
        "mlx-community/Qwen2-VL-2B-Instruct-4bit",        # Qwen2-VL model
    ]
    
    success = False
    for model in models:
        try:
            test_model(model, images, prompt)
            success = True
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\nFailed to test {model}: {str(e)}")
            continue
    
    if not success:
        print("\nAll model tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 