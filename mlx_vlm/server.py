# server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Tuple
import uvicorn
import gc
import mlx.core as mx
import json
import asyncio
import traceback 
import codecs
from .utils import generate, stream_generate
from .prompt_utils import apply_chat_template
from .generate import (
    get_model_and_processors, 
    DEFAULT_MODEL_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SEED,
    DEFAULT_PROMPT
)

app = FastAPI(title="MLX_VLM Inference API",
              description="API for using Vision Language Models (VLM) with MLX.",
              version="0.1.0")

MAX_IMAGES = 10 # Maximum number of images to process at once

model_cache = {}

def load_model_resources(model_path: str, adapter_path: Optional[str]):
    """
    Loads model, processor, and config based on paths.
    Handles potential loading errors.
    """
    try:
        print(f"Loading model from: {model_path}")
        if adapter_path:
            print(f"Loading adapter from: {adapter_path}")
        # Use the function from generate.py which handles path resolution and loading
        model, processor, config = get_model_and_processors(model_path, adapter_path)
        print("Model and processor loaded successfully.")
        return model, processor, config
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


def get_cached_model(model_path: str, adapter_path: Optional[str]):
    """
    Factory function to get or load the appropriate model resources from cache or by loading.
    """
    global model_cache

    # Construct a unique key for the cache
    cache_key = (model_path, adapter_path)

    # Return from cache if already loaded and matches the requested paths
    if model_cache.get("cache_key") == cache_key:
        print(f"Using cached model: {model_path}, Adapter: {adapter_path}")
        return model_cache["model"], model_cache["processor"], model_cache["config"]

    # If cache exists but doesn't match, clear it
    if model_cache:
        print("New model request, clearing existing cache...")
        unload_model_sync() # Use a synchronous version for internal call

    # Load the model resources
    model, processor, config = load_model_resources(model_path, adapter_path)

    # Update the cache
    model_cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config
    }

    return model, processor, config

# Synchronous unload function for internal use
def unload_model_sync():
    global model_cache
    if not model_cache:
        return False # Indicate nothing was unloaded

    print(f"Unloading model: {model_cache.get('model_path')}, Adapter: {model_cache.get('adapter_path')}")
    # Clear references
    model_cache = {}
    # Force garbage collection
    gc.collect()
    mx.clear_cache() # Clear mlx cache as well
    print("Model unloaded and cache cleared.")
    return True

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'system', 'user', 'assistant').")
    content: str = Field(..., description="Content of the message.")

class VLMRequest(BaseModel):
    model: str = Field(DEFAULT_MODEL_PATH, description="The path to the local model directory or Hugging Face repo.")
    adapter_path: Optional[str] = Field(None, description="The path to the adapter weights.")
    image: List[str] = Field(default_factory=list, description="List of URLs or local paths of images to process.")
    prompt: str = Field(DEFAULT_PROMPT, description="Message to be processed by the model.")
    system: Optional[str] = Field(None, description="Optional system message for the model (used in chat structure).")
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, description="Maximum number of tokens to generate.")
    temperature: float = Field(DEFAULT_TEMPERATURE, description="Temperature for sampling.")
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    resize_shape: Optional[Tuple[int, int]] = Field(None, description="Resize shape for the image (height, width). Provide two integers.")

class GenerationRequest(VLMRequest):
    """
    Inherits from VLMRequest and adds additional fields for the generation request.
    """
    stream: bool = Field(False, description="Whether to stream the response chunk by chunk.")

class ChatRequest(GenerationRequest):
    """
    Inherits from GenerationRequest and adds fields specific to chat interactions.
    """
    prompt: List[ChatMessage] = Field(default_factory=list, description="List of chat messages for the conversation.")    


class GenerationResponse(BaseModel):
    text: str
    model: str
    usage: Dict[str, Any] # Placeholder for verbose usage stats, can be expanded as needed

class StreamChunk(BaseModel):
    chunk: str
    model: str # placeholder (TBC).


@app.post("/generate", response_model=None) # Response model handled dynamically based on stream flag
async def generate_endpoint(request: GenerationRequest):
    """
    Generate text based on a prompt and optional images.
    Can operate in streaming or non-streaming mode.

    NOTE: ideally, generate.py should be refactored to allow reusing the same code here
    but for now, we just copy the logic from generate.py to avoid merging issues.
    """
    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(request.model, request.adapter_path)

        kwargs = {}

        if request.resize_shape is not None:
            if len(request.resize_shape) not in [1, 2]:
                raise HTTPException(status_code=400, detail="resize_shape must contain exactly two integers (height, width)")
            kwargs["resize_shape"] = (
                (request.resize_shape[0],) * 2
                if len(request.resize_shape) == 1
                else tuple(request.resize_shape)
            )

        # Prepare the prompt using the chat template logic
        # Adapt apply_chat_template for single prompt + optional system message
        chat_messages = []
        if request.system:
            system_prompt = codecs.decode(request.system, "unicode_escape")
            chat_messages.append({"role": "system", "content": system_prompt})
        prompt = codecs.decode(request.prompt, "unicode_escape")
        chat_messages.append({"role": "user", "content": prompt})

        # For now, assume it works or adapt it as needed.
        formatted_prompt = apply_chat_template(
            processor, config, chat_messages, num_images=len(request.image)
        )

        if request.stream:
            # Streaming response
            async def stream_generator():
                token_iterator = None
                try:
                    # Use stream_generate from utils
                    token_iterator = stream_generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=request.image,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                        **kwargs
                    )
                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, 'text'):
                           print("Warning: Received unexpected chunk format:", chunk)
                           continue 

                        # Yield chunks in Server-Sent Events (SSE) format
                        chunk_data = StreamChunk(chunk=chunk.text, model=request.model)
                        yield f"data: {chunk_data.model_dump_json()}\n\n"
                        await asyncio.sleep(0.01) # Small sleep to prevent blocking event loop entirely

                    # Signal stream end (optional, depends on client handling)
                    # yield f"data: {json.dumps({'end': True})}\n\n"

                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    # Try to send an error message over the stream if possible
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"
                finally:
                    # Clean up resources after generation completes or fails
                    mx.clear_cache()
                    gc.collect()
                    print("Stream finished, cleared cache.")


            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            # Non-streaming response
            try:
                # Use generate from utils
                output = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=request.image,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    verbose=False, # Keep API output clean
                    **kwargs
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                # Create response
                result = GenerationResponse(
                    text=output,
                    model=request.model,
                    usage={} # TODO
                )
                return result

            except Exception as e:
                 print(f"Error during generation: {e}")
                 traceback.print_exc()
                 # Clean up resources even on error
                 mx.clear_cache()
                 gc.collect()
                 raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error in /generate endpoint: {e}")
        traceback.print_exc()
        # Clean up resources on unexpected error
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/chat", response_model=None) # Response model handled dynamically based on stream flag
async def generate_endpoint(request: GenerationRequest):
    """
    Generate text based on a prompt and optional images.
    Prompt must be a list of chat messages, including system, user, and assistant messages.
    System message will be ignored of not already in the prompt.
    Can operate in streaming or non-streaming mode.

    NOTE: ideally, generate.py should be refactored to allow reusing the same code here
    but for now, we just copy the logic from generate.py to avoid merging issues.
    """

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(request.model, request.adapter_path)

        kwargs = {}

        if request.resize_shape is not None:
            if len(request.resize_shape) not in [1, 2]:
                raise HTTPException(status_code=400, detail="resize_shape must contain exactly two integers (height, width)")
            kwargs["resize_shape"] = (
                (request.resize_shape[0],) * 2
                if len(request.resize_shape) == 1
                else tuple(request.resize_shape)
            )

        # Prepare the prompt using the chat template logic
        chat_messages = request.prompt

        # For now, assume it works or adapt it as needed.
        formatted_prompt = apply_chat_template(
            processor, config, chat_messages, num_images=len(request.image)
        )

        if request.stream:
            # Streaming response
            async def stream_generator():
                token_iterator = None
                try:
                    # Use stream_generate from utils
                    token_iterator = stream_generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=request.image,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                        **kwargs
                    )
                    
                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, 'text'):
                           print("Warning: Received unexpected chunk format:", chunk)
                           continue 

                        # Yield chunks in Server-Sent Events (SSE) format
                        chunk_data = StreamChunk(chunk=chunk.text, model=request.model)
                        yield f"data: {chunk_data.model_dump_json()}\n\n"
                        await asyncio.sleep(0.01) # Small sleep to prevent blocking event loop entirely

                    # Signal stream end (optional, depends on client handling)
                    # yield f"data: {json.dumps({'end': True})}\n\n"

                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    # Try to send an error message over the stream if possible
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"
                finally:
                    # Clean up resources after generation completes or fails
                    mx.clear_cache()
                    gc.collect()
                    print("Stream finished, cleared cache.")

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            # Non-streaming response
            try:
                # Use generate from utils
                output = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=request.image,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    verbose=False, # Keep API output clean
                    **kwargs
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                # Create response
                result = GenerationResponse(
                    text=output,
                    model=request.model,
                    usage={} # TODO
                )
                return result

            except Exception as e:
                 print(f"Error during generation: {e}")
                 traceback.print_exc()
                 # Clean up resources even on error
                 mx.clear_cache()
                 gc.collect()
                 raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error in /generate endpoint: {e}")
        traceback.print_exc()
        # Clean up resources on unexpected error
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/batch_processing", response_model=List[GenerationResponse]) 
async def generate_endpoint(request: VLMRequest):

    """
    Uses the same prompt for each image in the batch in the --image argument
    TODO: change generation function to actually use the batch size and not only
    iterate over the images. This is a temporary solution as a placeholder.

    NOTE: ideally, generate.py should be refactored to allow reusing the same code here
    but for now, we just copy the logic from generate.py to avoid merging issues.
    """

    if len(request.image) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Too many images. Maximum is {MAX_IMAGES}.")
    
    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(request.model, request.adapter_path)
        
        prompt = codecs.decode(request.prompt, "unicode_escape")
        
        prompt = apply_chat_template(processor, config, prompt, num_images=1)

        kwargs = {}
        if request.resize_shape is not None:
            if len(request.resize_shape) not in [1, 2]:
                raise HTTPException(status_code=400, detail="resize_shape must contain exactly two integers (height, width)")
            kwargs["resize_shape"] = (
                (request.resize_shape[0],) * 2
                if len(request.resize_shape) == 1
                else tuple(request.resize_shape)
            )

        # TODO
        try:
            batched_result = []
            for image in request.image:
                # Use generate from utils
                output = generate(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    image=image,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    verbose=False, # Keep API output clean
                    **kwargs
                )

                # Clean up resources
                mx.clear_cache()
                gc.collect()

                # Create response
                result = GenerationResponse(
                    text=output,
                    model=request.model,
                    usage={} # Populate with actual usage stats if available
                )

                batched_result.append(result)

            return result

        except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                # Clean up resources even on error
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error in /generate endpoint: {e}")
        traceback.print_exc()
        # Clean up resources on unexpected error
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get("/health")
async def health_check():
    """
    Check if the server is healthy and what model is loaded.
    """
    return {
        "status": "healthy",
        "loaded_model": model_cache.get("model_path", None),
        "loaded_adapter": model_cache.get("adapter_path", None),
    }

@app.post("/unload")
async def unload_model_endpoint():
    """
    Unload the currently loaded model from memory.
    """
    unloaded_info = {
        "model_name": model_cache.get("model_path", None),
        "adapter_name": model_cache.get("adapter_path", None),
    }

    if not unload_model_sync(): # Use the synchronous unload function
         return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": f"Model unloaded successfully",
        "unloaded": unloaded_info
    }

if __name__ == "__main__":
    uvicorn.run("mlx_vlm.server:app", host="0.0.0.0", port=8000, workers=1, reload=True) # reload=True for development to automatically restart on code changes.

