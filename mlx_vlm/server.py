from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import (
    List, 
    Optional, 
    Dict, 
    Any, 
    Union, 
    Tuple, 
    Literal, 
    TypeAlias, 
    TypedDict, 
    Required
)
import uvicorn
import gc
import mlx.core as mx
import json
import asyncio
import traceback 
import codecs
import datetime
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

# Loading/unloading utilities

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
        return False 

    print(f"Unloading model: {model_cache.get('model_path')}, Adapter: {model_cache.get('adapter_path')}")
    # Clear references
    model_cache = {}
    # Force garbage collection
    gc.collect()
    mx.clear_cache() 
    print("Model unloaded and cache cleared.")
    return True

# API Models

class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    """The text input to the model."""

    type: Required[Literal["input_text"]]
    """The type of the input item. Always `input_text`."""


class ResponseInputImageParam(TypedDict, total=False):
    detail: Literal["high", "low", "auto"] = Field("auto", description="The detail level of the image to be sent to the model.")
    """The detail level of the image to be sent to the model.

    One of `high`, `low`, or `auto`. Defaults to `auto`.
    """

    type: Required[Literal["input_image"]]
    """The type of the input item. Always `input_image`."""

    image_url: Required[str]
    """The URL of the image to be sent to the model.

    A fully qualified URL or base64 encoded image in a data URL.
    """

    file_id: Optional[str]
    """The ID of the file to be sent to the model.
     NOTE : wouldn't this help the model if we passed the file_id as well to the vlm models
    
    """

ResponseInputContentParam: TypeAlias = Union[ResponseInputTextParam, ResponseInputImageParam]

ResponseInputMessageContentListParam: TypeAlias = List[ResponseInputContentParam]

class ResponseOutputText(TypedDict, total=False):
    text: Required[str]
    """The text input to the model."""

    type: Required[Literal["output_text"]]
    """The type of the output item. Always `output_text`."""

ResponseOutputMessageContentList: TypeAlias = List[ResponseOutputText]

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "developer"] = Field(..., description="Role of the message sender (e.g., 'system', 'user', 'assistant').")
    content: Union[str, ResponseInputMessageContentListParam,ResponseOutputMessageContentList] = Field(..., description="Content of the message.")

class OpenAIRequest(BaseModel):
    """
    OpenAI-compatible request structure.
    Using this structure : https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
    """
    input: Union[str, List[ChatMessage]] = Field(..., description="Input text or list of chat messages.")
    model: str = Field(..., description="The model to use for generation.")
    max_output_tokens: int = Field(DEFAULT_MAX_TOKENS, description="Maximum number of tokens to generate.")
    temperature: float = Field(DEFAULT_TEMPERATURE, description="Temperature for sampling.")
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    stream: bool = Field(False, description="Whether to stream the response chunk by chunk.")
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    resize_shape: Optional[Tuple[int, int]] = Field(None, description="Resize shape for the image (height, width). Provide two integers.")
    adapter_path: Optional[str] = Field(None, description="The path to the adapter weights.")

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

class OpenAIUsage(BaseModel):
    """Token usage details including input tokens, output tokens, breakdown, and total tokens used."""
    input_tokens: int
    output_tokens: int
    total_tokens: int 

class OpenAIErrorObject(BaseModel):
    """Error object returned when the model fails to generate a Response."""
    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str] = None

class OpenAIResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for this Response")
    object: Literal["response"] = Field(..., description="The object type of this resource - always set to response")
    created_at: int = Field(..., description="Unix timestamp (in seconds) of when this Response was created")
    status: Literal["completed", "failed", "in_progress", "incomplete"] = Field(..., description="The status of the response generation")
    error: Optional[OpenAIErrorObject] = Field(None, description="An error object returned when the model fails to generate a Response")
    instructions: Optional[str] = Field(None, description="Inserts a system (or developer) message as the first item in the model's context")
    max_output_tokens: Optional[int] = Field(None, description="An upper bound for the number of tokens that can be generated for a response")
    model: str = Field(..., description="Model ID used to generate the response")
    output: List[Union[ChatMessage, Any]] = Field(..., description="An array of content items generated by the model")
    output_text: Optional[str] = Field(None, description="SDK-only convenience property containing aggregated text output")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Sampling temperature between 0 and 2")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="Nucleus sampling probability mass")
    truncation: Union[Literal["auto", "disabled"], str] = Field("disabled", description="The truncation strategy to use")
    usage: OpenAIUsage = Field(..., description="Token usage details") # we need the model to return stats
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user")

class GenerationResponse(BaseModel):
    text: str
    model: str
    usage: Dict[str, Any] # Placeholder for verbose usage stats : we need the model to return stats

class StreamChunk(BaseModel):
    chunk: str
    model: str # placeholder (TBC).

# API endpoints

@app.post("/responses") 
async def openai_endpoint(request : Request):
    """
    OpenAI-compatible endpoint for generating text based on a prompt and optional images.

    using client.responses.create method.

    example:

    from openai import OpenAI

    API_URL = "http://0.0.0.0:8000"
    API_KEY = 'any'

    def run_openai(prompt, img_url,system, model="mlx-community/Qwen2.5-VL-3B-Instruct-8bit"): 
        ''' Calls the OpenAI API'''
        
        client = OpenAI(base_url=f"{API_URL}", api_key=API_KEY)
        
        try : 
            response = client.responses.create(
                model=model,
                input=[
                    {"role":"system",
                    "content": f"{system}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"{img_url}"},
                        ],
                    }
                ], 
            )

            return response.output_text
            
        except Exception as e:
            # building a response object to match the one returned when request is successful so that it can be processed in the same way
            return {"model - error":str(e),"content":{}, "model":model}

    """

    body = await request.json()
    openai_request = OpenAIRequest(**body)

    print(openai_request)

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(openai_request.model, openai_request.adapter_path)

        kwargs = {}

        if openai_request.resize_shape is not None:
            if len(openai_request.resize_shape) not in [1, 2]:
                raise HTTPException(status_code=400, detail="resize_shape must contain exactly two integers (height, width)")
            kwargs["resize_shape"] = (
                (openai_request.resize_shape[0],) * 2
                if len(openai_request.resize_shape) == 1
                else tuple(openai_request.resize_shape)
            )

        chat_messages = []
        images= []
        instructions = None
        if openai_request.input:
            if isinstance(openai_request.input, str):
                # If input is a string, treat it as a single text message
                chat_messages.append({"role": "user", "content": openai_request.input})
            elif isinstance(openai_request.input, list):
                # If input is a list, treat it as a series of chat messages
                for message in openai_request.input:
                    if isinstance(message, dict) and "role" in message and "content" in message:
                        if isinstance(message["content"], str):
                            chat_messages.append({"role": message["role"], "content": message["content"]})
                            if message["role"] == "system":
                                instructions = message["content"]
                        elif isinstance(message["content"], list):
                            # Handle list of content items
                            for item in message["content"]:
                                if isinstance(item, dict) and "type" in item :
                                    if item["type"] == "input_text" and "text" in item:
                                        chat_messages.append({"role": message["role"], "content": item["text"]})
                                        if message["role"] == "system":
                                            instructions = item["text"]
                                    # examples for multiple images (https://platform.openai.com/docs/guides/images?api-mode=responses)
                                    elif item["type"] == "input_image":
                                        images.append(item["image_url"])
                                    else:
                                        raise HTTPException(status_code=400, detail="Invalid input item type.")
                                else:
                                    raise HTTPException(status_code=400, detail="Missing type in input item.")
                        else:
                            raise HTTPException(status_code=400, detail="Invalid input format.")
                    else:
                        raise HTTPException(status_code=400, detail="Invalid input format.")
            else:
                raise HTTPException(status_code=400, detail="Invalid input format.")
        
        else:
                raise HTTPException(status_code=400, detail="Missing input.")

        # For now, assume it works or adapt it as needed.
        formatted_prompt = apply_chat_template(
            processor, config, chat_messages, num_images=len(images)
        )

        generated_at = datetime.now().timestamp()

        if openai_request.stream:
            # Streaming response
            async def stream_generator():
                token_iterator = None
                try:
                    # Use stream_generate from utils
                    token_iterator = stream_generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        temperature=openai_request.temperature,
                        max_tokens=openai_request.max_output_tokens,
                        top_p=openai_request.top_p,
                        **kwargs
                    )
                    
                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, 'text'):
                           print("Warning: Received unexpected chunk format:", chunk)
                           continue 

                        # TODO
                        # Yield chunks in Server-Sent Events (SSE) format
                        chunk_data = StreamChunk(chunk=chunk.text, model=openai_request.model)
                        yield f"data: {chunk_data.model_dump_json()}\n\n"
                        await asyncio.sleep(0.01) # Small sleep to prevent blocking event loop entirely

                    # Signal stream end (optional, depends on client handling)
                    # yield f"data: {json.dumps({'end': True})}\n\n"

                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
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
                    image=images,
                    temperature=openai_request.temperature,
                    max_tokens=openai_request.max_output_tokens,
                    top_p=openai_request.top_p,
                    verbose=False, # Keep API output clean
                    **kwargs
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                # Create response
                result = OpenAIResponse(
                    id= f"generation-{generated_at}",
                    object= "response",
                    created_at= int(generated_at),
                    status= "completed",
                    instructions= instructions,
                    max_output_tokens= openai_request.max_output_tokens,
                    model= openai_request.model,
                    output= [
                        {
                        "role": "assistant",
                        "content": [
                            {
                            "type": "output_text",
                            "text": output,
                            }
                        ]
                        }
                    ],
                    output_text= output,
                    temperature= openai_request.temperature,
                    top_p= openai_request.top_p,
                    usage= { # TODO
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0
                    }
                )
                return result

            except Exception as e:
                 print(f"Error during generation: {e}")
                 traceback.print_exc()
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
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


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
        chat_messages = []
        if request.system:
            system_prompt = codecs.decode(request.system, "unicode_escape")
            chat_messages.append({"role": "system", "content": system_prompt})
        prompt = codecs.decode(request.prompt, "unicode_escape")
        chat_messages.append({"role": "user", "content": prompt})

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
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"
                
                finally:
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
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/chat", response_model=None) # Response model handled dynamically based on stream flag
async def generate_endpoint(request: GenerationRequest):
    """
    Generate text based on a prompt and optional images.
    Prompt must be a list of chat messages, including system, user, and assistant messages.
    System message will be ignored if not already in the prompt.
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
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
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

