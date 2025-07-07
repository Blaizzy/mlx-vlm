import asyncio
import codecs
import gc
import json
import traceback
import uuid
from datetime import datetime
from typing import Any, List, Literal, Optional, Tuple, Union

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing_extensions import Required, TypeAlias, TypedDict

from .generate import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    generate,
    stream_generate,
)
from .prompt_utils import apply_chat_template
from .utils import load
from .version import __version__

app = FastAPI(
    title="MLX-VLM Inference API",
    description="API for using Vision Language Models (VLMs) and Omni Models (Vision, Audio and Video support) with MLX.",
    version=__version__,
)

MAX_IMAGES = 10  # Maximum number of images to process at once

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
        # Use the load function from utils.py which handles path resolution and loading
        model, processor = load(model_path, adapter_path, trust_remote_code=True)
        config = model.config
        print("Model and processor loaded successfully.")
        return model, processor, config
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        traceback.print_exc()  # Print detailed traceback for debugging
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


def get_cached_model(model_path: str, adapter_path: Optional[str] = None):
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
        unload_model_sync()  # Use a synchronous version for internal call

    # Load the model resources
    model, processor, config = load_model_resources(model_path, adapter_path)

    model_cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
    }

    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync():
    global model_cache
    if not model_cache:
        return False

    print(
        f"Unloading model: {model_cache.get('model_path')}, Adapter: {model_cache.get('adapter_path')}"
    )
    # Clear references
    model_cache = {}
    # Force garbage collection
    gc.collect()
    mx.clear_cache()
    print("Model unloaded and cache cleared.")
    return True


# OpenAI API Models


class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["input_text"]
    ]  # The type of the input item. Always `input_text`.


class ResponseInputImageParam(TypedDict, total=False):
    detail: Literal["high", "low", "auto"] = Field(
        "auto", description="The detail level of the image to be sent to the model."
    )
    """The detail level of the image to be sent to the model.

    One of `high`, `low`, or `auto`. Defaults to `auto`.
    """
    type: Required[
        Literal["input_image"]
    ]  # The type of the input item. Always `input_image`.
    image_url: Required[str]
    file_id: Optional[str]
    """The ID of the file to be sent to the model.
     NOTE : wouldn't this help the model if we passed the file_id as well to the vlm models
    """


ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam, ResponseInputImageParam
]

ResponseInputMessageContentListParam: TypeAlias = List[ResponseInputContentParam]


class ResponseOutputText(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["output_text"]
    ]  # The type of the output item. Always `output_text`


ResponseOutputMessageContentList: TypeAlias = List[ResponseOutputText]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "developer"] = Field(
        ...,
        description="Role of the message sender (e.g., 'system', 'user', 'assistant').",
    )
    content: Union[
        str, ResponseInputMessageContentListParam, ResponseOutputMessageContentList
    ] = Field(..., description="Content of the message.")


class OpenAIRequest(BaseModel):
    """
    OpenAI-compatible request structure.
    Using this structure : https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
    """

    input: Union[str, List[ChatMessage]] = Field(
        ..., description="Input text or list of chat messages."
    )
    model: str = Field(..., description="The model to use for generation.")
    max_output_tokens: int = Field(
        DEFAULT_MAX_TOKENS, description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )


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
    object: Literal["response"] = Field(
        ..., description="The object type of this resource - always set to response"
    )
    created_at: int = Field(
        ..., description="Unix timestamp (in seconds) of when this Response was created"
    )
    status: Literal["completed", "failed", "in_progress", "incomplete"] = Field(
        ..., description="The status of the response generation"
    )
    error: Optional[OpenAIErrorObject] = Field(
        None,
        description="An error object returned when the model fails to generate a Response",
    )
    instructions: Optional[str] = Field(
        None,
        description="Inserts a system (or developer) message as the first item in the model's context",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        description="An upper bound for the number of tokens that can be generated for a response",
    )
    model: str = Field(..., description="Model ID used to generate the response")
    output: List[Union[ChatMessage, Any]] = Field(
        ..., description="An array of content items generated by the model"
    )
    output_text: Optional[str] = Field(
        None,
        description="SDK-only convenience property containing aggregated text output",
    )
    temperature: Optional[float] = Field(
        None, ge=0, le=2, description="Sampling temperature between 0 and 2"
    )
    top_p: Optional[float] = Field(
        None, ge=0, le=1, description="Nucleus sampling probability mass"
    )
    truncation: Union[Literal["auto", "disabled"], str] = Field(
        "disabled", description="The truncation strategy to use"
    )
    usage: OpenAIUsage = Field(
        ..., description="Token usage details"
    )  # we need the model to return stats
    user: Optional[str] = Field(
        None, description="A unique identifier representing your end-user"
    )


class BaseStreamEvent(BaseModel):
    type: str


class ContentPartOutputText(BaseModel):
    type: Literal["output_text"]
    text: str
    annotations: List[str] = []


class MessageItem(BaseModel):
    id: str
    type: Literal["message"]
    status: Literal["in_progress", "completed"]
    role: str
    content: List[ContentPartOutputText] = []


class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal["response.created"]
    response: OpenAIResponse


class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal["response.in_progress"]
    response: OpenAIResponse


class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal["response.output_item.added"]
    output_index: int
    item: MessageItem


class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal["response.content_part.added"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.output_text.delta"]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal["response.output_text.done"]
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal["response.content_part.done"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal["response.output_item.done"]
    output_index: int
    item: MessageItem


class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal["response.completed"]
    response: OpenAIResponse


StreamEvent = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
]


# OpenAI endpoint
@app.post("/responses")
async def openai_endpoint(request: Request):
    """
    OpenAI-compatible endpoint for generating text based on a prompt and optional images.

    using client.responses.create method.

    example:

    from openai import OpenAI

    API_URL = "http://0.0.0.0:8000"
    API_KEY = 'any'

    def run_openai(prompt, img_url,system, stream=False, max_output_tokens=512, model="mlx-community/Qwen2.5-VL-3B-Instruct-8bit"):
        ''' Calls the OpenAI API
        '''

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
                max_output_tokens=max_output_tokens,
                stream=stream
            )
            if not stream:
                print(response.output[0].content[0].text)
                print(response.usage)
            else:
                for event in response:
                    # Process different event types if needed
                    if hasattr(event, 'delta') and event.delta:
                        print(event.delta, end="", flush=True)
                    elif event.type == 'response.completed':
                        print("\n--- Usage ---")
                        print(event.response.usage)

        except Exception as e:
            # building a response object to match the one returned when request is successful so that it can be processed in the same way
            return {"model - error":str(e),"content":{}, "model":model}

    """

    body = await request.json()
    openai_request = OpenAIRequest(**body)

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(openai_request.model)

        kwargs = {}

        chat_messages = []
        images = []
        instructions = None
        if openai_request.input:
            if isinstance(openai_request.input, str):
                # If input is a string, treat it as a single text message
                chat_messages.append({"role": "user", "content": openai_request.input})
            elif isinstance(openai_request.input, list):
                # If input is a list, treat it as a series of chat messages
                for message in openai_request.input:
                    if isinstance(message, ChatMessage):
                        if isinstance(message.content, str):
                            chat_messages.append(
                                {"role": message.role, "content": message.content}
                            )
                            if message.role == "system":
                                instructions = message.content
                        elif isinstance(message.content, list):
                            # Handle list of content items
                            for item in message.content:
                                if isinstance(item, dict):
                                    if item["type"] == "input_text":
                                        chat_messages.append(
                                            {
                                                "role": message.role,
                                                "content": item["text"],
                                            }
                                        )
                                        if message.role == "system":
                                            instructions = item["text"]
                                    # examples for multiple images (https://platform.openai.com/docs/guides/images?api-mode=responses)
                                    elif item["type"] == "input_image":
                                        images.append(item["image_url"])
                                    else:
                                        print(
                                            f"invalid input item type: {item['type']}"
                                        )
                                        raise HTTPException(
                                            status_code=400,
                                            detail="Invalid input item type.",
                                        )
                                else:
                                    print(
                                        f"Invalid message content item format: {item}"
                                    )
                                    raise HTTPException(
                                        status_code=400,
                                        detail="Missing type in input item.",
                                    )
                        else:
                            print("Invalid message content format.")
                            raise HTTPException(
                                status_code=400, detail="Invalid input format."
                            )
                    else:
                        print("not a ChatMessage")
                        raise HTTPException(
                            status_code=400, detail="Invalid input format."
                        )
            else:
                print("neither string not list")
                raise HTTPException(status_code=400, detail="Invalid input format.")

        else:
            print("no input")
            raise HTTPException(status_code=400, detail="Missing input.")

        formatted_prompt = apply_chat_template(
            processor, config, chat_messages, num_images=len(images)
        )

        generated_at = datetime.now().timestamp()
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"

        if openai_request.stream:
            # Streaming response
            async def stream_generator():
                token_iterator = None
                try:
                    # Create base response object (to match the openai pipeline)
                    base_response = OpenAIResponse(
                        id=response_id,
                        object="response",
                        created_at=int(generated_at),
                        status="in_progress",
                        instructions=instructions,
                        max_output_tokens=openai_request.max_output_tokens,
                        model=openai_request.model,
                        output=[],
                        output_text="",
                        temperature=openai_request.temperature,
                        top_p=openai_request.top_p,
                        usage={
                            "input_tokens": 0,  # get prompt tokens
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    )

                    # Send response.created event  (to match the openai pipeline)
                    yield f"event: response.created\ndata: {ResponseCreatedEvent(type='response.created', response=base_response).model_dump_json()}\n\n"

                    # Send response.in_progress event  (to match the openai pipeline)
                    yield f"event: response.in_progress\ndata: {ResponseInProgressEvent(type='response.in_progress', response=base_response).model_dump_json()}\n\n"

                    # Send response.output_item.added event  (to match the openai pipeline)
                    message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="in_progress",
                        role="assistant",
                        content=[],
                    )
                    yield f"event: response.output_item.added\ndata: {ResponseOutputItemAddedEvent(type='response.output_item.added', output_index=0, item=message_item).model_dump_json()}\n\n"

                    # Send response.content_part.added event
                    content_part = ContentPartOutputText(
                        type="output_text", text="", annotations=[]
                    )
                    yield f"event: response.content_part.added\ndata: {ResponseContentPartAddedEvent(type='response.content_part.added', item_id=message_id, output_index=0, content_index=0, part=content_part).model_dump_json()}\n\n"

                    # Stream text deltas
                    token_iterator = stream_generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        temperature=openai_request.temperature,
                        max_tokens=openai_request.max_output_tokens,
                        top_p=openai_request.top_p,
                        **kwargs,
                    )

                    full_text = ""
                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, "text"):
                            continue

                        delta = chunk.text
                        full_text += delta

                        usage_stats = {
                            "input_tokens": chunk.prompt_tokens,
                            "output_tokens": chunk.generation_tokens,
                        }

                        # Send response.output_text.delta event
                        yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"
                        await asyncio.sleep(0.01)

                    # Send response.output_text.done event (to match the openai pipeline)
                    yield f"event: response.output_text.done\ndata: {ResponseOutputTextDoneEvent(type='response.output_text.done', item_id=message_id, output_index=0, content_index=0, text=full_text).model_dump_json()}\n\n"

                    # Send response.content_part.done event (to match the openai pipeline)
                    final_content_part = ContentPartOutputText(
                        type="output_text", text=full_text, annotations=[]
                    )
                    yield f"event: response.content_part.done\ndata: {ResponseContentPartDoneEvent(type='response.content_part.done', item_id=message_id, output_index=0, content_index=0, part=final_content_part).model_dump_json()}\n\n"

                    # Send response.output_item.done event (to match the openai pipeline)
                    final_message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[final_content_part],
                    )
                    yield f"event: response.output_item.done\ndata: {ResponseOutputItemDoneEvent(type='response.output_item.done', output_index=0, item=final_message_item).model_dump_json()}\n\n"

                    # Send response.completed event (to match the openai pipeline)
                    completed_response = base_response.model_copy(
                        update={
                            "status": "completed",
                            "output": [final_message_item],
                            "usage": {
                                "input_tokens": usage_stats["input_tokens"],
                                "output_tokens": usage_stats["output_tokens"],
                                "total_tokens": usage_stats["input_tokens"]
                                + usage_stats["output_tokens"],
                            },
                        }
                    )
                    yield f"event: response.completed\ndata: {ResponseCompletedEvent(type='response.completed', response=completed_response).model_dump_json()}\n\n"

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
                # Use generate from generate.py
                result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=images,
                    temperature=openai_request.temperature,
                    max_tokens=openai_request.max_output_tokens,
                    top_p=openai_request.top_p,
                    verbose=False,  # stats are passed in the response
                    **kwargs,
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                response = OpenAIResponse(
                    id=response_id,
                    object="response",
                    created_at=int(generated_at),
                    status="completed",
                    instructions=instructions,
                    max_output_tokens=openai_request.max_output_tokens,
                    model=openai_request.model,
                    output=[
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": result.text,
                                }
                            ],
                        }
                    ],
                    output_text=result.text,
                    temperature=openai_request.temperature,
                    top_p=openai_request.top_p,
                    usage={
                        "input_tokens": result.prompt_tokens,
                        "output_tokens": result.generation_tokens,
                        "total_tokens": result.total_tokens,
                    },
                )
                return response

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
        print(f"Unexpected error in /responses endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


# MLX_VLM Models
class VLMRequest(BaseModel):
    model: str = Field(
        DEFAULT_MODEL_PATH,
        description="The path to the local model directory or Hugging Face repo.",
    )
    adapter_path: Optional[str] = Field(
        None, description="The path to the adapter weights."
    )
    image: List[str] = Field(
        default_factory=list,
        description="List of URLs or local paths of images to process.",
    )
    audio: List[str] = Field(
        default_factory=list,
        description="List of URLs or local paths of audio files to process.",
    )
    prompt: str = Field(
        DEFAULT_PROMPT, description="Message to be processed by the model."
    )
    system: Optional[str] = Field(
        None,
        description="Optional system message for the model (used in chat structure).",
    )
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS, description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    resize_shape: Optional[Tuple[int, int]] = Field(
        None,
        description="Resize shape for the image (height, width). Provide two integers.",
    )


class GenerationRequest(VLMRequest):
    """
    Inherits from VLMRequest and adds additional fields for the generation request.
    """

    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )


class ChatRequest(GenerationRequest):
    """
    Inherits from GenerationRequest and adds fields specific to chat interactions.
    """

    prompt: List[ChatMessage] = Field(
        default_factory=list, description="List of chat messages for the conversation."
    )


class UsageStats(OpenAIUsage):
    """
    Inherits from OpenAIUsage and adds additional fields for usage statistics.
    """

    prompt_tps: float = Field(..., description="Tokens per second for the prompt.")
    generation_tps: float = Field(
        ..., description="Tokens per second for the generation."
    )
    peak_memory: float = Field(
        ..., description="Peak memory usage during the generation."
    )


class GenerationResponse(BaseModel):
    text: str
    model: str
    usage: Optional[UsageStats]


class StreamChunk(BaseModel):
    chunk: str
    model: str
    usage: Optional[UsageStats]


# MLX_VLM API endpoints


@app.post(
    "/generate", response_model=None
)  # Response model handled dynamically based on stream flag
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
                raise HTTPException(
                    status_code=400,
                    detail="resize_shape must contain exactly two integers (height, width)",
                )
            kwargs["resize_shape"] = (
                (request.resize_shape[0],) * 2
                if len(request.resize_shape) == 1
                else tuple(request.resize_shape)
            )

        # Prepare the prompt using the chat template logic
        chat_messages = []
        if request.system:
            system_prompt = request.system
            chat_messages.append({"role": "system", "content": system_prompt})
        prompt = request.prompt
        chat_messages.append({"role": "user", "content": prompt})

        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(request.image),
            num_audios=len(request.audio),
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
                        image=request.image if request.image else None,
                        audio=request.audio,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                        **kwargs,
                    )
                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, "text"):
                            print("Warning: Received unexpected chunk format:", chunk)
                            continue

                        # Yield chunks in Server-Sent Events (SSE) format
                        usage_stats = {
                            "input_tokens": chunk.prompt_tokens,
                            "output_tokens": chunk.generation_tokens,
                            "total_tokens": chunk.prompt_tokens
                            + chunk.generation_tokens,
                            "prompt_tps": chunk.prompt_tps,
                            "generation_tps": chunk.generation_tps,
                            "peak_memory": chunk.peak_memory,
                        }
                        chunk_data = StreamChunk(
                            chunk=chunk.text, model=request.model, usage=usage_stats
                        )
                        yield f"data: {chunk_data.model_dump_json()}\n\n"
                        await asyncio.sleep(
                            0.01
                        )  # Small sleep to prevent blocking event loop entirely

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
                # Use generate from generate.py
                gen_result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=request.image if request.image else None,
                    audio=request.audio,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    verbose=False,  # stats are passed in the response
                    **kwargs,
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                usage_stats = UsageStats(
                    input_tokens=gen_result.prompt_tokens,
                    output_tokens=gen_result.generation_tokens,
                    total_tokens=gen_result.total_tokens,
                    prompt_tps=gen_result.prompt_tps,
                    generation_tps=gen_result.generation_tps,
                    peak_memory=gen_result.peak_memory,
                )
                result = GenerationResponse(
                    text=gen_result.text, model=request.model, usage=usage_stats
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
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post(
    "/chat", response_model=None
)  # Response model handled dynamically based on stream flag
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
                raise HTTPException(
                    status_code=400,
                    detail="resize_shape must contain exactly two integers (height, width)",
                )
            kwargs["resize_shape"] = (
                (request.resize_shape[0],) * 2
                if len(request.resize_shape) == 1
                else tuple(request.resize_shape)
            )

        chat_messages = request.prompt

        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(request.image),
            num_audios=len(request.audio),
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
                        audio=request.audio,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                        **kwargs,
                    )

                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, "text"):
                            print("Warning: Received unexpected chunk format:", chunk)
                            continue

                        # Yield chunks in Server-Sent Events (SSE) format
                        usage_stats = {
                            "input_tokens": chunk.prompt_tokens,
                            "output_tokens": chunk.generation_tokens,
                            "total_tokens": chunk.prompt_tokens
                            + chunk.generation_tokens,
                            "prompt_tps": chunk.prompt_tps,
                            "generation_tps": chunk.generation_tps,
                            "peak_memory": chunk.peak_memory,
                        }
                        chunk_data = StreamChunk(
                            chunk=chunk.text, model=request.model, usage=usage_stats
                        )
                        yield f"data: {chunk_data.model_dump_json()}\n\n"
                        await asyncio.sleep(
                            0.01
                        )  # Small sleep to prevent blocking event loop entirely

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
                # Use generate from generate.py
                gen_result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=request.image,
                    audio=request.audio,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    verbose=False,  # Keep API output clean
                    **kwargs,
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                usage_stats = UsageStats(
                    input_tokens=gen_result.prompt_tokens,
                    output_tokens=gen_result.generation_tokens,
                    total_tokens=gen_result.total_tokens,
                    prompt_tps=gen_result.prompt_tps,
                    generation_tps=gen_result.generation_tps,
                    peak_memory=gen_result.peak_memory,
                )
                result = GenerationResponse(
                    text=gen_result.text, model=request.model, usage=usage_stats
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
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


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

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": f"Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():

    uvicorn.run(
        "mlx_vlm.server:app", host="0.0.0.0", port=8000, workers=1, reload=True
    )  # reload=True for development to automatically restart on code changes.


if __name__ == "__main__":
    main()
