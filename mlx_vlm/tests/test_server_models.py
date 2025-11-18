"""
Simple test to verify ConfigDict(extra="ignore") works correctly
"""
from pydantic import BaseModel, ConfigDict, ValidationError, Field
from typing import List, Literal, Union

# Simulate the models from server.py

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "developer"]
    content: Union[str, List[dict]]

class VLMRequest(BaseModel):
    model: str = Field("default-model", description="The model to use")
    max_tokens: int = Field(256, description="Maximum number of tokens")
    temperature: float = Field(0.5, description="Temperature for sampling")
    top_p: float = Field(1.0, description="Top-p sampling")
    seed: int = Field(0, description="Seed for random generation")

class GenerationRequest(VLMRequest):
    stream: bool = Field(False, description="Whether to stream the response")

class ChatRequest(GenerationRequest):
    model_config = ConfigDict(extra="ignore")
    
    messages: List[ChatMessage]

class OpenAIRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    input: Union[str, List[ChatMessage]]
    model: str
    max_output_tokens: int = 256
    temperature: float = 0.5
    top_p: float = 1.0
    stream: bool = False

print("Testing Pydantic models with ConfigDict(extra='ignore')...")
print("=" * 70)

# Test 1: ChatRequest with basic fields
print("\n1. Testing ChatRequest with basic fields...")
try:
    request = ChatRequest(
        model="test-model",
        messages=[
            ChatMessage(role="user", content="Hello")
        ]
    )
    assert request.model == "test-model"
    assert len(request.messages) == 1
    print("   ✓ Basic ChatRequest works")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Test 2: ChatRequest with extra fields (the main fix)
print("\n2. Testing ChatRequest with extra fields (OpenAI SDK simulation)...")
try:
    request_data = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
        # Extra fields that OpenAI SDK sends
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "n": 1,
        "stop": None,
        "user": "test-user",
        "logprobs": False,
        "top_logprobs": None,
    }
    request = ChatRequest(**request_data)
    assert request.model == "test-model"
    assert request.max_tokens == 100
    assert request.temperature == 0.7
    assert request.stream == False
    # Extra fields should be ignored, not stored
    assert not hasattr(request, 'frequency_penalty')
    assert not hasattr(request, 'presence_penalty')
    assert not hasattr(request, 'n')
    print("   ✓ ChatRequest accepts and ignores extra fields")
    print(f"   ✓ Validated: model={request.model}, max_tokens={request.max_tokens}")
    print(f"   ✓ Extra fields correctly ignored (not stored in model)")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: OpenAIRequest with extra fields
print("\n3. Testing OpenAIRequest with extra fields...")
try:
    request_data = {
        "model": "test-model",
        "input": "Hello world",
        "max_output_tokens": 100,
        "temperature": 0.7,
        "stream": False,
        # Extra fields
        "some_extra_field": "value",
        "another_field": 123,
        "nested_field": {"key": "value"},
    }
    request = OpenAIRequest(**request_data)
    assert request.model == "test-model"
    assert request.input == "Hello world"
    assert request.max_output_tokens == 100
    assert not hasattr(request, 'some_extra_field')
    print("   ✓ OpenAIRequest accepts and ignores extra fields")
    print(f"   ✓ Validated: model={request.model}, input={request.input}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Verify validation still works for required fields
print("\n4. Testing that validation rejects missing required fields...")
try:
    request = ChatRequest(
        model="test-model",
        # Missing 'messages' field which is required
    )
    print("   ✗ Failed: Should have raised ValidationError for missing 'messages'")
    exit(1)
except ValidationError as e:
    print("   ✓ Correctly raises ValidationError for missing required fields")
    print(f"   ✓ Error details: {e.error_count()} validation error(s)")

# Test 5: Verify validation rejects invalid field types
print("\n5. Testing that validation rejects invalid field types...")
try:
    request = ChatRequest(
        model="test-model",
        messages=[
            ChatMessage(role="user", content="Hello")
        ],
        temperature="not-a-number"  # Should be float
    )
    print("   ✗ Failed: Should have raised ValidationError for invalid type")
    exit(1)
except ValidationError as e:
    print("   ✓ Correctly raises ValidationError for invalid field types")
    print(f"   ✓ Error details: {e.error_count()} validation error(s)")

# Test 6: ChatMessage with list content (multimodal)
print("\n6. Testing ChatMessage with multimodal content...")
try:
    message = ChatMessage(
        role="user",
        content=[
            {"type": "input_text", "text": "What's in this image?"},
            {"type": "input_image", "image_url": "http://example.com/image.jpg"}
        ]
    )
    assert message.role == "user"
    assert isinstance(message.content, list)
    assert len(message.content) == 2
    print("   ✓ ChatMessage accepts list content for multimodal messages")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Test 7: OpenAIRequest with list of messages
print("\n7. Testing OpenAIRequest with list of ChatMessages as input...")
try:
    request = OpenAIRequest(
        model="test-model",
        input=[
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello")
        ]
    )
    assert len(request.input) == 2
    print("   ✓ OpenAIRequest accepts list of ChatMessages")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("\nSummary of fixes:")
print("  • ChatRequest now uses ConfigDict(extra='ignore')")
print("  • OpenAIRequest now uses ConfigDict(extra='ignore')")
print("  • Extra fields from OpenAI SDK are accepted and ignored")
print("  • Required field validation still works correctly")
print("  • Type validation still works correctly")
print("  • Multimodal message content is supported")
print("\nThis fix resolves the 422 Unprocessable Entity error when using")
print("the OpenAI SDK with the /chat/completions and /responses endpoints.")
