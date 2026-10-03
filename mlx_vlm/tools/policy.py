"""OpenAI ``tool_choice`` semantics: validate the request and, for forced
choices, steer the model with an appended instruction.

Kept independent of any server module so both the OpenAI- and
Anthropic-compatible surfaces can share it.
"""

from typing import Any, Optional

from fastapi import HTTPException


def _tool_function_name(tool: Any) -> Optional[str]:
    if hasattr(tool, "model_dump"):
        tool = tool.model_dump(exclude_none=True)
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return None
    function = tool.get("function")
    if hasattr(function, "model_dump"):
        function = function.model_dump(exclude_none=True)
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _with_tool_choice_instruction(messages, instruction: str):
    messages = [dict(message) for message in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        messages[0]["content"] = f"{content}\n\n{instruction}".strip()
    user_instruction_added = False
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            message["content"] = f"{message['content']}\n\n{instruction}".strip()
            user_instruction_added = True
            break
    if not user_instruction_added and not (
        messages and messages[0].get("role") == "system"
    ):
        messages.insert(0, {"role": "system", "content": instruction})
    return messages


def _prepare_chat_tool_choice(messages, tools, tool_choice):
    """Validate and enforce OpenAI Chat Completions tool_choice semantics."""
    available_tools = list(tools or [])
    if tool_choice is None:
        return messages, available_tools or None, None

    if hasattr(tool_choice, "model_dump"):
        tool_choice = tool_choice.model_dump(exclude_none=True)

    if isinstance(tool_choice, str):
        if tool_choice not in ("none", "auto", "required"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid tool_choice. Expected 'none', 'auto', 'required', "
                    "or a specific function."
                ),
            )
        if tool_choice == "none":
            return messages, None, tool_choice
        if tool_choice == "auto":
            return messages, available_tools or None, tool_choice
        if not available_tools:
            raise HTTPException(
                status_code=400,
                detail="tool_choice 'required' requires at least one tool.",
            )
        instruction = (
            "You must call one or more of the available functions to answer the "
            "user's request. Do not answer directly without calling a function."
        )
        return (
            _with_tool_choice_instruction(messages, instruction),
            available_tools,
            tool_choice,
        )

    if not isinstance(tool_choice, dict):
        raise HTTPException(status_code=400, detail="Invalid tool_choice.")

    function = tool_choice.get("function")
    if hasattr(function, "model_dump"):
        function = function.model_dump(exclude_none=True)
    name = function.get("name") if isinstance(function, dict) else None
    if tool_choice.get("type") != "function" or not isinstance(name, str) or not name:
        raise HTTPException(
            status_code=400,
            detail=(
                "A specific tool_choice must be "
                "{'type':'function','function':{'name':'...'}}."
            ),
        )

    selected_tools = [
        tool for tool in available_tools if _tool_function_name(tool) == name
    ]
    if not selected_tools:
        raise HTTPException(
            status_code=400,
            detail=f"tool_choice references unknown function {name!r}.",
        )
    instruction = (
        f"You must call the {name!r} function to answer the user's request. "
        "Do not call any other function and do not answer directly."
    )
    return (
        _with_tool_choice_instruction(messages, instruction),
        selected_tools,
        tool_choice,
    )
