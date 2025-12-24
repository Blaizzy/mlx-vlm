# Take a screenshot of the screen
import os
import time

import mlx.core as mx
from PIL import ImageGrab
from utils import draw_point, update_navigation_history

from mlx_vlm import load
from mlx_vlm.utils import generate

min_pixels = 256 * 28 * 28
max_pixels = 1512 * 982

_NAV_SYSTEM = """You are an assistant trained to navigate the {_APP} screen.
Given a task instruction, a screen observation, and an action history sequence,
output the next action and wait for the next observation.
Here is the action space:
{_ACTION_SPACE}
"""

_NAV_FORMAT = """
Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

action_map = {
    "computer": """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required.
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required.
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required.
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
""",
}

system_prompt = _NAV_SYSTEM.format(
    _APP="computer", _ACTION_SPACE=action_map["computer"]
)


# Create functions for all computer actions
def click(position):
    """Click at the specified position"""
    x, y = position
    import pyautogui

    pyautogui.click(x=x, y=y, interval=0.2, clicks=2, button="left")


def input_text(position, value):
    """Input text at the specified position"""
    x, y = position
    import pyautogui

    pyautogui.click(x=x, y=y)
    pyautogui.write(value)


def select(position):
    """Select at the specified position"""
    x, y = position
    import pyautogui

    pyautogui.click(x=x, y=y)


def hover(position):
    """Hover at the specified position"""
    x, y = position
    import pyautogui

    pyautogui.moveTo(x=x, y=y, duration=0.1)


def answer(value):
    """Provide an answer"""


def enter():
    """Perform enter action"""
    import pyautogui

    pyautogui.press("enter")


def scroll(value):
    """Scroll in specified direction"""
    import pyautogui

    clicks = 20 if value.lower() == "down" else -20
    pyautogui.scroll(clicks)


def select_text(start_pos, end_pos):
    """Select text between two positions"""
    x1, y1 = start_pos
    x2, y2 = end_pos
    import pyautogui

    pyautogui.moveTo(x=x1, y=y1)
    pyautogui.mouseDown()
    pyautogui.moveTo(x=x2, y=y2)
    pyautogui.mouseUp()


def copy(text):
    """Copy specified text"""
    import pyperclip

    pyperclip.copy(text)


# Map actions to functions
action_functions = {
    "CLICK": {"function": click, "args": ["position"]},
    "INPUT": {"function": input_text, "args": ["position", "value"]},
    "SELECT": {"function": select, "args": ["position"]},
    "HOVER": {"function": hover, "args": ["position"]},
    "ANSWER": {"function": answer, "args": ["value"]},
    "ENTER": {"function": enter, "args": []},
    "SCROLL": {"function": scroll, "args": ["value"]},
    "SELECT_TEXT": {"function": select_text, "args": ["start_pos", "end_pos"]},
    "COPY": {"function": copy, "args": ["text"]},
}


def process_command(model, processor, query, past_actions=[]):
    # Wait a second before taking screenshot
    time.sleep(1)
    # Take and resize screenshot
    screenshot = ImageGrab.grab()
    # screenshot = screenshot.resize((1512, 982)) # Comment this out to use a different resolution

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": f"Task: {query}"},
                {"type": "text", "text": f"Past actions: {past_actions}"},
                {"type": "image", "min_pixels": min_pixels, "max_pixels": max_pixels},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = generate(
        model,
        processor,
        prompt,
        screenshot,
        temperature=0.1,
        max_tokens=1000,
        verbose=False,
    )
    mx.metal.clear_cache()

    print("Response: ", response)

    response = eval(response)
    # normalize the values
    response["position"] = [
        response["position"][0] * screenshot.width,
        response["position"][1] * screenshot.height,
    ]

    action_result = action_functions[response["action"]]
    values = {arg: response[arg] for arg in action_result["args"]}

    print(f"\033[32mExecuting action:\033[0m {response['action']}")
    action_result["function"](**values)

    try:
        # Create directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join("screenshots", filename)

        screenshot = draw_point(screenshot, point=response["position"])

        # Update CSV with query, response and image path
        update_navigation_history(query, response, filepath)

        # Save the image
        screenshot.save(filepath)
        print(f"Saved image to {filepath}")

        return filepath

    except Exception as e:
        print(f"Error saving image: {str(e)}")
        raise

    # Add current action to past actions
    past_actions.append(response)
    return past_actions


def main():
    print("\033[34mScreen Navigation Assistant\033[0m")
    print("Type 'exit' to quit")
    model, processor = load(
        "mlx-community/ShowUI-2B-bf16",
        {"min_pixels": min_pixels, "max_pixels": max_pixels},
    )

    past_actions = []
    while True:
        query = input("What would you like me to do?")
        if query.lower() == "exit":
            break

        try:
            past_actions = process_command(model, processor, query, past_actions)
        except Exception as e:
            print(f"\033[31mError:\033[0m {str(e)}")


if __name__ == "__main__":
    main()
