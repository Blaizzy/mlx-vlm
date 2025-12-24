# Take a screenshot of the screen
import json
import math
import os
import time
import tkinter as tk
from tkinter import Label

import mlx.core as mx
import numpy as np
import pyautogui
import sounddevice as sd
import soundfile as sf
from PIL import ImageGrab
from pynput.mouse import Controller as MouseController
from rich import print
from utils import draw_point, update_navigation_history

from mlx_vlm import load
from mlx_vlm.utils import generate

min_pixels = 256 * 28 * 28
max_pixels = 1512 * 982

WIDTH = 1512
HEIGHT = 982

mouse = MouseController()


def animate_cursor_movement(start_x, start_y, end_x, end_y, duration=0.3):
    """Animate the cursor movement with a smooth transition"""
    steps = 20
    for i in range(steps + 1):
        t = i / steps
        x = start_x + (end_x - start_x) * t
        y = start_y + (end_y - start_y) * t
        mouse.position = (x, y)
        time.sleep(duration / steps)


def highlight_click(x, y, duration=0.2):
    """Create a visual highlight effect at click position"""
    current_pos = mouse.position
    # Move to the target position
    animate_cursor_movement(current_pos[0], current_pos[1], x, y)
    # Perform a small circular motion to highlight
    radius = 5
    steps = 10
    for i in range(steps):
        angle = 2 * 3.14159 * i / steps
        mouse.position = (x + radius * math.cos(angle), y + radius * math.sin(angle))
        time.sleep(duration / steps)
    # Return to center
    mouse.position = (x, y)


def create_action_overlay(action_type, x, y, color="red", duration=1.0):
    """Create a temporary overlay showing the action type and a red circle"""
    root = tk.Tk()
    root.overrideredirect(True)  # Remove window decorations
    root.attributes("-topmost", True)  # Keep on top
    root.attributes("-alpha", 0.7)  # Make semi-transparent
    root.config(bg="black")

    # Position the window near the action point
    offset_x, offset_y = 20, 20
    root.geometry(f"+{x + offset_x}+{y + offset_y}")

    # Create label with action text
    label = Label(
        root, text=action_type, fg=color, bg="black", font=("Arial", 12, "bold")
    )
    label.pack(padx=10, pady=5)

    # Create a canvas for the circle
    canvas = tk.Canvas(root, width=60, height=60, bg="black", highlightthickness=0)
    canvas.pack()

    # Draw a red circle
    canvas.create_oval(5, 5, 55, 55, outline=color, width=2)

    # Schedule the window to close after duration
    root.after(int(duration * 1000), root.destroy)

    # Update the window to show it immediately
    root.update()

    return root


def click(position):
    # position = extract_point(start_box)
    position = (round(int(position[0])), round(int(position[1])))
    print("Clicking at position", position)

    # Create overlay showing the action
    overlay = create_action_overlay("CLICK", position[0], position[1])

    # Animate the cursor movement
    highlight_click(position[0], position[1])
    pyautogui.click(x=position[0], y=position[1], interval=0.2, clicks=2, button="left")

    # Keep overlay visible for a moment after action
    overlay.after(500, overlay.destroy)


def left_double(position):
    # position = extract_point(start_box)
    position = (round(int(position[0])), round(int(position[1])))

    # Create overlay showing the action
    overlay = create_action_overlay("DOUBLE CLICK", position[0], position[1])

    # Animate the cursor movement
    highlight_click(position[0], position[1])
    pyautogui.doubleClick(x=position[0], y=position[1], interval=0.2)

    # Keep overlay visible for a moment after action
    overlay.after(500, overlay.destroy)


def input_text(position, value):
    """Input text at the specified position"""
    x, y = position
    position = (round(int(x)), round(int(y)))

    # Create overlay showing the action
    overlay = create_action_overlay("TYPE", position[0], position[1])

    # Animate cursor and perform action
    highlight_click(position[0], position[1])
    pyautogui.click(x=position[0], y=position[1])
    pyautogui.write(value)

    # Keep overlay visible for a moment
    overlay.after(500, overlay.destroy)


def select(position):
    """Select at the specified position"""
    x, y = position
    position = (round(int(x)), round(int(y)))

    # Create overlay showing the action
    overlay = create_action_overlay("SELECT", position[0], position[1])

    # Animate cursor and perform action
    highlight_click(position[0], position[1])
    pyautogui.click(x=position[0], y=position[1])

    # Keep overlay visible for a moment
    overlay.after(500, overlay.destroy)


def hover(position):
    """Hover at the specified position"""
    x, y = position
    position = (round(int(x)), round(int(y)))

    # Create overlay showing the action
    overlay = create_action_overlay("HOVER", position[0], position[1])

    # Animate cursor movement
    highlight_click(position[0], position[1])
    pyautogui.moveTo(x=position[0], y=position[1], duration=0.1)

    # Keep overlay visible for a moment
    overlay.after(500, overlay.destroy)


def answer(value):
    """Provide an answer"""


def enter():
    """Perform enter action"""
    pyautogui.press("enter")


def scroll(value):
    """Scroll in specified direction"""
    clicks = 20 if value.lower() == "down" else -20

    # Create overlay showing the action
    overlay = create_action_overlay("SCROLL " + value.upper(), 100, 100)

    pyautogui.scroll(clicks)

    # Keep overlay visible for a moment
    overlay.after(500, overlay.destroy)


def select_text(start_pos, end_pos):
    """Select text between two positions"""
    x1, y1 = start_pos
    x2, y2 = end_pos
    start_pos = (round(int(x1)), round(int(y1)))
    end_pos = (round(int(x2)), round(int(y2)))

    # Create overlay showing the action
    overlay = create_action_overlay("SELECT TEXT", start_pos[0], start_pos[1])

    # Animate cursor and perform action
    highlight_click(start_pos[0], start_pos[1])
    pyautogui.moveTo(x=start_pos[0], y=start_pos[1])
    pyautogui.mouseDown()
    highlight_click(end_pos[0], end_pos[1])
    pyautogui.moveTo(x=end_pos[0], y=end_pos[1])
    pyautogui.mouseUp()

    # Keep overlay visible for a moment
    overlay.after(500, overlay.destroy)


def copy(text):
    """Copy specified text"""
    import pyperclip

    pyperclip.copy(text)


def finished():
    """Finish the task"""
    print("Task completed successfully!")


def wait():
    pyautogui.sleep(5)


def finished():
    print("Task completed")


def call_user():
    print("Requesting user assistance")


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
    "FINISHED": {"function": finished, "args": []},
    "WAIT": {"function": wait, "args": []},
    "CALL_USER": {"function": call_user, "args": []},
}


def build_messages(system_prompt, query, past_actions=[]):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                # {"type": "text", "text": system_prompt},
                {"type": "text", "text": query}
            ],
        },
    ]

    # Add past action pairs
    for action in past_actions:
        # Assistant response
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": action}]}
        )

        # User response without image
        messages.append({"role": "user", "content": []})
    if len(past_actions) > 0:
        # Add image to final user message
        if messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": [{"type": "image"}]})
        else:
            messages[-1]["content"] = [{"type": "image"}]
    else:
        messages[-1]["content"] += [{"type": "image"}]

    return messages


from PIL import ImageDraw


def extract_thought_action(response):
    return (
        response.split("Thought:")[1].split("Action:")[0].strip(),
        response.split("Action:")[1],
    )


def extract_point(response):
    if "<|box_start|>" in response:
        if "Action:" in response:
            return (
                response.split("Action:")[1]
                .split("```")[0]
                .split("<|box_start|>")[1]
                .split("<|box_end|>")[0]
                .strip("()")
                .split(",")
            )
        else:
            return (
                response.split("<|box_start|>")[1]
                .split("<|box_end|>")[0]
                .strip("()")
                .split(",")
            )
    else:
        if "Action:" in response:
            return (
                response.split("Action:")[1]
                .split("start_box='")[1]
                .split("'")[0]
                .replace("(", "")
                .replace(")", "")
                .split(",")
            )
        else:
            if "(" in response:
                return response.replace("(", "").replace(")", "").split(",")
            else:
                return response.split(",")


def draw_point_area(image, point):
    radius = min(image.width, image.height) // 45
    x, y = round(point[0] / 1000 * image.width), round(point[1] / 1000 * image.height)
    ImageDraw.Draw(image).ellipse(
        (x - radius, y - radius, x + radius, y + radius), outline="red", width=2
    )
    ImageDraw.Draw(image).ellipse((x - 2, y - 2, x + 2, y + 2), fill="red")
    return image


PLANNER_SYSTEM_PROMPT = """
    You are a GUI agent. You are given a task and your action history, with screenshots. You need to plan the next action to complete the task.

    ## Output Format
    ```\nPlan: ...
    Thought: ...
    Action: ...\n```

    ## Action Space

    1. `CLICK`: Click on an element name
    2. `INPUT`: Type text at location with value string
    3. `SELECT`: Select element at location
    4. `HOVER`: Hover over element at location
    5. `ANSWER`: Provide answer with value string
    6. `ENTER`: Press enter key
    7. `SCROLL`: Scroll screen in direction value (up, down, left, right)
    8. `SELECT_TEXT`: Select text in location
    9. `COPY`: Copy specified text value
    10. `FINISHED`: Use this when the task is completed.
    11. `WAIT`: Wait for 5 seconds and take a screenshot to check for any changes.
    12. `CALL_USER`: Submit the task and call the user when the task is unsolvable, or when you need the user's help.


    ## Note
    - The screen's resolution is {max_pixels}.
    - Use English in `Thought` part.
    - Summarize your next action (with its target element) in one sentence in `Thought` part.
    - Use `finished()` when the task is completed.
    - Use `call_user()` when the task is unsolved after 5 attempts, or when you need the user's help.

    # User Instruction
    """


_NAV_SYSTEM = """You are an assistant trained to navigate the {_APP} screen.
Given a task instruction, a screen observation, and an action history sequence,
output the next action and wait for the next observation.


Make sure to use a wide range of actions to complete the task.

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
10. `FINISHED`: The task is finished, value and position are not applicable.
""",
}

GUI_AGENT_SYSTEM_PROMPT = _NAV_SYSTEM.format(
    _APP="computer", _ACTION_SPACE=action_map["computer"]
)


def save_image_navigation_history(query, screenshot, planner_response, response=None):
    try:
        # Create directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join("screenshots", filename)

        # Check if response is a string (like "finished()" or "call_user()")
        if isinstance(response, dict):
            if (
                not any(
                    action in planner_response.lower()
                    for action in ["finished", "call_user", "wait"]
                )
                and "position" in response
                and response["position"] is not None
            ):
                screenshot = draw_point(screenshot, point=response["position"])

        # Update CSV with query, response and image path
        if response is not None:
            combined_response = planner_response + "\n" + json.dumps(response)
        else:
            combined_response = planner_response

        update_navigation_history(query, combined_response, filepath)
        print("Updated navigation history")

        # Save the image
        screenshot.save(filepath)
        print(f"Saved image to {filepath}")

    except Exception as e:
        print(f"Error saving image: {str(e)}")
        raise e


def run_planner(model, processor, screenshot, system_prompt, query, past_actions=[]):

    messages = build_messages(
        system_prompt,
        query,
        past_actions,
    )
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
    return response


def run_gui_agent(
    gui_agent, gui_processor, screenshot, system_prompt, query, past_actions=[]
):
    screenshot = screenshot.resize((1512, 982))
    response = run_planner(
        gui_agent, gui_processor, screenshot, system_prompt, query, past_actions
    )
    return response


def process_command(model, processor, gui_agent, gui_processor, query, past_actions=[]):
    # Wait a second before taking screenshot
    time.sleep(1)

    # Take screenshot and add to list
    screenshot = ImageGrab.grab()
    screenshot = screenshot.resize((1512, 982))

    planner_response = run_planner(
        model, processor, screenshot, PLANNER_SYSTEM_PROMPT, query, past_actions
    )
    mx.metal.clear_cache()
    print("Planner response:\n", planner_response)
    if any(
        action in planner_response.lower()
        for action in ["finished", "call_user", "wait"]
    ):
        past_actions.append(planner_response)
        play_audio("/Users/prince_canuma/task_completed.wav")
        print("Task completed successfully!")
        screenshot = ImageGrab.grab()
        screenshot = screenshot.resize((1512, 982))
        save_image_navigation_history(query, screenshot, planner_response)
        return past_actions

    response = run_gui_agent(
        gui_agent,
        gui_processor,
        screenshot,
        GUI_AGENT_SYSTEM_PROMPT,
        planner_response,
        past_actions,
    )
    mx.metal.clear_cache()

    print("GUI Agent Response:\n", response)

    response = eval(response)
    if "position" in response and response["position"] is not None:
        # normalize the values
        response["position"] = [
            response["position"][0] * screenshot.width,
            response["position"][1] * screenshot.height,
        ]

    action_result = action_functions[response["action"]]
    values = {arg: response[arg] for arg in action_result["args"]}

    print(f"[green]Executing action:[/green] {response['action']}")
    action_result["function"](**values)

    save_image_navigation_history(query, screenshot, planner_response, response)

    # Add current action to past actions
    past_actions.append(planner_response + "\n" + json.dumps(response))
    return past_actions


def listen(mlx_whisper, mic, r):
    with mic as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        # Convert audio to numpy array
        audio_data = (
            np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        # Process audio with Apple MLXWhisper model
        query = mlx_whisper.transcribe(
            audio_data, path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
        )["text"]

        # Print the transcribed text
        print(f"\nHeard: {query}")
    return query


def play_audio(path):
    data, samplerate = sf.read(path)
    sd.play(data, samplerate)
    sd.wait()


def main():
    print("[bold blue]Screen Navigation Assistant[/bold blue]")
    print("Press Ctrl+C to quit")
    model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    gui_agent, gui_processor = load(
        "mlx-community/ShowUI-2B-bf16",
        processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
    )

    import mlx_whisper
    import speech_recognition as sr

    r = sr.Recognizer()
    mic = sr.Microphone(sample_rate=16000)
    past_actions = []
    counter = 0
    play_audio("./audio/ask_voice.wav")
    input("Press Enter to start listening...")
    query = listen(mlx_whisper, mic, r)
    # Play audio confirmation sound
    play_audio("./audio/ok.wav")

    try:
        while True:
            # Wait for a click to start listening
            if query.lower() == "exit":
                break
            try:
                past_actions = process_command(
                    model, processor, gui_agent, gui_processor, query, past_actions
                )
            except Exception as e:
                print(f"[red]Error:[/red] {str(e)}")
                raise e
            counter += 1
            if counter % 3 == 0 or any(
                action in past_actions[-1].lower()
                for action in ["finished", "call_user", "wait"]
            ):
                play_audio("./audio/ask_voice_2.wav")
                input("Press Enter to start listening...")
                query = listen(mlx_whisper, mic, r)
                past_actions = []

    except KeyboardInterrupt:
        print("Stopped listening.")


if __name__ == "__main__":
    main()
