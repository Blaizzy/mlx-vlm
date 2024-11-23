import json
import re
from typing import Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_bbox(model_type, x_min, y_min, x_max, y_max, image_width, image_height):
    # Normalize coordinates
    if model_type == "paligemma":
        y_min_norm = max(0, min(y_min / 1024 * image_height, image_height))
        x_min_norm = max(0, min(x_min / 1024 * image_width, image_width))
        y_max_norm = max(0, min(y_max / 1024 * image_height, image_height))
        x_max_norm = max(0, min(x_max / 1024 * image_width, image_width))
    elif model_type == "qwen":
        y_min_norm = y_min / 1000 * image_height
        x_min_norm = x_min / 1000 * image_width
        y_max_norm = y_max / 1000 * image_height
        x_max_norm = x_max / 1000 * image_width
    return x_min_norm, y_min_norm, x_max_norm, y_max_norm


def plot_image_with_bboxes(image, bboxes=None, labels=None, model_type="qwen"):
    # Load the image if it's a string path, otherwise use the image object
    image = Image.open(image) if isinstance(image, str) else image
    image_width, image_height = image.size

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    if bboxes is not None:
        # Check if bboxes is a list of dictionaries
        if isinstance(bboxes, list) and all(isinstance(bbox, dict) for bbox in bboxes):
            # Define colors for different objects
            num_boxes = len(bboxes)
            colors = plt.cm.rainbow(np.linspace(0, 1, num_boxes))

            # Iterate over each bbox dictionary
            for i, (bbox, color) in enumerate(zip(bboxes, colors)):
                label = bbox.get("object", None)
                bbox_coords = bbox.get("bboxes", None)
                x_min, y_min, x_max, y_max = bbox_coords[0]

                # Normalize coordinates
                x_min_norm, y_min_norm, x_max_norm, y_max_norm = normalize_bbox(
                    model_type, x_min, y_min, x_max, y_max, image_width, image_height
                )

                # Calculate width and height
                width = x_max_norm - x_min_norm
                height = y_max_norm - y_min_norm

                # Create and add the rectangle patch
                rect = patches.Rectangle(
                    (x_min_norm, y_min_norm),
                    width,
                    height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add label if available
                if label is not None:
                    ax.text(
                        x_min_norm,
                        y_min_norm,
                        label,
                        color=color,
                        fontweight="bold",
                        bbox=dict(facecolor="white", edgecolor=color, alpha=0.8),
                    )
        else:
            # Original code for handling bboxes as a flat list
            num_boxes = len(bboxes) // 4
            colors = plt.cm.rainbow(np.linspace(0, 1, num_boxes))

            for i, color in zip(range(0, len(bboxes), 4), colors):
                x_min, y_min, x_max, y_max = bboxes[i : i + 4]

                # Normalize coordinates
                x_min_norm, y_min_norm, x_max_norm, y_max_norm = normalize_bbox(
                    model_type, x_min, y_min, x_max, y_max, image_width, image_height
                )

                # Calculate width and height
                width = x_max_norm - x_min_norm
                height = y_max_norm - y_min_norm

                # Create and add the rectangle patch
                rect = patches.Rectangle(
                    (x_min_norm, y_min_norm),
                    width,
                    height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add label if available
                if labels is not None and i // 4 < len(labels):
                    ax.text(
                        x_min_norm,
                        y_min_norm,
                        labels[i // 4],
                        color=color,
                        fontweight="bold",
                        bbox=dict(facecolor="white", edgecolor=color, alpha=0.8),
                    )

    plt.axis("off")
    plt.tight_layout()


def swap_x_y(bboxes):
    bboxes[0], bboxes[1] = bboxes[1], bboxes[0]
    bboxes[2], bboxes[3] = bboxes[3], bboxes[2]
    return bboxes


def parse_bbox(bbox_str, model_type="paligemma"):
    if model_type == "paligemma":
        objects = []
        bboxes = bbox_str.split(";")
        for bbox in bboxes:
            reg = re.finditer(r"<loc(\d+)>", bbox)
            name = bbox.split(">")[-1].strip()
            objects.append(
                {
                    "object": name,
                    "bboxes": [swap_x_y([int(loc.group(1)) for loc in reg])],
                }
            )
        return objects
    if model_type == "qwen":
        return json.loads(bbox_str.replace("```json", "").replace("```", ""))


def parse_points(points_str):
    """Parse points from XML-like string into x and y coordinates"""
    # Handle case where points_str is already a tuple
    if isinstance(points_str, tuple):
        return points_str

    # Extract x and y coordinates from attributes
    x_coords = []
    y_coords = []

    # Handle multi-point format
    if 'x1="' in points_str:
        i = 1
        while True:
            try:
                x = float(points_str.split(f'x{i}="')[1].split('"')[0])
                y = float(points_str.split(f'y{i}="')[1].split('"')[0])
                x_coords.append(x)
                y_coords.append(y)
                i += 1
            except IndexError:
                break
    # Handle single point format
    elif 'x="' in points_str:
        x = float(points_str.split('x="')[1].split('"')[0])
        y = float(points_str.split('y="')[1].split('"')[0])
        x_coords.append(x)
        y_coords.append(y)

    # Extract labels from points string
    try:
        labels = points_str.split('alt="')[1].split('">')[0].split(", ")
        item_labels = labels  # Use the alt labels to maintain order
    except IndexError:
        # If no labels found, create numbered labels
        item_labels = [f"Point {i+1}" for i in range(len(x_coords))]

    return x_coords, y_coords, item_labels


# Parse the points string
def plot_locations(points: Union[str, tuple], image):
    # Parse points
    if isinstance(points, str):
        x_coords, y_coords, item_labels = parse_points(points)
    else:
        x_coords, y_coords, item_labels = points

    # Create figure and axis
    plt.figure(figsize=(10, 8))

    # Get image dimensions for normalization
    img_width, img_height = image.size

    # Normalize coordinates
    x_norm = [(x / 100) * img_width for x in x_coords]
    y_norm = [(y / 100) * img_height for y in y_coords]
    if len(item_labels) != len(x_norm):
        item_labels *= len(x_norm)

    plt.imshow(image)

    # Plot points with different colors for each label
    colors = ["red", "blue", "green", "orange", "purple"]
    for i, (x, y, label) in enumerate(zip(x_norm, y_norm, item_labels)):
        color = colors[i % len(colors)]
        plt.plot(x, y, "o", color=color, markersize=5, label=label)

        # Add individual labels
        plt.annotate(
            label,
            (x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=color,
        )

    plt.title("Object Locations")
    plt.legend()
    plt.show()
