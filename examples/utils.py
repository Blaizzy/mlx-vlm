import json
import re

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
