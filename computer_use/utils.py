import os
from io import BytesIO

import pandas as pd
import requests
from PIL import Image, ImageDraw


REQUEST_TIMEOUT = (5, 10)
MAX_IMAGE_DOWNLOAD_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def draw_point(
    image_input, point: tuple = None, radius: int = 8, color: str = "red"
) -> Image.Image:
    """
    Draw a point on an image and return the modified image.
    `point` should be (x_norm, y_norm) in normalized coordinates (0 to 1).
    """
    # Load image from path/URL or use directly if it's already a PIL Image
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            image_bytes = BytesIO()
            downloaded = 0
            with requests.get(
                image_input, timeout=REQUEST_TIMEOUT, stream=True
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        if int(content_length) > MAX_IMAGE_DOWNLOAD_SIZE:
                            raise ValueError("Remote image is too large")
                    except ValueError:
                        pass
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_IMAGE_DOWNLOAD_SIZE:
                        raise ValueError("Remote image is too large")
                    image_bytes.write(chunk)
            image_bytes.seek(0)
            image = Image.open(image_bytes)
        else:
            image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be a string path/URL or a PIL Image object")

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions are too large")

    # Only draw if a valid point is provided
    if point is not None:
        x = int(point[0])
        y = int(point[1])
        print(f"Drawing ellipse at pixel coordinates: ({x}, {y})")

        draw = ImageDraw.Draw(image)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    return image


# Update CSV with query, response and image path
def update_navigation_history(
    query, response, filepath, csv_path="navigation_history.csv"
):
    """
    Update the navigation history CSV file with the query, response and screenshot filepath.

    Args:
        query: The user's query/task
        response: The system's response/action
        filepath: Path to the screenshot image
        csv_path: Path to the CSV file (default: navigation_history.csv)
    """
    # Create new row as a DataFrame
    new_row = pd.DataFrame(
        {"Query": [query], "Response": [str(response)], "Screenshot Path": [filepath]}
    )

    if os.path.exists(csv_path):
        # Append to existing CSV
        new_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        # Create new CSV with headers
        new_row.to_csv(csv_path, index=False)
