import ast
import json
import logging
import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw


def omni_decode_points(output: str) -> List[List[float]]:
    """
    Universal decoder to parse 2D point coordinates from model string outputs.
    Supports a wide variety of VLM formats (Qwen, GPT-4V, XML-style, etc.).

    Supported Formats:
    1. Qwen/JSON Dict: '[{"point_2d": [[x, y]], "label": "target"}]'
    2. List of Lists/Tuples: '[[x1, y1], [x2, y2]]' or '[(x1, y1)]'
    3. XML Tags: '<point>[[x, y]]</point>' or '<points>[x1,y1],[x2,y2]</points>'
    4. XML Attributes: '<point x="63.5" y="44.5" alt="label">text</point>'
    5. Raw Coordinates: 'The point is at 100, 200'
    6. Markdown Code Blocks: '```json\n[x, y]\n```'

    Args:
        output: The raw string output from the model.

    Returns:
        List[List[float]]: A list of [x, y] coordinates. Returns empty list if no points found.
    """
    if not isinstance(output, str) or not output.strip():
        return []

    points = []

    # --- Strategy 1: XML Attribute Extraction ---
    # Matches: <point x="10" y="20"> or <point y="20" x="10">
    if '<point' in output.lower():
        points = _extract_from_xml_attributes(output)
        if points:
            return points

    # --- Strategy 2: Clean Markdown & XML Tags ---
    text = _preprocess_text(output)

    # --- Strategy 3: Python Literal / JSON Parsing ---
    # We try ast.literal_eval first because models often use single quotes or 
    # Python-like structures that valid JSON doesn't support.
    try:
        # Help literal_eval by removing common prefix labels like "Output: "
        clean_text = re.sub(r'^[a-zA-Z0-9_\s]+:\s*', '', text)
        data = ast.literal_eval(clean_text)
        points = _parse_structured_data(data)
        if points:
            return points
    except (ValueError, SyntaxError, MemoryError):
        pass

    # --- Strategy 4: Regex Fallback (The "Catch-All") ---
    # Matches: [10, 20], (10, 20), or even raw 10.5, 20.1
    # This captures points embedded in natural language.
    points = _extract_points_by_regex(text)
    
    return points

def _preprocess_text(text: str) -> str:
    """Removes markdown wrappers and extracts content from within XML-style tags."""
    # Remove Markdown blocks
    text = re.sub(r'```(?:json|python|html)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    
    # Extract content from <point> or <points> tags if they exist
    tag_match = re.search(r'<(?:point|points)>(.*?)</(?:point|points)>', text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        text = tag_match.group(1)
        
    return text.strip()

def _parse_structured_data(data: Any) -> List[List[float]]:
    """Recursively traverses Python objects to find lists/dicts representing points."""
    points = []
    
    if isinstance(data, dict):
        # Handle Qwen/VLM specific keys
        for key in ["point_2d", "points", "point", "coordinates"]:
            if key in data:
                return _parse_structured_data(data[key])
                
    elif isinstance(data, (list, tuple)):
        if not data:
            return []
            
        # Check if it's a flat point: [x, y]
        if len(data) == 2 and all(isinstance(x, (int, float)) for x in data):
            return [[float(data[0]), float(data[1])]]
        
        # Check if it's a nested structure: [[x, y], ...] or [{"point_2d": [x, y]}, ...]
        for item in data:
            extracted = _parse_structured_data(item)
            if extracted:
                points.extend(extracted)
                
    return points


def _extract_from_xml_attributes(text):
    all_points = []
    for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)

    for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)
    for match in re.finditer(r'(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})', text):
        try:
            point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)

    return all_points


def _extract_points_by_regex(text: str) -> List[List[float]]:
    """Regex to find coordinate pairs in loosely formatted text."""
    points = []
    # Pattern for [x, y] or (x, y)
    bracket_pattern = r'[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]'
    matches = re.findall(bracket_pattern, text)
    
    if matches:
        for m in matches:
            points.append([float(m[0]), float(m[1])])
    else:
        # Last resort: look for "Number, Number" patterns in the text
        # Only if no bracketed points were found to avoid duplicates
        raw_pattern = r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
        matches = re.findall(raw_pattern, text)
        for m in matches:
            points.append([float(m[0]), float(m[1])])
            
    return points


def vis_point(
    image: Union[Image.Image, str],
    points: List[List[float]],
    save_path: str,
    outer_radius: int = 8,
    inner_radius: int = 5,
) -> Image.Image:
    """
    Visualize normalized points (range [0, 1000]) on an image and save it.

    Point style: white outer circle + solid red inner circle.

    Args:
        image: PIL image object or image path.
        points: Normalized points [[x, y], ...], each coordinate in [0, 1000].
        save_path: Output path for visualization image.
        outer_radius: Radius of white outer circle.
        inner_radius: Radius of red inner circle.

    Returns:
        PIL.Image.Image: The visualization image.
    """
    if isinstance(image, str):
        vis_image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        vis_image = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if not points:
        vis_image.save(save_path)
        return vis_image

    width, height = vis_image.size
    draw = ImageDraw.Draw(vis_image)

    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        x_norm, y_norm = float(point[0]), float(point[1])

        # points are normalized to [0, 1000]
        x_pixel = int(round((x_norm / 1000.0) * (width - 1)))
        y_pixel = int(round((y_norm / 1000.0) * (height - 1)))

        x_pixel = max(0, min(width - 1, x_pixel))
        y_pixel = max(0, min(height - 1, y_pixel))

        draw.ellipse(
            [
                x_pixel - outer_radius,
                y_pixel - outer_radius,
                x_pixel + outer_radius,
                y_pixel + outer_radius,
            ],
            fill="white",
        )
        draw.ellipse(
            [
                x_pixel - inner_radius,
                y_pixel - inner_radius,
                x_pixel + inner_radius,
                y_pixel + inner_radius,
            ],
            fill="red",
        )

    vis_image.save(save_path)
    return vis_image


def check_points_in_mask(points: List[List[float]], mask: Image.Image) -> Tuple[int, int]:
    """
    Check if points (absolute pixel coordinates) are in the mask foreground.
    
    Args:
        points: List of points in absolute pixel coordinates [[x1, y1], [x2, y2], ...]
        mask: PIL Image mask where foreground pixels have value > 0
        
    Returns:
        Tuple[int, int]: (number of points in mask, total number of points)
    """
    if not points or mask is None:
        return 0, 0
    
    # Convert to grayscale
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    width, height = mask.size
    mask_array = np.array(mask)
    
    points_in_mask = 0
    total_points = len(points)
    
    for point in points:
        x_pixel = int(round(point[0]))
        y_pixel = int(round(point[1]))
        
        # Check if coordinates are within image bounds
        if 0 <= x_pixel < width and 0 <= y_pixel < height:
            # mask_array is in [y, x] format
            if mask_array[y_pixel, x_pixel] > 0:
                points_in_mask += 1
    
    return points_in_mask, total_points


def check_masks_coverage(points: List[List[float]], masks: List[Image.Image]) -> Tuple[int, int]:
    """
    Check how many masks have at least one point in them.

    Args:
        points: List of points in absolute pixel coordinates [[x1, y1], [x2, y2], ...]
        masks: List of PIL Image masks where foreground pixels have value > 0

    Returns:
        Tuple[int, int]: (number of masks with at least one point, total number of masks)
    """
    if not points or not masks:
        return 0, len(masks) if masks else 0

    total_masks = len(masks)
    masks_with_points = 0

    for mask in masks:
        # Convert to grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')

        width, height = mask.size
        mask_array = np.array(mask)

        # Check if any point is in this mask
        has_point = False
        for point in points:
            x_pixel = int(round(point[0]))
            y_pixel = int(round(point[1]))

            # Check if coordinates are within image bounds
            if 0 <= x_pixel < width and 0 <= y_pixel < height:
                # mask_array is in [y, x] format
                if mask_array[y_pixel, x_pixel] > 0:
                    has_point = True
                    break  # Found at least one point in this mask, move to next mask

        if has_point:
            masks_with_points += 1

    return masks_with_points, total_masks


def check_points_in_bbox(points: List[List[float]], bbox: List[float]) -> Tuple[int, int]:
    """
    Check if points (absolute pixel coordinates) are within a bounding box.
    
    Args:
        points: List of points in absolute pixel coordinates [[x1, y1], [x2, y2], ...]
        bbox: Bounding box in format [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right
        
    Returns:
        Tuple[int, int]: (number of points in bbox, total number of points)
    """
    if not points or bbox is None or len(bbox) < 4:
        return 0, 0
    
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    points_in_bbox = 0
    total_points = len(points)
    
    for point in points:
        x_pixel = int(round(point[0]))
        y_pixel = int(round(point[1]))
        
        # Check if point is within bounding box
        if x1 <= x_pixel <= x2 and y1 <= y_pixel <= y2:
            points_in_bbox += 1
    
    return points_in_bbox, total_points


if __name__ == "__main__":
    test_cases = [
        '[{"point_2d": [[100, 200]], "label": "eye"}]',        # Qwen-style
        '<point x="63.5" y="44.5">Mountain</point><point x="63.8" y="44.5">Mountain</point>',           # Tag attributes
        '```json\n[[10, 20], [30, 40]]\n```',                  # Markdown
        'The center is at (500, 500) and (789, 1000).',                        # Natural language
        '<points>[123, 456]</points>',
        '<points>[[122, 333], [222, 333]]</points>',           # Custom tags
        'point: 12.5, 13.5',                                   # Lazy labeling
    ]
    
    for case in test_cases:
        print(f"Input: {case}")
        print(f"Parsed: {omni_decode_points(case)}\n")
