import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

__all__ = [
    "random_polygon_mask",
    "blend_texture",
    "create_synthetic_deforestation",
    "_rounded_rect_mask",
]

def _rect_mask(h: int, w: int, min_area: float, max_area: float) -> np.ndarray:
    """Generate rectangle mask within area bounds."""
    mask = np.zeros((h, w), np.uint8)
    for _ in range(30):
        rect_w = random.randint(int(w * 0.05), int(w * 0.9))
        rect_h = random.randint(int(h * 0.05), int(h * 0.9))
        area_ratio = (rect_w * rect_h) / (h * w)
        if not (min_area <= area_ratio <= max_area):
            continue
        x = random.randint(0, w - rect_w)
        y = random.randint(0, h - rect_h)
        cv2.rectangle(mask, (x, y), (x + rect_w, y + rect_h), 1, -1)
        return mask
    return mask  # empty fallback


def _ellipse_mask(h: int, w: int, min_area: float, max_area: float) -> np.ndarray:
    """Generate ellipse mask within area bounds."""
    mask = np.zeros((h, w), np.uint8)
    for _ in range(30):
        axis_a = random.randint(int(w * 0.05), int(w * 0.45))
        axis_b = random.randint(int(h * 0.05), int(h * 0.45))
        area_ratio = (np.pi * axis_a * axis_b) / (h * w)
        if not (min_area <= area_ratio <= max_area):
            continue
        center_x = random.randint(axis_a, w - axis_a)
        center_y = random.randint(axis_b, h - axis_b)
        angle = random.randint(0, 180)
        cv2.ellipse(mask, (center_x, center_y), (axis_a, axis_b), angle, 0, 360, 1, -1)
        return mask
    return mask


def _polygon_mask(h: int, w: int, min_area_ratio: float, max_area_ratio: float, num_vertices: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for _ in range(30):
        num_pts = random.randint(*num_vertices)
        pts = np.stack([
            np.random.randint(0, w, size=num_pts),
            np.random.randint(0, h, size=num_pts)
        ], axis=-1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        area = mask.sum() / (h * w)
        if min_area_ratio <= area <= max_area_ratio:
            return mask
        mask.fill(0)
    cv2.fillConvexPoly(mask, pts, 1)
    return mask


def _rounded_rect_mask(h: int, w: int, min_area: float, max_area: float, corner_radius_ratio: float = 0.2) -> np.ndarray:
    """Generate rectangle with rounded corners mask within area bounds."""
    mask = np.zeros((h, w), np.uint8)
    for _ in range(30):
        rect_w = random.randint(int(w * 0.05), int(w * 0.9))
        rect_h = random.randint(int(h * 0.05), int(h * 0.9))
        area_ratio = (rect_w * rect_h) / (h * w)
        if not (min_area <= area_ratio <= max_area):
            continue

        x = random.randint(0, w - rect_w)
        y = random.randint(0, h - rect_h)

        radius = int(min(rect_w, rect_h) * corner_radius_ratio)
        if radius < 2:
            radius = 2

        # Draw central rectangle without corners
        cv2.rectangle(mask, (x + radius, y), (x + rect_w - radius, y + rect_h), 1, -1)
        cv2.rectangle(mask, (x, y + radius), (x + rect_w, y + rect_h - radius), 1, -1)

        # Four corner circles
        cv2.circle(mask, (x + radius, y + radius), radius, 1, -1)
        cv2.circle(mask, (x + rect_w - radius, y + radius), radius, 1, -1)
        cv2.circle(mask, (x + radius, y + rect_h - radius), radius, 1, -1)
        cv2.circle(mask, (x + rect_w - radius, y + rect_h - radius), radius, 1, -1)

        return mask
    return mask


def generate_mask(
    h: int,
    w: int,
    shape: str = "polygon",
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.4,
    num_vertices: Tuple[int, int] = (5, 12),
) -> np.ndarray:
    """Factory to create mask of given shape."""
    shape = shape.lower()
    if shape == "rect" or shape == "rectangle":
        return _rect_mask(h, w, min_area_ratio, max_area_ratio)
    if shape == "ellipse" or shape == "oval":
        return _ellipse_mask(h, w, min_area_ratio, max_area_ratio)
    if shape == "polygon":
        return _polygon_mask(h, w, min_area_ratio, max_area_ratio, num_vertices)
    if shape in ("roundrect", "rounded_rect", "rounded-rect", "roundedrect"):
        return _rounded_rect_mask(h, w, min_area_ratio, max_area_ratio)
    if shape == "mixed":
        return generate_mask(h, w, random.choice(["rect", "ellipse", "polygon"]), min_area_ratio, max_area_ratio)
    raise ValueError(f"Unknown shape: {shape}")

def blend_texture(
    forest: np.ndarray,
    texture: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Seamlessly blend ``texture`` into ``forest`` where mask==1.

    Uses OpenCV seamlessClone which needs a source patch the same size as destination mask.
    """
    h, w = forest.shape[:2]
    # resize texture if needed
    texture_resized = cv2.resize(texture, (w, h), interpolation=cv2.INTER_LINEAR)
    # Position for seamlessClone (center of image)
    center = (w // 2, h // 2)
    mask_3c = (mask * 255).astype(np.uint8)
    clone = cv2.seamlessClone(texture_resized, forest, mask_3c, center, cv2.NORMAL_CLONE)
    return clone

def create_synthetic_deforestation(
    forest_img: np.ndarray,
    deforested_img: np.ndarray,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a composite image and its mask.

    Returns:
        (composite_rgb, mask) where mask is uint8 {0,1}
    """
    h, w = forest_img.shape[:2]
    # default shape is polygon unless otherwise specified via global var passed in kwargs
    shape_choice = globals().get("_current_shape_choice", "polygon")
    mask = generate_mask(h, w, shape_choice, min_area_ratio, max_area_ratio)

    composite = blend_texture(forest_img, deforested_img, mask)
    return composite, mask 