"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""

import cv2
import numpy as np

from PIL import Image, ImageDraw
from typing import List, Tuple

from src.utils.polygon_tools import generate_random_triangles
from src.utils.profiler import profile


def draw_triangle(image: Image.Image, triangle: List[Tuple], color: tuple) -> Image:

    drw = ImageDraw.Draw(image, "RGBA")
    drw.polygon(triangle, color)

    return image


def get_mask(img: np.ndarray) -> np.ndarray:
    return np.all(img == 255, axis=2)


def apply_mask(img_to_mask: np.ndarray, img_source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_to_mask[mask] = img_source[mask]
    return img_to_mask


def blend_images(img1: Image.Image, img2: Image.Image) -> Image.Image:

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    img_blend = np.uint8(np.mean(np.array([img1_array, img2_array]), axis=0))

    mask = get_mask(img=img2_array)
    img_blend = apply_mask(img_to_mask=img_blend, img_source=img1_array, mask=mask)

    # mask = get_mask(img=img1_array)
    # img_blend = apply_mask(img_to_mask=img_blend, img_source=img2_array, mask=mask)

    return Image.fromarray(img_blend)


def draw_triangle_two(img: Image, triangle: List[Tuple], color: tuple) -> Image:
    drw = ImageDraw.Draw(img, 'RGBA')
    drw.polygon(triangle, color)

    return img


def convert_to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


# @profile
def compute_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = convert_to_lab(img=img1)
    img2 = convert_to_lab(img=img2)
    return np.mean(np.sqrt(np.sum(np.square(np.subtract(np.uint32(img1), np.uint32(img2))), axis=2)))


@profile
def compute_distance_two(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = convert_to_lab(img=img1)
    img2 = convert_to_lab(img=img2)
    return np.mean(np.linalg.norm(np.subtract(np.float64(img1), np.float64(img2)), axis=2))


def convert_pil_to_array(image_pil: Image) -> np.ndarray:
    image_pil = np.array(image_pil)
    return cv2.cvtColor(image_pil, cv2.COLOR_RGBA2BGR)


def compare_images(img1: np.ndarray, img2: Image, distance_max: float) -> (bool, float):
    img2 = convert_pil_to_array(image_pil=img2)
    dist = compute_distance(img1=img1, img2=img2)
    if dist <= distance_max:
        return True, dist
    return False, distance_max


def show_image(image_pil: Image) -> bool:
    img = convert_pil_to_array(image_pil=image_pil)
    # img = np.array(image_pil)
    cv2.imshow("Show by CV2", img)
    cv2.waitKey(0)
    return True


def generate_triangle_image(width: int, height: int, triangles: np.ndarray = None) -> Image:
    image_pil = Image.new('RGB', (width, height), color=(255, 255, 255))

    triangles = triangles if triangles is not None else generate_random_triangles(xmax=width, ymax=height)

    for triangle in triangles:

        coordinates = [(triangle[i], triangle[i+2]) for i in range(3)]
        image_pil = draw_triangle(
            image=image_pil,
            triangle=coordinates,
            color=tuple(triangle[6:])
        )
        # image_pil = blend_images(img1=image_pil, img2=image_triangle)

    return image_pil


def main():
    pass


if __name__ == "__main__":
    main()
