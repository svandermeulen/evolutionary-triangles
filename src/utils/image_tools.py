"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""

import cv2
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw
from typing import List, Tuple

from src.utils.polygon_tools import generate_random_triangles
from src.utils.profiler import profile


def draw_triangle(img: Image, triangle: List[Tuple], color: tuple) -> Image:
    drw = ImageDraw.Draw(img, 'RGBA')
    drw.polygon(triangle, color)

    return img


def convert_to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


@profile
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
    cv2.imshow("Show by CV2", img)
    cv2.waitKey(0)
    return True


def generate_triangle_image(width: int, height: int, triangles: np.ndarray = None) -> Image:
    image_pil = Image.new('RGBA', (width, height), color=(255, 255, 255, 255))

    triangles = triangles if triangles is not None else generate_random_triangles(xmax=width, ymax=height)

    for triangle in triangles:

        coordinates = [(triangle[i], triangle[i+2]) for i in range(3)]
        image_pil = draw_triangle(
            img=image_pil,
            triangle=coordinates,
            color=tuple(triangle[6:])
        )

    return image_pil


def main():
    pass


if __name__ == "__main__":
    main()
