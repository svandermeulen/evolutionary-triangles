"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""

import cv2
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple

from src.utils.polygon_tools import generate_random_triangles, generate_delaunay_triangles, convert_delaunay_points
from src.utils.profiler import profile


def draw_triangle(image: Image.Image, triangle: List[Tuple], color: tuple) -> Image:
    drw = ImageDraw.Draw(image, "RGBA")
    drw.polygon(triangle, color)

    return image


def draw_text(image: Image.Image, text: str, text_color: tuple, font: ImageFont.ImageFont = None) -> Image:

    if font is None:

        font_size = int(image.width * 0.05)
        font = ImageFont.truetype('C:\Windows\Fonts\Arialbd.ttf', font_size)

    text_width, text_height = font.getsize(text)
    text_position = ((image.width - text_width) / 2, (image.height - text_height) / 2)

    drw = ImageDraw.Draw(image, "RGBA")
    drw.multiline_text(text_position, text, fill=text_color, font=font)
    return image


def get_mask(img: np.ndarray) -> np.ndarray:
    return np.all(img == 255, axis=2)


def apply_mask(img_to_mask: np.ndarray, img_source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_to_mask[mask] = img_source[mask]
    return img_to_mask


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


def generate_triangle_image(width: int, height: int, triangles: np.ndarray = None, triangulation_method: str = "random") -> Image:
    image_pil = Image.new('RGB', (width, height), color=(255, 255, 255))

    if triangles is None:
        if triangulation_method == "random":
            triangles = generate_random_triangles(xmax=width, ymax=height)
        else:
            triangles = generate_delaunay_triangles(xmax=width, ymax=height)

    if triangulation_method == "non_overlapping":
        triangles_new = convert_delaunay_points(points=triangles[:, :2])
        triangles_new = np.hstack([triangles_new[:, :, 0], triangles_new[:, :, 1]])
        df_triangles = pd.DataFrame(triangles_new, columns=["x1", "x2", "x3", "y1", "y2", "y3"])
        df_points = pd.DataFrame(triangles, columns=["x", "y", "c1", "c2", "c3", "c4"])

        for i in range(3):
            if i == 0:
                df_m = df_triangles.merge(df_points, left_on=["x1", "y1"], right_on=["x", "y"]).drop(columns=["x", "y"])
            else:
                df_m = df_m.merge(df_points, left_on=[f"x{i+1}", f"y{i+1}"], right_on=["x", "y"]).drop(columns=["x", "y"])

        for i in range(4):
            df_m[f"c{i+1}"] = df_m[[c for c in df_m if c.startswith(f"c{i+1}")]].median(axis=1).astype(int)

        triangles = df_m.drop(columns=[c for c in df_m if "_" in c]).values

    for triangle in triangles:
        coordinates = [(triangle[i], triangle[i + 3]) for i in range(3)]
        image_pil = draw_triangle(
            image=image_pil,
            triangle=coordinates,
            color=tuple(triangle[6:])
        )

    return image_pil


def main():
    pass


if __name__ == "__main__":
    main()
