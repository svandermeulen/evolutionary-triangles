"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""
import cv2
import numpy as np
import os
import platform

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple


FONT_DEFAULT = 'C:\Windows\Fonts\Arialbd.ttf' if platform.system() == "Windows" else \
    "/opt/conda/lib/python3.8/site-packages/cv2/qt/fonts/DejaVuSans-Bold.ttf"


def draw_triangle(image: Image.Image, triangle: List[Tuple], color: tuple) -> Image:
    drw = ImageDraw.Draw(image, "RGBA")
    drw.polygon(triangle, color)

    return image


def draw_text(image: Image.Image, text: str, text_color: tuple, font: ImageFont.ImageFont = None) -> Image:
    if font is None and os.path.isfile(FONT_DEFAULT):
        font = FONT_DEFAULT
        font_size = int(image.width * 0.07)
        font = ImageFont.truetype(font=font, size=font_size)

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


def compute_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = convert_to_lab(img=img1)
    img2 = convert_to_lab(img=img2)
    return np.mean(np.linalg.norm(np.subtract(np.float64(img1), np.float64(img2)), axis=2))


def convert_pil_to_array(image_pil: Image) -> np.ndarray:
    image_pil = np.array(image_pil)
    return cv2.cvtColor(image_pil, cv2.COLOR_RGBA2BGR)


def show_image(image_pil: Image) -> bool:
    img = convert_pil_to_array(image_pil=image_pil)
    # img = np.array(image_pil)
    cv2.imshow("Show by CV2", img)
    cv2.waitKey(0)
    return True


def generate_triangle_image(width: int, height: int, triangles: np.ndarray) -> Image:
    image_pil = Image.new('RGB', (width, height), color=(255, 255, 255))

    for triangle in triangles:
        coordinates = [(triangle[i], triangle[i + 3]) for i in range(3)]
        image_pil = draw_triangle(
            image=image_pil,
            triangle=coordinates,
            color=tuple(triangle[6:])
        )

    return image_pil


def resize_image(image: np.ndarray, height_max: int = 256) -> np.ndarray:

    height, width, depth = image.shape
    resize_scale = height_max / height
    height_new, width_new = height_max, width * resize_scale
    return cv2.resize(image, (int(width_new), int(height_new)))


def main():
    pass


if __name__ == "__main__":
    main()
