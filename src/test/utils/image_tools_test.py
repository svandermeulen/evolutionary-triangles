"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 26/05/2020
"""

import cv2
import numpy as np
import os
import unittest

from PIL import Image

from src.config import Config
from src.utils.image_tools import compute_distance, draw_triangle, draw_text, resize_image
from src.utils.logger import Logger


class TestImageTools(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.image_base = Image.new('RGB', (500, 500), (255, 255, 255))

    def test_compute_distance(self):
        img1 = cv2.cvtColor(np.uint8(np.zeros((128, 128, 4))), cv2.COLOR_RGBA2BGR)
        img2 = cv2.cvtColor(np.uint8(np.ones((128, 128, 4)) * 255), cv2.COLOR_RGBA2BGR)
        mean_distance = compute_distance(img1=img1, img2=img2)
        self.assertEqual(mean_distance, 255)

        Logger().info("Testing with real image")
        img2 = cv2.imread(os.path.join(self.config.path_data, "test_flower.jpg"))
        img1 = cv2.cvtColor(np.uint8(np.zeros(img2.shape)), cv2.COLOR_RGBA2BGR)
        mean_distance = compute_distance(img1=img1, img2=img2)
        self.assertEqual(np.round(mean_distance, 3), 144.259)

    def test_draw_triangle(self):
        image_triangle = draw_triangle(
            image=self.image_base,
            triangle=[(0, 0), (0, 500), (500, 500)],
            color=(100, 0, 100, 255)
        )

        self.assertIsInstance(image_triangle, Image.Image)
        image = np.array(image_triangle)
        self.assertEqual(image[0, 0, 0], 100)
        self.assertEqual(image[0, 0, 1], 0)
        self.assertEqual(image[0, 0, 2], 100)

    def test_draw_triangles(self):
        image_triangle = draw_triangle(
            image=self.image_base.copy(),
            triangle=[(0, 0), (0, 500), (500, 500)],
            color=(100, 0, 100, 55)
        )
        image_triangle = draw_triangle(
            image=image_triangle,
            triangle=[(500, 0), (0, 500), (500, 500)],
            color=(100, 0, 100, 125)
        )

        image_triangle = draw_triangle(
            image=image_triangle,
            triangle=[(0, 0), (150, 100), (0, 250)],
            color=(0, 100, 100, 125)
        )

        # image_triangle.show()
        image_array = np.array(image_triangle)
        self.assertIsInstance(image_triangle, Image.Image)
        self.assertEqual(image_array[249, 0, 0], self._compute_alpha_interpolation(
            color1=0, alpha1=125, color2=100, alpha2=55, colorbg=255
        ))
        self.assertEqual(image_array[249, 0, 1], self._compute_alpha_interpolation(
            color1=100, alpha1=125, color2=0, alpha2=55, colorbg=255
        ))
        self.assertEqual(image_array[249, 0, 2], self._compute_alpha_interpolation(
            color1=100, alpha1=125, color2=100, alpha2=55, colorbg=255
        ))

    def test_draw_text(self):
        image_text = draw_text(
            image=self.image_base.copy(),
            text="TEST",
            text_color=(0, 0, 0, 255)
        )
        # image_text.show()
        self.assertIsInstance(image_text, Image.Image)
        image_array = np.array(image_text)
        self.assertEqual(image_array[243, 219, 0], 0)

    def test_resize_image(self):
        image = cv2.imread(os.path.join(self.config.path_data, "test_panda.jpg"))
        image_resized = resize_image(image=image)
        self.assertEqual(image_resized.shape[0], 256)

    @staticmethod
    def _compute_alpha_interpolation(color1: int, alpha1: int, color2: int, alpha2: int, colorbg: int) -> int:
        """
        Found interpolation logic here:
        http://web.cse.ohio-state.edu/~parent.1/classes/581/Lectures/13.TransparencyHandout.pdf
        """

        alpha1 = alpha1 / 255
        alpha2 = alpha2 / 255

        a = color1 * alpha1
        b = (1 - alpha1) * (alpha2 * color2)
        c = (1 - alpha1) * (1 - alpha2) * colorbg
        return int(round(sum((a, b, c)), 0))


if __name__ == '__main__':
    unittest.main()
