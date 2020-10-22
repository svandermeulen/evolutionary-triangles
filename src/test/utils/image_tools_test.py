"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 26/05/2020
"""

import cv2
import numpy as np
import os
import unittest

from PIL import Image, ImageDraw

from src.config import Config
from src.utils.image_tools import compute_distance, compute_distance_two, draw_triangle, blend_images


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

    def test_compute_distance_two(self):

        img1 = cv2.cvtColor(np.uint8(np.zeros((128, 128, 4))), cv2.COLOR_RGBA2BGR)
        img2 = cv2.cvtColor(np.uint8(np.ones((128, 128, 4)) * 255), cv2.COLOR_RGBA2BGR)
        mean_distance = compute_distance_two(img1=img1, img2=img2)
        self.assertEqual(mean_distance, 255)

        Logger().info("Testing with real image")
        img2 = cv2.imread(os.path.join(self.config.path_data, "test_flower.jpg"))
        img1 = cv2.cvtColor(np.uint8(np.zeros(img2.shape)), cv2.COLOR_RGBA2BGR)
        mean_distance = compute_distance_two(img1=img1, img2=img2)
        self.assertEqual(np.round(mean_distance, 3), 144.259)

    def test_draw_triangle(self):

        image_triangle = draw_triangle(
            image=self.image_base,
            triangle=[(0, 0), (0, 500), (500, 500)],
            color=(100, 0, 100, 125)
        )
        # image_triangle.show()

        self.assertIsInstance(image_triangle, Image.Image)
        image = np.array(image_triangle)
        self.assertEqual(image[0, 0, 0], 100)
        self.assertEqual(image[0, 0, 1], 0)
        self.assertEqual(image[0, 0, 2], 100)
        self.assertEqual(image[0, 0, 3], 125)

    def test_draw_triangles(self):

        image_triangle = self.image_base.copy()
        drw = ImageDraw.Draw(image_triangle, "RGBA")
        drw.polygon([(0, 0), (0, 500), (500, 500)], (100, 0, 100, 125))
        drw.polygon([(500, 0), (0, 500), (500, 500)], (100, 0, 100, 125))
        drw.polygon([(0, 0), (150, 100), (0, 250)], (0, 100, 100, 125))

        image_triangle = draw_triangle(
            image=self.image_base.copy(),
            triangle=[(0, 0), (0, 500), (500, 500)],
            color=(100, 0, 100, 125)
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

        image_triangle.show()
        image_array = np.array(image_triangle)
        self.assertIsInstance(image_triangle, Image.Image)

    def test_blend_images(self):

        img1 = draw_triangle(
            image=self.image_base.copy(),
            triangle=[(0, 0), (0, 500), (500, 500)],
            color=(100, 0, 100, 200)
        )

        img2 = draw_triangle(
            image=self.image_base.copy(),
            triangle=[(500, 0), (0, 500), (500, 500)],
            color=(100, 0, 100, 125)
        )

        img_blend = blend_images(img1=img1, img2=img2)

        # img_blend.show()

        img3 = draw_triangle(
            image=self.image_base.copy(),
            triangle=[(0, 0), (150, 100), (0, 250)],
            color=(0, 100, 100, 125)
        )

        img_blend = blend_images(img1=img_blend, img2=img3)
        # img_blend.show()
        self.assertIsInstance(img_blend, Image.Image)
        for i in range(4):
            self.assertEqual(np.array(img_blend)[0, 250, i], 255)


if __name__ == '__main__':
    unittest.main()
