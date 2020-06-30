"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 26/05/2020
"""

import cv2
import numpy as np
import os
import unittest

from src.config import Config
from src.utils.image_tools import compute_distance, compute_distance_two


class TestImageTools(unittest.TestCase):

    def setUp(self):

        self.config = Config()

    def test_compute_distance(self):

        img1 = cv2.cvtColor(np.uint8(np.zeros((128, 128, 4))), cv2.COLOR_RGBA2BGR)
        img2 = cv2.cvtColor(np.uint8(np.ones((128, 128, 4)) * 255), cv2.COLOR_RGBA2BGR)
        mean_distance = compute_distance(img1=img1, img2=img2)
        self.assertEqual(mean_distance, 255)

        print("Testing with real image")
        img2 = cv2.imread(os.path.join(self.config.path_data, "test_flower.jpg"))
        img1 = cv2.cvtColor(np.uint8(np.zeros(img2.shape)), cv2.COLOR_RGBA2BGR)
        mean_distance = compute_distance(img1=img1, img2=img2)
        self.assertEqual(np.round(mean_distance, 3), 144.259)

    def test_compute_distance_two(self):

        img1 = cv2.cvtColor(np.uint8(np.zeros((128, 128, 4))), cv2.COLOR_RGBA2BGR)
        img2 = cv2.cvtColor(np.uint8(np.ones((128, 128, 4)) * 255), cv2.COLOR_RGBA2BGR)
        mean_distance = compute_distance_two(img1=img1, img2=img2)
        self.assertEqual(mean_distance, 255)

        print("Testing with real image")
        img2 = cv2.imread(os.path.join(self.config.path_data, "test_flower.jpg"))
        img1 = cv2.cvtColor(np.uint8(np.zeros(img2.shape)), cv2.COLOR_RGBA2BGR)
        mean_distance = compute_distance_two(img1=img1, img2=img2)
        self.assertEqual(np.round(mean_distance, 3), 144.259)


if __name__ == '__main__':
    unittest.main()
