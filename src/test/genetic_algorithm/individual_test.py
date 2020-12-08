"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 24/11/2020
"""
import os

import cv2
import numpy as np
import unittest

from src.config import Config
from src.genetic_algorithm.individual import Individual


class TestIndividual(unittest.TestCase):

    def setUp(self) -> None:

        self.image = cv2.imread(os.path.join(Config().path_data, "test", "test_flower.jpg"))

    def test_individual(self):
        individual = Individual(image=self.image)
        self.assertIsInstance(individual.individual, np.ndarray)


if __name__ == '__main__':
    unittest.main()
