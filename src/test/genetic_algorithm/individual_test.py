"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 24/11/2020
"""

import numpy as np
import unittest

from src.genetic_algorithm.individual import Individual


class TestIndividual(unittest.TestCase):
    def test_individual(self):
        individual = Individual()
        self.assertIsInstance(individual.individual, np.ndarray)


if __name__ == '__main__':
    unittest.main()
