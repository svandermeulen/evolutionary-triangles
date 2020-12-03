"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 24/11/2020
"""
import random

import numpy as np
import unittest

from src.config import Config
from src.genetic_algorithm.individual import Individual
from src.genetic_algorithm.mutation import mutate_array, mutate_individual


class TestMutation(unittest.TestCase):

    def setUp(self) -> None:
        self.mother = np.ones((4, 2))
        self.config = Config()

        random.seed(1234)

    def test_mutate_array(self):

        mother_mutated = mutate_array(array=self.mother, mutation_rate=0.5)
        self.assertTrue(np.equal(self.mother.shape, mother_mutated.shape).all())

    def test_mutate_individual(self):

        individual = Individual(config=self.config)
        individual_mutated = mutate_individual(individual=individual)

        self.assertIsInstance(individual_mutated, Individual)
        self.assertFalse(np.equal(individual.individual, individual_mutated.individual).all())

        self.config.triangulation_method = "non_overlapping"
        individual = Individual(config=self.config)
        individual_mutated = mutate_individual(individual=individual, yidx=1, coloridx=2)
        self.assertEqual(individual_mutated.individual.shape[1], 6)


if __name__ == '__main__':
    unittest.main()