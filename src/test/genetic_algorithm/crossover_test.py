import random

import numpy as np
import unittest

from src.genetic_algorithm.crossover import Crossover
from src.genetic_algorithm.individual import Individual


class TestCrossover(unittest.TestCase):

    def setUp(self) -> None:

        random.seed(1234)
        self.mother = Individual().individual
        self.father = Individual().individual
        self.crossover = Crossover(mother=self.mother, father=self.father)

    def test_crossover_onepoint(self):
        child_one, child_two = self.crossover.crossover_onepoint(crossover_point=2)

        self.assertTrue(np.equal(child_one[:2, :], self.mother[:2, :]).all())
        self.assertTrue(np.equal(child_one[2:, :], self.father[2:, :]).all())
        self.assertTrue(np.equal(child_two[2:, :], self.mother[2:, :]).all())
        self.assertTrue(np.equal(child_two[:2, :], self.father[:2, :]).all())

        child_one, child_two = self.crossover.crossover_onepoint()
        self.assertIsInstance(child_one, np.ndarray)
        self.assertIsInstance(child_two, np.ndarray)

    def test_crossover_twopoint(self):

        child_one, child_two = self.crossover.crossover_twopoint(crossover_points=[10, 20])

        self.assertTrue(np.equal(child_one[:10, :], self.mother[:10, :]).all())
        self.assertTrue(np.equal(child_one[10:20, :], self.father[10:20, :]).all())
        self.assertTrue(np.equal(child_one[20:, :], self.mother[20:, :]).all())

        child_one, child_two = self.crossover.crossover_twopoint()
        self.assertIsInstance(child_one, np.ndarray)
        self.assertIsInstance(child_two, np.ndarray)

    def test_crossover_uniform(self):

        crossover_points = random.sample(range(self.mother.shape[0]), self.mother.shape[0] // 2)
        child_one, child_two = self.crossover.crossover_uniform(crossover_points=crossover_points)
        self.assertTrue(np.equal(child_one[crossover_points, :], self.mother[crossover_points, :]).all())
        self.assertTrue(np.equal(child_two[crossover_points, :], self.father[crossover_points, :]).all())

        child_one, child_two = self.crossover.crossover_uniform()
        self.assertIsInstance(child_one, np.ndarray)
        self.assertIsInstance(child_two, np.ndarray)


if __name__ == '__main__':
    unittest.main()
