import numpy as np
import unittest

from src.utils.breeding_tools import mutate_array, crossover


class TestBreedingTools(unittest.TestCase):

    def setUp(self) -> None:

        self.mum = np.ones((4, 2))
        self.father = self.mum * 2

    def test_crossover(self):

        child_one, child_two = crossover(mother=self.mum, father=self.father, crossover_point=2)

        self.assertTrue(np.equal(child_one[:, 0], np.array([1, 1, 2, 2])).all())
        self.assertTrue(np.equal(child_two[:, 1], np.array([2, 2, 1, 1])).all())

    def test_mutate_array(self):

        mum_mutated = mutate_array(array=self.mum, mutation_rate=0.5)
        self.assertTrue(np.equal(self.mum.shape, mum_mutated.shape).all())


if __name__ == '__main__':
    unittest.main()
