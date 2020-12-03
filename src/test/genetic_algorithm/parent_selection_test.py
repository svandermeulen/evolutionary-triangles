"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 03/12/2020
"""
import random
import unittest
from typing import Tuple

from src.genetic_algorithm.individual import Individual
from src.genetic_algorithm.parent_selection import select_random_individual, tournament_selection, \
    biased_random_selection, get_parent, select_parents


class TestParentSelection(unittest.TestCase):

    def setUp(self) -> None:
        self.population = [Individual() for _ in range(10)]

    def test_select_random_individual(self):
        individual = select_random_individual(population=self.population)
        self.assertIsInstance(individual, int)

    def test_tournament_selection(self):
        parent = tournament_selection(population=self.population)
        self.assertIsInstance(parent, int)

    def test_biased_random_selection(self):
        parent = biased_random_selection(population=self.population)
        self.assertIsInstance(parent, int)

    def test_get_parent(self):
        parent = get_parent(population=self.population)
        self.assertIsInstance(parent, int)

    def test_select_parents(self):
        parents = select_parents(population=self.population)
        self.assertIsInstance(parents, tuple)
        self.assertTrue(all([isinstance(parent, int) for parent in parents]))

        with self.assertRaises(AssertionError) as context:
            select_parents(population=["not", "a", "list", "of", "Individuals"])

        self.assertTrue("Not all individuals have the attribute 'fitness'" == context.exception.args[0])


if __name__ == '__main__':
    unittest.main()
