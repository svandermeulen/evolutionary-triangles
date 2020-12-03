"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 22/05/2020
"""
import numpy as np
import random

from typing import Tuple, Union, Callable

from src.genetic_algorithm.individual import Individual


class Crossover(object):

    def __init__(self, mother: np.ndarray, father: np.ndarray, crossover_rate: float = 0.95):
        self.mother = mother
        self.father = father
        self.crossover_rate = crossover_rate
        self.crossover_methods = [self.crossover_onepoint, self.crossover_twopoint, self.crossover_uniform]

    def apply_crossover(self) -> Union[Tuple, Tuple[Individual, Individual]]:

        if random.uniform(0, 1) > self.crossover_rate:
            return ()

        crossover_method = self.select_crossover_method()
        children = crossover_method()
        if children:
            return Individual(individual=children[0].astype(np.uint16)), \
                   Individual(individual=children[1].astype(np.uint16))
        return children

    def select_crossover_method(self) -> Callable:
        crossover_method_idx = random.randint(0, len(self.crossover_methods) - 1)
        return self.crossover_methods[crossover_method_idx]

    def crossover_onepoint(self, crossover_point: int = None) -> Tuple[np.ndarray, np.ndarray]:

        crossover_point = crossover_point if crossover_point is not None else np.random.randint(0, len(self.mother))
        child_one = np.vstack((self.mother[:crossover_point, :], self.father[crossover_point:, :]))
        child_two = np.vstack((self.father[:crossover_point, :], self.mother[crossover_point:, :]))

        return child_one, child_two

    def crossover_twopoint(self, crossover_points: list = None) -> Tuple[np.ndarray, np.ndarray]:

        crossover_points = crossover_points if crossover_points is not None else sorted(
            random.sample(
                range(self.mother.shape[0]), 2)
        )

        child_one = np.vstack(
            (
                self.mother[:crossover_points[0]],
                self.father[crossover_points[0]:crossover_points[1]],
                self.mother[crossover_points[1]:]
            )
        )
        child_two = np.vstack(
            (
                self.father[:crossover_points[0]],
                self.mother[crossover_points[0]:crossover_points[1]],
                self.father[crossover_points[1]:]
            )
        )

        return child_one, child_two

    def crossover_uniform(self, crossover_points: list = None) -> Tuple[np.ndarray, np.ndarray]:

        crossover_points = crossover_points if crossover_points is not None else sorted(
            random.sample(range(self.mother.shape[0]), self.mother.shape[0] // 2))
        crossover_points_father = [i for i in range(self.mother.shape[0]) if i not in crossover_points]

        child_one = np.zeros(self.mother.shape)
        child_two = np.zeros(self.father.shape)
        child_one[crossover_points] = self.mother[crossover_points]
        child_one[crossover_points_father] = self.father[crossover_points_father]

        child_two[crossover_points] = self.father[crossover_points]
        child_two[crossover_points_father] = self.mother[crossover_points_father]

        return child_one, child_two


def main():
    pass


if __name__ == "__main__":
    main()
