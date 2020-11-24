"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 22/05/2020
"""

import numpy as np
import random

from copy import copy
from itertools import combinations, permutations
from typing import List, Tuple

from src.config import Config
from src.genetic_algorithm.individual import Individual
from src.utils.logger import Logger


def get_top_pairs(idx: List) -> List[Tuple]:
    return sorted(list(combinations(idx, 2)), key=lambda x: x[1])


def get_random_pairs(idx: List) -> List[Tuple]:
    random.shuffle(idx)
    return [(i1, i2) for i1, i2 in zip(idx[::2], idx[1::2])]


def crossover(mother: np.ndarray, father: np.ndarray, crossover_point: int = None) -> Tuple[Individual, Individual]:
    crossover_point = crossover_point if crossover_point is not None else np.random.randint(0, len(mother))
    child_one = np.vstack((mother[:crossover_point, :], father[crossover_point:, :]))
    child_two = np.vstack((father[:crossover_point, :], mother[crossover_point:, :]))

    return Individual(individual=child_one), Individual(individual=child_two)


def mutate_array(array: np.ndarray, min_value: int = 0, max_value: int = 256,
                 mutation_rate: float = 0.05) -> np.ndarray:
    mutation_coords = np.random.uniform(
        low=0,
        high=1,
        size=np.prod(array.shape)
    ).reshape(array.shape)

    points_to_mutate = mutation_coords > 1 - mutation_rate

    if not points_to_mutate.any():
        return array

    array_new = copy(array)
    array_new[points_to_mutate] = np.random.randint(
        low=min_value,
        high=max_value,
        size=np.sum(points_to_mutate)
    )

    return array_new


def mutate_individual(individual: Individual, yidx: int = 3, coloridx: int = 6) -> Individual:

    coordinates_x = mutate_array(individual.individual[:, :yidx], max_value=individual.width)
    coordinates_y = mutate_array(individual.individual[:, yidx:coloridx], max_value=individual.height)
    colors = mutate_array(individual.individual[:, coloridx:], max_value=256)

    return Individual(individual=np.hstack((coordinates_x, coordinates_y, colors)))


def cross_breed_population(pair: tuple, population: list) -> Tuple[Individual]:

    # Crossbreed individuals to obtain new offspring

    children = crossover(
            mother=population[pair[0]].individual,
            father=population[pair[1]].individual
        )

    return children


def main():
    pass


if __name__ == "__main__":
    main()
