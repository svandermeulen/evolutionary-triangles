"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 24/11/2020
"""

import numpy as np

from copy import copy, deepcopy

from src.genetic_algorithm.individual import Individual


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

    individual_mutated = deepcopy(individual)

    coordinates_x = mutate_array(individual_mutated.individual[:, :yidx], max_value=individual_mutated.width)
    coordinates_y = mutate_array(individual_mutated.individual[:, yidx:coloridx], max_value=individual_mutated.height)
    colors = mutate_array(individual_mutated.individual[:, coloridx:], max_value=256)

    individual_mutated.individual = np.hstack((coordinates_x, coordinates_y, colors))

    # Recompute fitness
    individual_mutated.get_fitness()

    return individual_mutated


def main():
    pass


if __name__ == "__main__":
    main()
