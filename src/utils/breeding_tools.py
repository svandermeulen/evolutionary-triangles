"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 22/05/2020
"""

import numpy as np
import random

from copy import copy
from itertools import combinations, product, permutations
from typing import List, Tuple

from src.config import Config
from src.utils.logger import Logger


def get_top_pairs(idx: List) -> List[Tuple]:
    # return [(id1, id2) for id1, id2 in zip(idx[::2], idx[1::2])]
    return sorted(list(combinations(idx, 2)), key=lambda x: x[1])


def get_random_pairs(idx: List) -> List[Tuple]:
    random.shuffle(idx)
    return [(i1, i2) for i1, i2 in zip(idx[::2], idx[1::2])]


def crossover(mother: np.ndarray, father: np.ndarray, crossover_point: int = None) -> (np.ndarray, np.ndarray):
    crossover_point = crossover_point if crossover_point is not None else np.random.randint(0, len(mother))
    child_one = np.vstack((mother[:crossover_point, :], father[crossover_point:, :]))
    child_two = np.vstack((father[:crossover_point, :], mother[crossover_point:, :]))

    return child_one, child_two


def mutate_array(array: np.ndarray, min_value: int = 0, max_value: int = 256,
                 mutation_rate: float = 0.95) -> np.ndarray:
    mutation_coords = np.random.uniform(
        low=0,
        high=1,
        size=np.prod(array.shape)
    ).reshape(array.shape)

    points_to_mutate = mutation_coords > mutation_rate

    if not points_to_mutate.any():
        return array

    array_new = copy(array)
    array_new[points_to_mutate] = np.random.randint(
        low=min_value,
        high=max_value,
        size=np.sum(points_to_mutate)
    )

    return array_new


def mutate_children(children: np.ndarray, xmax: int, ymax: int, yidx: int = 3, coloridx: int = 6) -> np.ndarray:
    coordinates_x = mutate_array(children[:, :yidx, :], max_value=xmax)
    coordinates_y = mutate_array(children[:, yidx:coloridx, :], max_value=ymax)
    colors = mutate_array(children[:, coloridx:, :], max_value=256)

    return np.hstack((coordinates_x, coordinates_y, colors))


def cross_breed_population(population: np.ndarray, config: Config, width: int, height: int) -> np.ndarray:
    # Crossbreed best performing individuals to obtain new offspring

    n_children = (config.n_population - population.shape[-1])
    pairing_indices = list(range(population.shape[-1]))  # * repetitions

    pairs = list(permutations(pairing_indices, 2))
    if len(pairs) < n_children:
        difference = n_children - len(pairs)
        pairs += [pairs[i] for i in np.random.randint(0, len(pairs), difference)]

    if config.pairing_method == "random":
        random.shuffle(pairs)
    elif config.pairing_method == "best":
        pairs = sorted(pairs, key=lambda x: sum(x))
    else:
        Logger().error(f"Invalid pairing_method given: '{config.pairing_method}'. "
                       f"Choose from 'random' or 'best'")
        raise ValueError

    pairs = pairs[:np.ceil(n_children/2).astype(int)]

    if config.triangulation_method != "overlapping":
        children = np.zeros((population.shape[0], 6, n_children))
    else:
        children = np.zeros((population.shape[0], 10, n_children))

    for i, pair in enumerate(pairs):

        if (i*2) + 1 >= n_children:
            children[:, :, i*2], _ = crossover(
                mother=population[:, :, pair[0]],
                father=population[:, :, pair[1]]
            )
        else:
            children[:, :, i*2], children[:, :, (i*2)+1] = crossover(
                mother=population[:, :, pair[0]],
                father=population[:, :, pair[1]]
            )

    if config.triangulation_method != "overlapping":
        children_mutated = mutate_children(children=children, xmax=width, ymax=height, yidx=1, coloridx=2)
    else:
        children_mutated = mutate_children(children=children, xmax=width, ymax=height)

    # Combine children with best performing individuals
    return np.uint16(np.dstack((population, children_mutated)))


def main():
    pass


if __name__ == "__main__":
    main()
