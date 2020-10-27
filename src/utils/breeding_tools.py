"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 22/05/2020
"""
import random
from copy import copy
from typing import List, Tuple

import numpy as np

from src.config import Config
from src.utils.logger import Logger


def get_top_pairs(idx: List) -> List[Tuple]:
    return [(id1, id2) for id1, id2 in zip(idx[::2], idx[1::2])]


def get_random_pairs(idx: List):
    random.shuffle(idx)
    return [(i1, i2) for i1, i2 in zip(idx[::2], idx[1::2])]


def crossover(mother: np.ndarray, father: np.ndarray, crossover_point: int = None) -> (np.ndarray, np.ndarray):

    crossover_point = crossover_point if crossover_point is not None else np.random.randint(0, len(mother))
    child_one = np.vstack((mother[:crossover_point, :], father[crossover_point:, :]))
    child_two = np.vstack((father[:crossover_point, :], mother[crossover_point:, :]))

    return child_one, child_two


def mutate_array(array: np.ndarray, min_value: int = 0, max_value: int = 256, mutation_rate: float = 0.95) -> np.ndarray:

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


def mutate_children(children: np.ndarray, xmax: int, ymax: int) -> np.ndarray:
    coordinates_x = mutate_array(children[:, :3, :], max_value=xmax)
    coordinates_y = mutate_array(children[:, 3:6, :], max_value=ymax)
    colors = mutate_array(children[:, 6:, :], max_value=256)

    return np.hstack((coordinates_x, coordinates_y, colors))


def cross_breed_population(population: np.ndarray, config: Config, width: int, height: int) -> np.ndarray:

    # Crossbreed best performing individuals to obtain new offspring
    if config.pairing_method == "random":
        pairs = get_random_pairs(idx=list(range(population.shape[-1])))
    else:
        pairs = get_top_pairs(idx=list(range(population.shape[-1])))
    children = np.zeros((config.n_triangles, 10, config.n_population // 2))
    for pair in pairs:
        try:
            children[:, :, pair[0]], children[:, :, pair[1]] = crossover(
                mother=population[:, :, pair[0]],
                father=population[:, :, pair[1]]
            )
        except ValueError as e:
            Logger().error(f"{e}")
    children_mutated = mutate_children(children=children, xmax=width, ymax=height)

    # Combine children with best performing individuals
    return np.uint16(np.dstack((population, children_mutated)))


def main():
    pass


if __name__ == "__main__":

    main()