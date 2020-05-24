"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 22/05/2020
"""
import random
from copy import copy
from typing import List

import numpy as np

from src.utils.config import Config


def get_random_pairs(number_list: List):

    idx = list(range(len(number_list)))
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

    points_to_mutate = mutation_coords > Config().mutation_rate

    if not points_to_mutate.any():
        return array

    array_new = copy(array)
    array_new[points_to_mutate] = np.random.randint(
        low=min_value,
        high=max_value,
        size=np.sum(points_to_mutate)
    )

    return array_new


def main():
    mum = np.ones((4, 2))
    father = mum * 2

    child_one, child_two = crossover(mother=mum, father=father, crossover_point=2)

    assert np.equal(child_one[:, 0], np.array([1, 1, 2, 2])).all()
    assert np.equal(child_two[:, 1], np.array([2, 2, 1, 1])).all()

    mum_mutated = mutate_array(array=mum, mutation_rate=0.5)

    assert np.equal(mum.shape, mum_mutated.shape).all()



if __name__ == "__main__":

    main()