"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 03/12/2020
"""
from typing import Tuple, List

import numpy as np
import random


def select_random_individual(population: list) -> int:
    return random.randint(0, len(population) - 1)


def tournament_selection(population: List) -> int:
    candidate_one = select_random_individual(population=population)
    candidate_two = select_random_individual(population=population)
    while candidate_one == candidate_two:
        candidate_two = select_random_individual(population=population)

    if population[candidate_one].fitness > population[candidate_two].fitness:
        return candidate_one
    return candidate_two


def biased_random_selection(population: List) -> int:
    fitness_sum = sum([individual.fitness for individual in population])
    proportions = [fitness_sum / individual.fitness for individual in population]
    proportions_sum = sum(proportions)
    proportions_norm = [p / proportions_sum for p in proportions]
    proportions_cummulative = np.cumsum(proportions_norm)
    select_value = random.uniform(0, 1)
    return int(np.min(np.where(proportions_cummulative > select_value)))


def get_parent(population: List) -> int:

    assert all([hasattr(individual, "fitness") for individual in population]), \
        "Not all individuals have the attribute 'fitness'"
    if random.uniform(0, 1) > 0.5:
        return tournament_selection(population=population)
    return biased_random_selection(population=population)


def select_parents(population: List) -> Tuple[int, int]:

    mother = get_parent(population=population)
    father = get_parent(population=population)
    while mother == father:
        father = get_parent(population=population)  # avoid crossover with oneself
    return mother, father


def main():
    pass


if __name__ == "__main__":
    main()
