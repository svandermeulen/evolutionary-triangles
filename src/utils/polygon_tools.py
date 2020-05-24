"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""

import numpy as np

from src.utils.config import Config

N_POPULATION = Config().n_population
N_TRIANGLES = Config().n_triangles


def get_random_coordinate(xmax: int, ymax: int) -> tuple:
    return np.random.randint(0, xmax), np.random.randint(0, ymax)


def get_random_rgba_color() -> tuple:
    return tuple([np.random.randint(0, 256) for _ in range(4)])


def get_random_polyhon(xmax: int, ymax: int, degree: int = 3) -> list:
    return [get_random_coordinate(xmax=xmax, ymax=ymax) for _ in range(degree)]


def generate_random_triangles(xmax: int, ymax: int) -> np.ndarray:

    coordinates_x = np.random.randint(0, xmax, N_POPULATION * N_TRIANGLES * 3).reshape((N_TRIANGLES, 3, N_POPULATION))
    coordinates_y = np.random.randint(0, ymax, N_POPULATION * N_TRIANGLES * 3).reshape((N_TRIANGLES, 3, N_POPULATION))
    colors = np.random.randint(0, 256, N_POPULATION * N_TRIANGLES * 4).reshape((N_TRIANGLES, 4, N_POPULATION))

    return np.hstack((coordinates_x, coordinates_y, colors))


def main():
    pass


if __name__ == "__main__":
    main()
