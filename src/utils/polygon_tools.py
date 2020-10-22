"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""

import numpy as np

from src.config import Config

N_POPULATION = Config().n_population
N_TRIANGLES = Config().n_triangles


def get_random_coordinate(xmax: int, ymax: int) -> tuple:
    return np.random.randint(0, xmax), np.random.randint(0, ymax)


def get_random_rgba_color() -> tuple:
    return tuple([np.random.randint(0, 256) for _ in range(4)])


def get_random_polyhon(xmax: int, ymax: int, degree: int = 3) -> list:
    return [get_random_coordinate(xmax=xmax, ymax=ymax) for _ in range(degree)]


def generate_random_triangles(xmax: int, ymax: int, n_triangles: int = N_TRIANGLES,
                              n_population: int = N_POPULATION) -> np.ndarray:

    coordinates_x = np.random.randint(0, xmax, n_population * n_triangles * 3).reshape((n_triangles, 3, n_population))
    coordinates_y = np.random.randint(0, ymax, n_population * n_triangles * 3).reshape((n_triangles, 3, n_population))
    colors = np.random.randint(0, 256, n_population * n_triangles * 4).reshape((n_triangles, 4, n_population))

    return np.hstack((coordinates_x, coordinates_y, colors))


def main():
    pass


if __name__ == "__main__":
    main()
