"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""
import itertools
import numpy as np
import pandas as pd
import random

from scipy.spatial.qhull import Delaunay

from src.config import Config

N_POPULATION = Config().n_population
N_TRIANGLES = Config().n_triangles


def get_random_coordinate(xmax: int, ymax: int) -> tuple:
    return np.random.randint(0, xmax), np.random.randint(0, ymax)


def get_random_rgba_color() -> tuple:
    return tuple([np.random.randint(0, 256) for _ in range(4)])


def get_random_polyhon(xmax: int, ymax: int, degree: int = 3) -> list:
    return [get_random_coordinate(xmax=xmax, ymax=ymax) for _ in range(degree)]


def generate_random_colors(n_population: int, n_triangles: int) -> np.ndarray:
    return np.random.randint(0, 256, n_population * n_triangles * 4).reshape((n_triangles, 4, n_population))


def generate_random_triangles(xmax: int, ymax: int, n_triangles: int = N_TRIANGLES,
                              n_population: int = N_POPULATION) -> np.ndarray:
    coordinates_x = np.random.randint(0, xmax, n_population * n_triangles * 3).reshape((n_triangles, 3, n_population))
    coordinates_y = np.random.randint(0, ymax, n_population * n_triangles * 3).reshape((n_triangles, 3, n_population))
    colors = generate_random_colors(n_population=n_population, n_triangles=n_triangles)

    return np.hstack((coordinates_x, coordinates_y, colors))


def generate_edge_points(xmax: int, ymax: int, length_scale: int = 200,
                         n_horizontal_points: int = None,
                         n_vertical_points: int = None):
    """
    Returns points around the edge of an image.
    :param length_scale: how far to space out the points if no
                         fixed number of points is given
    :param n_horizontal_points: number of points on the horizonal edge.
                                Leave as None to use lengthscale to determine
                                the value
    :param n_vertical_points: number of points on the horizonal edge.
                                Leave as None to use lengthscale to determine
                                the value
    :return: array of coordinates
    """
    if n_horizontal_points is None:
        n_horizontal_points = int(xmax / length_scale)

    if n_vertical_points is None:
        n_vertical_points = int(ymax / length_scale)

    delta_x = xmax / n_horizontal_points
    delta_y = ymax / n_vertical_points

    return np.array(
        [[0, 0], [xmax, 0], [0, ymax], [xmax, ymax]]
        + [[delta_x * i, 0] for i in range(1, n_horizontal_points)]
        + [[delta_x * i, ymax] for i in range(1, n_horizontal_points)]
        + [[0, delta_y * i] for i in range(1, n_vertical_points)]
        + [[xmax, delta_y * i] for i in range(1, n_vertical_points)]
    )


def generate_uniform_random_points(xmax: int, ymax: int, n_points=100):
    """
    Generates a set of uniformly distributed points over the area of image
    :param n_points: int number of points to generate
    :return: array of points
    """

    points = random.sample(list(itertools.product(range(1, xmax-1), range(1, ymax-1))), n_points)
    points = np.array(points)
    # points_edge = generate_edge_points(xmax=xmax, ymax=ymax, n_horizontal_points=5, n_vertical_points=5)
    # points = np.concatenate([points, points_edge]).astype(int)
    return points


def get_triangles(n_triangles: int, xmax: int, ymax: int) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Delaunay_triangulation

    n_points = int(round((n_triangles + 5) / 2, 0)) + 4  # Add 4 to enhance chances
    points = generate_uniform_random_points(xmax=xmax, ymax=ymax, n_points=n_points)
    triangles = convert_delaunay_points(points=points)
    if len(triangles) != n_triangles:
        print(len(triangles))
        triangles = get_triangles(n_triangles=n_triangles, xmax=xmax, ymax=ymax)
    return triangles


def generate_delaunay_triangles(xmax: int, ymax: int, n_points: int = 100, n_population: int = N_POPULATION) -> np.ndarray:

    coordinates_x = np.zeros((n_points, n_population)).astype(int)
    coordinates_y = np.zeros((n_points, n_population)).astype(int)

    for p in range(n_population):
        points = generate_uniform_random_points(xmax=xmax, ymax=ymax, n_points=n_points)
        coordinates_x[:, p] = points[:, 0]
        coordinates_y[:, p] = points[:, 1]

    colors = generate_random_colors(n_population=n_population, n_triangles=len(coordinates_x))

    return np.hstack((
        coordinates_x.reshape((n_points, 1, n_population)),
        coordinates_y.reshape((n_points, 1, n_population)),
        colors))


def convert_delaunay_points(points: np.ndarray) -> np.ndarray:
    tri = Delaunay(points)
    return np.array([[points[i] for i in triangle] for triangle in tri.vertices])


def convert_population_to_triangles(population: np.ndarray) -> np.ndarray:

    triangles = convert_delaunay_points(points=population[:, :2])
    triangles = np.hstack([triangles[:, :, 0], triangles[:, :, 1]])
    df_triangles = pd.DataFrame(triangles, columns=["x1", "x2", "x3", "y1", "y2", "y3"])
    df_population = pd.DataFrame(population, columns=["x", "y", "c1", "c2", "c3", "c4"])

    for i in range(3):
        if i == 0:
            df_m = df_triangles.merge(df_population, left_on=["x1", "y1"], right_on=["x", "y"]).drop(columns=["x", "y"])
        else:
            df_m = df_m.merge(df_population, left_on=[f"x{i + 1}", f"y{i + 1}"], right_on=["x", "y"]).drop(
                columns=["x", "y"])

    df_m[[c for c in df_m if c.startswith("c")]] = df_m[[c for c in df_m if c.startswith("c")]].pow(2)
    for i in range(4):
        df_m[f"c{i + 1}"] = df_m[[c for c in df_m if c.startswith(f"c{i + 1}")]].mean(axis=1).pow(0.5).astype(int)

    return df_m.drop(columns=[c for c in df_m if "_" in c]).values


def main():
    pass


if __name__ == "__main__":
    main()
