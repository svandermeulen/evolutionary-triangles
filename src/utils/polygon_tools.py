"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""
import itertools
import numpy as np
import random

from scipy.spatial.qhull import Delaunay


def get_random_coordinate(xmax: int, ymax: int) -> tuple:
    return np.random.randint(0, xmax), np.random.randint(0, ymax)


def get_random_rgba_color() -> tuple:
    return tuple([np.random.randint(0, 256) for _ in range(4)])


def get_random_polyhon(xmax: int, ymax: int, degree: int = 3) -> list:
    return [get_random_coordinate(xmax=xmax, ymax=ymax) for _ in range(degree)]


def generate_random_colors(n_colors: int) -> np.ndarray:
    """
    Generate a sequence of n random colors.
    Returns a array with length n and width 4 corresponding to the R, G, B and alpha channels
    """
    return np.random.randint(0, 256, n_colors * 4).reshape((n_colors, 4))


def generate_random_triangles(xmax: int, ymax: int, n_triangles: int) -> np.ndarray:
    coordinates_x = np.random.randint(0, xmax, n_triangles * 3).reshape((n_triangles, 3))
    coordinates_y = np.random.randint(0, ymax, n_triangles * 3).reshape((n_triangles, 3))
    colors = generate_random_colors(n_colors=n_triangles)

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

    points = random.sample(list(itertools.product(range(1, xmax - 1), range(1, ymax - 1))), n_points)
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


def generate_delaunay_triangles(xmax: int, ymax: int, n_triangles: int = 100) -> np.ndarray:
    points = generate_uniform_random_points(xmax=xmax, ymax=ymax, n_points=n_triangles)
    coordinates_x = points[:, 0]
    coordinates_y = points[:, 1]
    colors = generate_random_colors(n_colors=len(coordinates_x))

    return np.hstack((
        coordinates_x.reshape((n_triangles, 1)),
        coordinates_y.reshape((n_triangles, 1)),
        colors)
    )


def convert_delaunay_points(points: np.ndarray) -> np.ndarray:
    tri = Delaunay(points)
    return np.array([[points[i] for i in triangle] for triangle in tri.vertices])


def convert_points_to_triangles(points: np.ndarray) -> np.ndarray:
    triangles = convert_delaunay_points(points=points[:, :2])  # ignore the colors
    return np.hstack([triangles[:, :, 0], triangles[:, :, 1]])  # stack x and y coordinates horizontally


def main():
    pass


if __name__ == "__main__":
    main()
