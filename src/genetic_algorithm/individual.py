"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 23/11/2020
"""
import cv2
import numpy as np
import os
import pandas as pd

from src.config import Config
from src.utils.image_tools import resize_image, generate_triangle_image, compute_distance, convert_pil_to_array
from src.utils.polygon_tools import generate_delaunay_triangles, generate_random_triangles, \
    convert_points_to_triangles


class Individual(object):

    def __init__(self, config: Config = Config(), individual: np.ndarray = None):

        self.config = config
        self.image_ref = config.image_ref
        self.height, self.width, self.depth = self.image_ref.shape

        spawner = generate_delaunay_triangles if config.triangulation_method == "non_overlapping" \
            else generate_random_triangles

        self.individual = individual if individual is not None else spawner(
            xmax=self.width,
            ymax=self.height,
            n_triangles=config.n_triangles
        )
        self.fitness = self.get_fitness()

    def get_fitness(self):

        individual = self.individual
        if self.config.triangulation_method == "non_overlapping":
            individual = self.convert_points_to_triangles()

        image_triangles = generate_triangle_image(
            width=self.width,
            height=self.height,
            triangles=individual
        )
        return compute_distance(img1=self.image_ref, img2=convert_pil_to_array(image_triangles))

    def convert_points_to_triangles(self):

        triangles = convert_points_to_triangles(points=self.individual)
        return self._average_point_colors(triangles=triangles)

    def _average_point_colors(self, triangles: np.ndarray):

        df_triangles = pd.DataFrame(triangles, columns=["x1", "x2", "x3", "y1", "y2", "y3"])
        df_points = pd.DataFrame(self.individual, columns=["x", "y", "c1", "c2", "c3", "c4"])

        df_m = df_triangles.merge(df_points, left_on=["x1", "y1"], right_on=["x", "y"]).drop(columns=["x", "y"])
        for i in range(1, 3):
            df_m = df_m.merge(df_points, left_on=[f"x{i + 1}", f"y{i + 1}"], right_on=["x", "y"]).drop(
                columns=["x", "y"])

        df_m[[c for c in df_m if c.startswith("c")]] = df_m[[c for c in df_m if c.startswith("c")]].pow(2)
        for i in range(4):
            df_m[f"c{i + 1}"] = df_m[[c for c in df_m if c.startswith(f"c{i + 1}")]].mean(axis=1).pow(0.5).astype(int)

        return df_m.drop(columns=[c for c in df_m if "_" in c]).values


def main():
    pass


if __name__ == "__main__":
    main()
