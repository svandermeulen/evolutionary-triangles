"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""
import os

from abc import ABCMeta

import cv2

from src.utils.image_tools import resize_image


class SingletonABCMeta(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=SingletonABCMeta):

    def __init__(
            self,
            path_image_ref: str = "",
            n_triangles: int = 25,
            n_population: int = 50,
            n_generations: int = 10,
            mutation_rate: float = 0.05
    ):

        self.path_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path_data = os.path.join(self.path_home, "data")
        self.path_output = os.path.join(self.path_home, "output")
        self.path_logs = os.path.join(self.path_home, "logs")

        for p in [self.path_data, self.path_output]:
            self.create_folder(path_folder=p)

        path_image_ref = path_image_ref if path_image_ref else os.path.join(self.path_data, "test", "test_flower.jpg")
        image_ref = cv2.imread(path_image_ref)
        self.image_ref = resize_image(image_ref)

        self.n_triangles = n_triangles
        self.n_population = n_population
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.survival_rate = 0.25
        self.triangulation_method = "non_overlapping"  # overlapping
        self.side_by_side = True  # Indicates whether intermediate triangle images are shown next to the reference image
        self.fps = 20

    @staticmethod
    def create_folder(path_folder: str):
        if not os.path.isdir(path_folder):
            os.mkdir(path_folder)


def main():
    pass


if __name__ == "__main__":
    main()
