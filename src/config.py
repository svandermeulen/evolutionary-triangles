"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""
import os

from abc import ABCMeta
from datetime import datetime


class SingletonABCMeta(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=SingletonABCMeta):

    def __init__(
            self,
            n_triangles: int = 25,
            n_population: int = 100,
            n_generations: int = 10,
            mutation_rate: float = 0.05,
            crossover_rate: float = 0.95,
            triangulation_method: str = "overlapping"
    ):

        self.path_home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path_data = os.path.join(self.path_home, "data")
        self.path_output = os.path.join(self.path_home, "output")
        self.path_static = os.path.join(self.path_home, "static")

        for p in [self.path_data, self.path_output]:
            self.create_folder(path_folder=p)

        self.n_triangles = n_triangles
        self.n_population = n_population
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.triangulation_method = triangulation_method
        self.side_by_side = True  # Indicates whether intermediate triangle images are shown next to the reference image
        self.fps = 20

    @staticmethod
    def create_folder(path_folder: str):
        if not os.path.isdir(path_folder):
            os.makedirs(path_folder)


def main():

    config = Config(n_triangles=20)
    config_two = Config()

    assert config.n_triangles == config_two.nt


if __name__ == "__main__":
    main()
