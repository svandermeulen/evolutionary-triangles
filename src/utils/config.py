"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 23/05/2020
"""

import os


class Config:

    def __init__(self):
        self.path_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.path_data = os.path.join(self.path_home, "data")
        self.path_output = os.path.join(self.path_home, "output")

        for p in [self.path_data, self.path_output]:
            self.create_folder(path_folder=p)

        self.n_triangles = 500
        self.n_population = 100
        self.generations = 100
        self.mutation_rate = 0.90

    @staticmethod
    def create_folder(path_folder: str):
        if not os.path.isdir(path_folder):
            os.mkdir(path_folder)


def main():
    pass


if __name__ == "__main__":
    main()
