"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 22/10/2020
"""

import argparse
import os

from src.utils.logger import Logger


def parse_args(args: list) -> dict:
    parser = argparse.ArgumentParser(
        prog="Evolutionary Triangles",
        description='Running evolutionary triangles'
    )

    parser.add_argument(
        "-f", "--file", dest="file_path", type=str, required=True,
        help="string specifying the path to the image that needs to be drawn"
    )
    parser.add_argument(
        "-np", dest="n_population", type=int, required=False, default=100,
        help="integer specifying the number of individuals comprising the population"
    )
    parser.add_argument(
        "-nt", dest="n_triangles", type=int, required=False, default=100,
        help="integer specifying the number of triangles drawn to create the image"
    )
    parser.add_argument(
        "-ng", dest="n_generations", type=int, required=False, default=50,
        help="integer specifying the number of generations over which to iterate"
    )
    parser.add_argument(
        "-mr", dest="mutation_rate", type=float, required=False, default=0.95,
        help="floating number specifying the rate at which individuals are mutated"
    )

    args = vars(parser.parse_args(args))
    Logger().info(args)
    assert os.path.isfile(args["file_path"]), f"{args['file_path']} does not exist."
    return args


def main():
    pass


if __name__ == "__main__":
    main()
