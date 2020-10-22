"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 12/10/2020
"""
import numpy as np
import os

from src.config import Config
from src.utils.image_tools import generate_triangle_image, show_image
from src.utils.polygon_tools import generate_random_triangles


def main():

    height, width, depth = 500, 500, 3
    triangles = generate_random_triangles(xmax=width, ymax=height, n_triangles=4, n_population=1)
    image_pil = generate_triangle_image(width=width, height=height, triangles=triangles)

    config = Config()
    name_file = "test_image.png"
    path_image = os.path.join(config.path_output, "", name_file)
    image_pil.show()
    image_pil.save(path_image)

    show_image(image_pil)

    return


if __name__ == "__main__":
    main()
