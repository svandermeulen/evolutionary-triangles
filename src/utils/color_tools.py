"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 21/05/2020
"""
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor


def convert_rgb_to_lab(rgb_color: list) -> list:
    """Returning the lab color equivalent of the given rgb color"""
    if rgb_color == [None, None, None]:
        return rgb_color

    rgb = sRGBColor(*[val/255 for val in rgb_color])
    lab = convert_color(rgb, LabColor)
    lab = [lab.lab_l, lab.lab_a, lab.lab_b]
    return [round(a, 4) for a in lab]


def main():
    pass


if __name__ == "__main__":
    main()
