"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 11/01/2021
"""

import os


def read_text_file(path_txt_file: str) -> list:

    assert os.path.isfile(path_txt_file), f'{path_txt_file} does not exist'
    with open(path_txt_file) as f:
        lines = [line for line in f.readlines()]

    return lines


def main():
    pass


if __name__ == "__main__":
    main()
