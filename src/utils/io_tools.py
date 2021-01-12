"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 11/01/2021
"""
import io
import os
import zipfile


def read_text_file(path_txt_file: str) -> list:
    assert os.path.isfile(path_txt_file), f'{path_txt_file} does not exist'
    with open(path_txt_file) as f:
        lines = [line for line in f.readlines()]

    return lines


def compress_folder(path_folder: str):
    assert os.path.isdir(path_folder)

    folder_output = os.path.split(path_folder)[-1]
    data = io.BytesIO()
    with zipfile.ZipFile(data, mode='w') as z:
        for f_name in os.listdir(path_folder):
            z.write(os.path.join(path_folder, f_name), os.path.join(folder_output, f_name))

    data.seek(0)
    return data


def main():
    pass


if __name__ == "__main__":
    main()
