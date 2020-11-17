"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 13/10/2020
"""
import os

import cv2
import re

from src.config import Config


def get_frame_number(file_name: str) -> int:
    return int(re.findall("\d+.png$", file_name)[0].strip(".png"))


def get_file_creation_time(file_name: str):
    return os.path.getctime(file_name)


def main():
    config = Config()

    fps = 20
    path_frames = os.path.join(config.path_output, "run_20201117_123441")
    path_video = os.path.join(path_frames, "video.avi")
    files_frames = [os.path.join(path_frames, f) for f in os.listdir(path_frames) if f.endswith(".png")]

    files_frames = sorted(files_frames, key=get_file_creation_time)

    frame = cv2.imread(files_frames[0])
    height, width, layers = frame.shape

    out = cv2.VideoWriter(path_video, 0, fps, (width, height))
    for f in files_frames:
        # reading each files
        img = cv2.imread(f)
        out.write(img)
    out.release()

    pass


if __name__ == "__main__":
    main()
