"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 13/10/2020
"""
import cv2
import numpy as np
import os

from PIL import Image

from src.config import Config
from src.utils.image_tools import resize_image
from src.utils.logger import Logger


DEFAULT_HEIGHT = 256


def get_file_creation_time(file_name: str):
    return os.path.getctime(file_name)


def create_video(path_image_ref: str, dir_images: str, path_video: str, fps: int) -> bool:

    files_frames = [os.path.join(dir_images, f) for f in os.listdir(dir_images) if f.endswith(".png") and
                    f.startswith("generation")
                    ]
    files_frames = sorted(files_frames, key=get_file_creation_time)

    image_ref = cv2.imread(path_image_ref)
    image_ref = resize_image(image=image_ref)

    if path_video.endswith(".gif"):
        images = []
        for file in files_frames:

            image = cv2.imread(file)
            image = resize_image(image=image)
            image = np.hstack((image_ref, image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            images.append(Image.fromarray(image, mode="RGB"))
        images[0].save(path_video, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
    elif path_video.endswith(".avi"):
        frame = cv2.imread(files_frames[0])
        frame = np.hstack((image_ref, frame))
        height, width, layers = frame.shape

        out = cv2.VideoWriter(path_video, 0, fps, (width, height))
        for f in files_frames:
            # reading each files
            img = cv2.imread(f)
            img = np.hstack((image_ref, img))
            out.write(img)
        out.release()
    else:
        Logger().error(f"Invalid file extension for the video given {path_video}. Choose from '.avi' or '.gif'")

    return True


def main():

    config = Config()
    path_frames = os.path.join(config.path_output, "run_20201117_193549")
    path_video = os.path.join(path_frames, "video.gif")
    create_video(dir_images=path_frames, path_video=path_video, fps=config.fps)

    return True


if __name__ == "__main__":
    main()
