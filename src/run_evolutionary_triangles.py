"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 21/05/2020
"""
import json
import sys

import cv2
import numpy as np
import os
import pandas as pd
import plotly as py
import plotly.graph_objects as go

from PIL import Image
from flask import Flask, redirect, request
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from src.utils.argument_parser import parse_args
from src.utils.breeding_tools import cross_breed_population
from src.config import Config
from src.utils.image_tools import compute_distance, generate_triangle_image, convert_pil_to_array
from src.utils.logger import Logger
from src.utils.polygon_tools import generate_random_triangles
from src.utils.profiler import profile


@profile
def plot_distances(df: pd.DataFrame) -> Figure:
    df_agg = df.groupby(by="Generation").agg({"Mean_squared_distance": [np.nanmean, np.nanstd]}).reset_index()

    fig = make_subplots(rows=1, cols=1, print_grid=False)

    trace = go.Scatter(
        x=df_agg["Generation"],
        y=df_agg["Mean_squared_distance"]["nanmean"],
        error_y=dict(
            type='data',
            array=df_agg["Mean_squared_distance"]["nanstd"].values
        ),
        name="Generation_progress",
        mode="markers"
    )
    fig.append_trace(trace, 1, 1)

    fig["layout"]["xaxis"]["title"] = "Generation"
    fig["layout"]["yaxis"]["title"] = "MSD"
    fig["layout"]["yaxis"]["range"] = (0, 140)
    fig.update_layout(template="plotly_white")

    return fig


def write_image(img_ref: Image, img: Image, generation: int, img_idx: int, path_output, side_by_side: bool) -> bool:
    image_triangles = convert_pil_to_array(image_pil=img)

    if side_by_side:
        image_triangles = np.hstack((img_ref, image_triangles))

    path_img = os.path.join(path_output, f"generation_{str(generation).zfill(2)}__best_image_{img_idx}.png")
    cv2.imwrite(path_img, image_triangles)
    return True


@profile
def run_evolution(path_to_image: str, config: Config, web_app_handle: Flask = None) -> bool:
    Logger().debug(config.__dict__)
    path_output = config.path_output
    image_ref = cv2.imread(path_to_image)
    height, width, depth = image_ref.shape

    image_white = Image.new('RGBA', (width, height), color=(255, 255, 255, 255))
    mean_distance = compute_distance(img1=image_ref, img2=convert_pil_to_array(image_white))
    Logger().info(f"Initial distance with completely white image: {mean_distance}")

    population = generate_random_triangles(
        xmax=width,
        ymax=height,
        n_population=config.n_population,
        n_triangles=config.n_triangles
    )
    df_dist = pd.DataFrame({"Generation": [0], "Mean_squared_distance": [mean_distance]})

    generations = list(range(1, config.n_generations + 1))
    for i in generations:
        Logger().info(f"Generation: {i}")
        mean_distances = []
        for j, p in enumerate(range(population.shape[-1])):
            image_triangles = generate_triangle_image(width=width, height=height, triangles=population[:, :, p])
            mean_distances.append(compute_distance(img1=image_ref, img2=convert_pil_to_array(image_triangles)))

        df_temp = pd.DataFrame({"Generation": [i] * len(mean_distances), "Mean_squared_distance": mean_distances})

        df_dist = df_dist.append(df_temp, ignore_index=True, sort=False)
        Logger().info(f"Average distance: {np.mean(mean_distances)}")

        # Find top 50% individuals
        top_indices = df_temp.sort_values(by="Mean_squared_distance").index[:config.n_population // 2]
        population = population[:, :, top_indices]
        image_best = generate_triangle_image(width=width, height=height, triangles=population[:, :, 0])
        write_image(
            img_ref=image_ref,
            img=image_best,
            generation=i,
            img_idx=top_indices[0],
            path_output=path_output,
            side_by_side=config.side_by_side
        )

        population = cross_breed_population(population=population, config=config, width=width, height=height)

        if web_app_handle is not None:
            redirect(request.url)
        else:
            Logger().debug("No application given")

    fig = plot_distances(df=df_dist)

    df_dist.to_csv(os.path.join(path_output, "distances.csv"), sep=";", index=False)
    py.offline.plot(fig, filename=os.path.join(path_output, "distances.html"), auto_open=True)

    with open(os.path.join(path_output, "config.json"), "w", encoding='utf-8') as f:
        json.dump(config.__dict__, f, indent=4)

    return True


if __name__ == "__main__":

    args = sys.argv[1:]

    if not args:
        config_test = Config()
        name_file = "test_panda.jpg"
        path_image_ref = os.path.join(config_test.path_data, name_file)
    else:

        args = parse_args(args)
        config_test = Config(
            n_population=args["n_population"],
            n_triangles=args["n_triangles"],
            n_generations=args["n_generations"],
            mutation_rate=args["mutation_rate"]
        )
        path_image_ref = args["file_path"]

    run_evolution(path_to_image=path_image_ref, config=config_test, web_app_handle=Flask)

    Logger().info("Done")
