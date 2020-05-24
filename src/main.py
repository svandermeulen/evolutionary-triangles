"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 21/05/2020
"""
import datetime

import cv2
import numpy as np
import os
import pandas as pd
import plotly as py
import plotly.graph_objects as go

from copy import copy
from PIL import Image
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from src.utils.breeding_tools import crossover, mutate_array, get_random_pairs
from src.utils.config import Config
from src.utils.image_tools import compute_distance, generate_triangle_image, convert_pil_to_array
from src.utils.polygon_tools import generate_random_triangles


def mutate_children(children: np.ndarray, xmax: int, ymax: int) -> np.ndarray:
    coordinates_x = mutate_array(children[:, :3, :], max_value=xmax)
    coordinates_y = mutate_array(children[:, 3:6, :], max_value=ymax)
    colors = mutate_array(children[:, 6:, :], max_value=256)

    return np.hstack((coordinates_x, coordinates_y, colors))


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
    fig["layout"]["yaxis"]["range"] = (0, 120)
    fig.update_layout(template="plotly_white")

    return fig


def run_evolution():

    dt_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    config = Config()
    name_file = "test_flower.jpg"
    path_image_ref = os.path.join(config.path_data, name_file)
    path_output = os.path.join(config.path_output, f"run_{dt_date}")
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    image_ref = cv2.imread(path_image_ref)
    height, width, depth = image_ref.shape

    image_pil = Image.new('RGB', (width, height), color=(255, 255, 255))

    mean_distance = compute_distance(img1=image_ref, img2=np.array(image_pil))
    print(f"Initial distance: {mean_distance}")

    population = generate_random_triangles(xmax=width, ymax=height)
    df = pd.DataFrame()

    for i in range(config.generations):
        print(f"Generation: {i}")
        mean_distances = []
        for j, p in enumerate(range(population.shape[-1])):
            image_pil = generate_triangle_image(width=width, height=height, triangles=population[:, :, p])
            mean_distances.append(compute_distance(img1=image_ref, img2=np.array(image_pil)))

        df_temp = pd.DataFrame(
                {
                    "Generation": [i] * len(mean_distances),
                    "Mean_squared_distance": mean_distances
                }
            )

        df = df.append(
            df_temp,
            ignore_index=True,
            sort=False
        )
        print(f"Average distance: {np.mean(mean_distances)}")

        top_indices = df_temp.sort_values(by="Mean_squared_distance").index[:config.n_population // 2]
        population = population[:, :, top_indices]
        population_new = copy(population)

        # Store best image
        image_pil = generate_triangle_image(width=width, height=height, triangles=population[:, :, 0])
        image_pil = convert_pil_to_array(image_pil=image_pil)
        image_pil = np.hstack((image_ref, image_pil))
        path_img = os.path.join(path_output, f"generation_{str(i).zfill(2)}__best_image_{top_indices[0]}.png")
        cv2.imwrite(path_img, image_pil)

        # Crossbreed new offspring
        pairs = get_random_pairs(number_list=list(range(population_new.shape[-1])))
        children = np.zeros((config.n_triangles, 10, config.n_population // 2))
        for pair in pairs:
            children[:, :, pair[0]], children[:, :, pair[1]] = crossover(
                mother=population_new[:, :, pair[0]],
                father=population_new[:, :, pair[1]]
            )
        children_mutated = mutate_children(children=children, xmax=width, ymax=height)
        population = np.uint16(np.dstack((population, children_mutated)))

    df.to_csv(os.path.join(path_output, "distances.csv"), sep=";", index=False)
    fig = plot_distances(df=df)
    py.offline.plot(fig, filename=os.path.join(path_output, "distances.html"), auto_open=True)

    return True


if __name__ == "__main__":
    run_evolution()
