"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 21/05/2020
"""
from typing import Union

import cv2
import json
import numpy as np
import os
import pandas as pd
import plotly as py
import plotly.graph_objects as go
import sys

from datetime import datetime
from PIL import Image
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from src.utils.argument_parser import parse_args
from src.utils.breeding_tools import cross_breed_population
from src.config import Config
from src.utils.image_tools import compute_distance, generate_triangle_image, convert_pil_to_array
from src.utils.logger import Logger
from src.utils.polygon_tools import generate_random_triangles, generate_delaunay_triangles
from src.utils.profiler import profile


class EvolutionaryTriangles(object):

    def __init__(self, path_image: str, config: Config, path_output: str = "", local: bool = True):

        Logger().debug(config.__dict__)

        if not path_output:
            date = datetime.now().strftime('%Y%m%d_%H%M%S')
            path_output = os.path.join(config.path_output, f"run_{date}")

        config.path_output = path_output
        config.create_folder(config.path_output)

        self.config = config
        self.local = local
        self.image_ref = cv2.imread(path_image)
        self.height, self.width, self.depth = self.image_ref.shape

        image_white = Image.new('RGBA', (self.width, self.height), color=(255, 255, 255, 255))
        self.mean_distance_init = compute_distance(img1=self.image_ref, img2=convert_pil_to_array(image_white))
        Logger().info(f"Initial distance with completely white image: {self.mean_distance_init}")

        if config.triangulation_method == "non_overlapping":
            self.population = generate_delaunay_triangles(
                xmax=self.width,
                ymax=self.height,
                n_population=config.n_population
            )
        else:
            self.population = generate_random_triangles(
                xmax=self.width,
                ymax=self.height,
                n_population=config.n_population,
                n_triangles=config.n_triangles
            )

    def run_generation(self, i: int) -> pd.DataFrame:

        mean_distances = []
        for p in range(self.population.shape[-1]):
            image_triangles = generate_triangle_image(
                width=self.width,
                height=self.height,
                triangles=self.population[:, :, p],
                triangulation_method=self.config.triangulation_method
            )
            mean_distances.append(compute_distance(img1=self.image_ref, img2=convert_pil_to_array(image_triangles)))

        df_temp = pd.DataFrame({"Generation": [i] * len(mean_distances), "Mean_squared_distance": mean_distances})

        # Find top 50% individuals
        top_indices = df_temp.sort_values(by="Mean_squared_distance").index[:self.config.n_population // 2]
        self.population = self.population[:, :, top_indices]
        image_best = generate_triangle_image(
            width=self.width,
            height=self.height,
            triangles=self.population[:, :, 0],
            triangulation_method=self.config.triangulation_method
        )

        self.write_image(img=image_best, generation=i, img_idx=top_indices[0])

        self.population = cross_breed_population(
            population=self.population,
            config=self.config,
            width=self.width,
            height=self.height
        )

        return df_temp

    def run(self) -> bool:

        df_distances = pd.DataFrame({"Generation": [0], "Mean_squared_distance": [self.mean_distance_init]})
        for i in range(self.config.n_generations):
            Logger().info(f"Generation: {i}")
            df_distance = self.run_generation(i=i)
            df_distances = df_distances.append(df_distance, ignore_index=True, sort=False)
            Logger().info(f"Average distance: {df_distance['Mean_squared_distance'].mean()}")

        fig = self.plot_distances(df=df_distances)
        self.write_results(fig=fig, df_distances=df_distances)

        return True

    @staticmethod
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

    def write_image(self, img: Image, generation: int, img_idx: int) -> bool:
        image_triangles = convert_pil_to_array(image_pil=img)

        if self.config.side_by_side:
            image_triangles = np.hstack((self.image_ref, image_triangles))

        path_img = os.path.join(self.config.path_output,
                                f"generation_{str(generation).zfill(2)}_best_image_{img_idx}.png")
        cv2.imwrite(path_img, image_triangles)
        return True

    def write_results(self, fig: Figure, df_distances: pd.DataFrame) -> Union[bool, str]:

        df_distances.to_csv(os.path.join(self.config.path_output, "distances.csv"), sep=";", index=False)
        with open(os.path.join(self.config.path_output, "config.json"), "w", encoding='utf-8') as f:
            config_dict = self.config.__dict__
            config_dict = {k: val for k, val in config_dict.items() if not k.startswith("path")}
            json.dump(config_dict, f, indent=4)

        if self.local:
            py.offline.plot(fig, filename=os.path.join(self.config.path_output, "distances.html"), auto_open=False)
            return True
        else:
            return py.offline.plot(fig, output_type="div", auto_open=False)


if __name__ == "__main__":

    args = sys.argv[1:]

    if not args:
        config_test = Config()
        name_file = "test_flower.jpg"
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

    EvolutionaryTriangles(
        path_image=path_image_ref,
        config=config_test
    ).run()

    Logger().info("Done")
