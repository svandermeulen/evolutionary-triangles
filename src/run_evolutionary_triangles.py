"""
-*- coding: utf-8 -*-
Written by: stef.vandermeulen
Date: 21/05/2020
"""
import random
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

from src.genetic_algorithm.individual import Individual
from src.genetic_algorithm.mutation import mutate_individual
from src.utils.argument_parser import parse_args
from src.genetic_algorithm.crossover import Crossover
from src.config import Config
from src.utils.image_tools import compute_distance, generate_triangle_image, convert_pil_to_array, resize_image
from src.utils.logger import Logger
from src.utils.profiler import profile


class EvolutionaryTriangles(object):

    def __init__(self, config: Config, path_output: str = "", local: bool = True):

        if not path_output:
            date = datetime.now().strftime('%Y%m%d_%H%M%S')
            path_output = os.path.join(config.path_output, f"run_{date}")

        config.path_output = path_output
        config.create_folder(config.path_output)

        self.config = config
        self.local = local
        self.image_ref = config.image_ref
        self.height, self.width, self.depth = self.image_ref.shape

        image_white = Image.new('RGBA', (self.width, self.height), color=(255, 255, 255, 255))
        self.fitness_initial = compute_distance(img1=self.image_ref, img2=convert_pil_to_array(image_white))
        Logger().info(f"Initial distance with completely white image: {self.fitness_initial}")

        self.population = self.generate_population()

    def generate_population(self) -> list:
        return [Individual() for _ in range(self.config.n_population)]

    def store_best_individual(self, best_idx: int, generation: int) -> bool:

        Logger().info(f"Best individual: {best_idx}")
        individual_best = self.population[best_idx].individual if self.config.triangulation_method == "overlapping" \
            else self.population[best_idx].convert_points_to_triangles()

        image_best = generate_triangle_image(
            width=self.width,
            height=self.height,
            triangles=individual_best
        )
        return self.write_image(img=image_best, generation=generation, img_idx=best_idx)

    def run_generation(self, generation: int) -> pd.DataFrame:

        fitnesses = [individual.fitness for individual in self.population]
        df_temp = pd.DataFrame(fitnesses, columns=["Fitness"])
        df_temp["Individual"] = df_temp.index
        df_temp["Generation"] = generation
        df_temp = df_temp.sort_values(by="Fitness")
        self.store_best_individual(best_idx=df_temp.index[0], generation=generation)

        # Keep top n individuals (n = population_size)
        df_temp = df_temp.head(self.config.n_population)
        self.population = [self.population[i] for i in df_temp.index]

        for p in range(0, self.config.n_population):

            if (p * 2) + 1 >= self.config.n_population:
                break

            # Pair selection
            mother_idx, father_idx = self.select_parents()

            # Apply crossover
            children = Crossover(
                mother=self.population[mother_idx].individual,
                father=self.population[father_idx].individual
                ).apply_crossover()

            # Mutate
            for child in children:

                if self.config.triangulation_method != "overlapping":
                    child_mutated = mutate_individual(individual=child, yidx=1, coloridx=2)
                else:
                    child_mutated = mutate_individual(individual=child)

                self.population.append(
                    child_mutated
                )

        return df_temp

    @profile
    def run(self) -> bool:

        df_distances = pd.DataFrame({"Individual": [0], "Generation": [0], "Fitness": [self.fitness_initial]})
        for i in range(1, self.config.n_generations):
            Logger().info(f"Generation: {i}")
            df_distance = self.run_generation(generation=i)
            df_distances = df_distances.append(df_distance, ignore_index=True, sort=False)
            Logger().info(f"Average fitness: {df_distance['Fitness'].mean()}")

        fig = self.plot_distances(df=df_distances)
        self.write_results(fig=fig, df_distances=df_distances)

        return True

    def get_parent(self):
        if random.uniform(0, 1) > 0.5:
            return self.tournament_selection()
        return self.biased_random_selection()

    def select_random_individual(self) -> int:
        return random.randint(0, self.config.n_population - 1)

    def tournament_selection(self) -> int:

        candidate_one = self.select_random_individual()
        candidate_two = self.select_random_individual()
        while candidate_one == candidate_two:
            candidate_two = self.select_random_individual()

        if self.population[candidate_one].fitness > self.population[candidate_two].fitness:
            return candidate_one
        return candidate_two

    def biased_random_selection(self) -> int:

        fitness_sum = sum([individual.fitness for individual in self.population])
        proportions = [fitness_sum / individual.fitness for individual in self.population]
        proportions_sum = sum(proportions)
        proportions_norm = [p / proportions_sum for p in proportions]
        proportions_cummulative = np.cumsum(proportions_norm)
        select_value = random.uniform(0, 1)
        return np.min(np.where(proportions_cummulative > select_value))

    def select_parents(self) -> tuple:

        mother = self.get_parent()
        father = self.get_parent()
        while mother == father:
            father = self.get_parent()  # avoid crossover with oneself
        return mother, father

    @staticmethod
    @profile
    def plot_distances(df: pd.DataFrame) -> Figure:
        df_agg = df.groupby(by="Generation").agg({"Fitness": [np.nanmean, np.nanstd]}).reset_index()

        fig = make_subplots(rows=1, cols=1, print_grid=False)

        trace = go.Scatter(
            x=df_agg["Generation"],
            y=df_agg["Fitness"]["nanmean"],
            error_y=dict(
                type='data',
                array=df_agg["Fitness"]["nanstd"].values
            ),
            name="Generation_progress",
            mode="markers"
        )
        fig.append_trace(trace, 1, 1)

        fig["layout"]["xaxis"]["title"] = "Generation"
        fig["layout"]["yaxis"]["title"] = "Fitness"
        fig["layout"]["yaxis"]["range"] = (0, 140)
        fig.update_layout(template="plotly_white")

        return fig

    def write_image(self, img: Image, generation: int, img_idx: int) -> bool:
        image_triangles = convert_pil_to_array(image_pil=img)
        path_img = os.path.join(self.config.path_output,
                                f"generation_{str(generation).zfill(2)}_best_image_{img_idx}.png")
        cv2.imwrite(path_img, image_triangles)
        return True

    def write_results(self, fig: Figure, df_distances: pd.DataFrame) -> Union[bool, str]:

        df_distances.to_csv(os.path.join(self.config.path_output, "distances.csv"), sep=";", index=False)
        with open(os.path.join(self.config.path_output, "config.json"), "w", encoding='utf-8') as f:
            config_dict = self.config.__dict__
            config_dict = {k: val for k, val in config_dict.items() if not k.startswith("path") and k != "image_ref"}
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
        path_image_ref = os.path.join(config_test.path_data, "test", name_file)
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
        config=config_test
    ).run()

    Logger().info("Done")
