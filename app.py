"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 23/10/2020
"""
from datetime import datetime
from random import randint

import cv2
import numpy as np
import os
import pandas as pd
import platform

from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
from flask_socketio import SocketIO
from markupsafe import Markup
from threading import Thread, Event
from werkzeug.utils import secure_filename
from wtforms import Form, validators, IntegerField, SelectField

from src.config import Config
from src.create_video import create_video
from src.run_evolutionary_triangles import EvolutionaryTriangles
from src.utils.io_tools import read_text_file
from src.utils.logger import Logger

app = Flask("evolutionary-triangles")
app.config['SECRET_KEY'] = 'your secret key'  # TODO: change to long random string and store in ENV
app.config["IMAGE_FILENAME"] = ""
app.config["EXTENSIONS_ALLOWED"] = ["JPEG", "JPG", "PNG"]
app.config['MAX_IMAGE_FILESIZE'] = 50 * 1024 * 1024
app.config["MAX_IMAGE_PIXELS"] = 256
app.config['GENERATIONS'] = 10
app.config['INDIVIDUALS'] = 10
app.config['TRIANGLES'] = 10
app.config['MUTATION_PERCENTAGE'] = 5
app.config["CROSSOVER_RATE"] = 95
app.config["TRIANGULATION_METHOD"] = "overlapping"
app.config["SUCCESS"] = False
app.config["PORT"] = 5000
app.config["DEBUG"] = True
app.config["FOLDER_OUTPUT"] = ""
app.config["GRAPH_DIV"] = ""

# turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=False, engineio_logger=False)

# Evolutionary Triangles thread
thread = Thread()
thread_stop_event = Event()


class InputForm(Form):
    generations = IntegerField(
        label='# generations',
        default=app.config['GENERATIONS'],
        validators=[
            validators.NumberRange(min=1, max=100)
        ]
    )
    individuals = IntegerField(
        label='# individuals',
        default=app.config['INDIVIDUALS'],
        validators=[
            validators.InputRequired(),
            validators.NumberRange(min=1, max=100)
        ]
    )
    triangles = IntegerField(
        label='# triangles',
        default=app.config['TRIANGLES'],
        validators=[
            validators.InputRequired(),
            validators.NumberRange(min=1, max=100)
        ]
    )
    mutation_rate = IntegerField(
        label='mutation rate',
        default=app.config['MUTATION_PERCENTAGE'],
        validators=[
            validators.InputRequired(),
            validators.NumberRange(min=0, max=100)
        ]
    )
    crossover_rate = IntegerField(
        label="crossover rate",
        default=app.config["CROSSOVER_RATE"],
        validators=[
            validators.InputRequired(),
            validators.NumberRange(min=0, max=100)
        ]
    )
    triangulation_method = SelectField(
        label="triangulation_method",
        coerce=str,
        choices=["non_overlapping", "overlapping"],
        validators=[
            validators.InputRequired()
        ]
    )


def allowed_image(filename: str) -> bool:
    # We only want files with a . in the filename
    if "." not in filename:
        return False

    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in EXTENSIONS_ALLOWED
    if ext.upper() in app.config["EXTENSIONS_ALLOWED"]:
        return True
    return False


def allowed_image_filesize(filesize):
    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    return False


def get_image_size() -> tuple:
    image = cv2.imread(app.config["PATH_IMAGE"])
    return image.shape[:2]


@socketio.on('connect', namespace='/index')
@app.route("/home", methods=('GET', 'POST'))
@app.route("/", methods=('GET', 'POST'))
def index():
    lines_intro = read_text_file(os.path.join(Config().path_data, "introduction.txt"))
    lines_evo = read_text_file(os.path.join(Config().path_data, "evo_algorithm.txt"))
    lines_diy = read_text_file(os.path.join(Config().path_data, "do_it_yourself.txt"))

    if request.method == "POST":
        if request.form['submit_button'] == 'submit':
            return redirect(url_for("configure_process"))

    return render_template(
        "public/index.html",
        lines_intro=lines_intro,
        lines_evo=lines_evo,
        lines_diy=lines_diy,
        folder=os.path.basename(app.config["FOLDER_OUTPUT"])
    )


@app.route('/results')
def results():
    if not app.config["GRAPH_DIV"]:
        return redirect(url_for("index"))
    return render_template(
        "public/results.html",
        div_placeholder=Markup(app.config["GRAPH_DIV"]),
        folder=os.path.basename(app.config["FOLDER_OUTPUT"]),
        file=os.path.basename(app.config["PATH_GIF"])
    )


@app.route("/configure-process", methods=["GET", "POST"])
def configure_process():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        app.config['GENERATIONS'] = form.generations.data
        app.config['INDIVIDUALS'] = form.individuals.data
        app.config['TRIANGLES'] = form.triangles.data
        app.config['MUTATION_PERCENTAGE'] = form.mutation_rate.data
        app.config["CROSSOVER_RATE"] = form.crossover_rate.data
        app.config["TRIANGULATION_METHOD"] = form.triangulation_method.data
    else:
        if form.errors:
            Logger().error(form.errors)
            for item, message in form.errors.items():
                flash(f'{item}: {message}')
            return redirect(request.url)

    if not request.method == "POST":
        return render_template("public/configure_process.html", form=form)
    if not request.files:
        flash(f'Upload an image!')
        Logger().error("No file given")
        return redirect(request.url)
    if "filesize" not in request.cookies:
        flash(f'Failed to extract the file size from the given image')
        Logger().error("Filesize could not be retrieved from input")
        return redirect(request.url)
    if not allowed_image_filesize(request.cookies["filesize"]):
        flash(f'Filesize should be below {app.config["MAX_IMAGE_FILESIZE"]} bytes')
        Logger().error(f"Filesize exceeded maximum limit of {app.config['MAX_IMAGE_FILESIZE']} bytes")
        return redirect(request.url)

    image = request.files["image"]
    if not image.filename:
        flash(f'Upload an image')
        Logger().error("No filen given")
        return redirect(request.url)

    if not allowed_image(filename=image.filename):
        Logger().error("That file extension is not allowed")
        return redirect(request.url)

    filename = secure_filename(image.filename)

    if request.method == 'POST' and request.form['submit_button'] == 'submit':

        config_evo = Config()
        date_run = datetime.now().strftime('%Y%m%d_%H%M%S')
        app.config["FOLDER_OUTPUT"] = os.path.join(config_evo.path_output, f"run_{date_run}")
        os.mkdir(app.config["FOLDER_OUTPUT"])
        path_upload = os.path.join(app.config["FOLDER_OUTPUT"], filename)
        npimg = np.frombuffer(image.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        app.config["PATH_IMAGE"] = path_upload
        app.config["IMAGE_FILENAME"] = filename
        app.config["PATH_GIF"] = os.path.join(app.config["FOLDER_OUTPUT"], "evolutionary_triangles.gif")
        config_evo.path_image_ref = path_upload
        config_evo.n_population = app.config["INDIVIDUALS"]
        config_evo.n_triangles = app.config["TRIANGLES"]
        config_evo.n_generations = app.config["GENERATIONS"]
        config_evo.mutation_rate = app.config["MUTATION_PERCENTAGE"] / 100
        config_evo.crossover_rate = app.config["CROSSOVER_RATE"] / 100
        config_evo.triangulation_method = app.config["TRIANGULATION_METHOD"]
        config_evo.side_by_side = True

        et = EvolutionaryTriangles(
            image_ref=image,
            image_name=filename,
            path_output=app.config["FOLDER_OUTPUT"],
            config=config_evo,
            local=False
        )

        global thread
        Logger().info('Client connected')
        if not thread.is_alive():
            Logger().info("Starting Thread")
            thread = socketio.start_background_task(run_evolution, et=et)

    return redirect(url_for("index"))


@socketio.on('disconnect', namespace='/index')
def test_disconnect():
    Logger().info('Client disconnected')


def run_evolution(et: EvolutionaryTriangles) -> bool:
    df_distances = pd.DataFrame({"Generation": [0], "Mean_squared_distance": [et.fitness_initial]})
    for generation in range(app.config["GENERATIONS"]):
        socketio.emit('generation', {'integer': generation+1, 'total': app.config["GENERATIONS"]}, namespace='/index')
        socketio.sleep(1)
        df_distance = et.run_generation(generation=generation)
        df_distances = df_distances.append(df_distance, ignore_index=True, sort=False)

    fig = et.plot_distances(df=df_distances)
    app.config["GRAPH_DIV"] = et.write_results(fig=fig, df_distances=df_distances)
    create_video(
        path_image_ref=app.config["PATH_IMAGE"],
        dir_images=app.config["FOLDER_OUTPUT"],
        path_video=app.config["PATH_GIF"],
        fps=et.config.fps
    )

    return True


@app.route('/upload/<folder>/<filename>')
def display_image(folder: str, filename: str):
    Logger().info(f'Displaying: {filename}')
    return send_from_directory(app.config["FOLDER_OUTPUT"], filename)


def main():
    if platform.system() == "Windows":
        socketio.run(app, port=5000, debug=True)
    else:
        socketio.run(app, port=80, host="0.0.0.0", debug=True)
    return True


if __name__ == "__main__":
    main()
