"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 23/10/2020
"""
import cv2
import datetime
import os
import pandas as pd
import platform

from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
from flask_socketio import SocketIO
from markupsafe import Markup
from PIL import Image
from threading import Thread, Event
from werkzeug.utils import secure_filename
from wtforms import Form, validators, IntegerField, SelectField

from src.config import Config
from src.create_video import create_video
from src.run_evolutionary_triangles import EvolutionaryTriangles
from src.utils.image_tools import draw_text, convert_pil_to_array, resize_image
from src.utils.logger import Logger

app = Flask("evolutionary-triangles")
app.config['SECRET_KEY'] = 'your secret key'  # TODO: change to long random string and store in ENV
app.config["IMAGE_FILENAME"] = ""
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]
app.config['MAX_IMAGE_FILESIZE'] = 50 * 1024 * 1024
app.config["MAX_IMAGE_PIXELS"] = 256
app.config['GENERATIONS'] = 10
app.config['INDIVIDUALS'] = 10
app.config['TRIANGLES'] = 10
app.config['MUTATION_PERCENTAGE'] = 5
app.config["SURVIVAL_PERCENTAGE"] = 50
app.config["TRIANGULATION_METHOD"] = "overlapping"
app.config["SUCCESS"] = False
app.config["PORT"] = 5000
app.config["DEBUG"] = True
app.config["OUTPUT_FOLDER"] = ""
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
    survival_rate = IntegerField(
        label="survival rate",
        default=app.config["SURVIVAL_PERCENTAGE"],
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

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    return False


def allowed_image_filesize(filesize):
    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    return False


def get_image_size() -> tuple:
    image = cv2.imread(app.config["PATH_IMAGE"])
    return image.shape[:2]


def get_files() -> list:
    files = []
    if os.path.isdir(app.config["OUTPUT_FOLDER"]) and app.config["IMAGE_FILENAME"]:
        files = [
            f for f in os.listdir(app.config["OUTPUT_FOLDER"]) if
            any(f.endswith(ext.lower()) for ext in app.config["ALLOWED_IMAGE_EXTENSIONS"])
        ]
        files = [app.config["IMAGE_FILENAME"]] + sorted([f for f in files if f != app.config["IMAGE_FILENAME"]])

    return files


@app.route("/home", methods=('GET', 'POST'))
@app.route("/", methods=('GET', 'POST'))
def index():
    return render_template("public/index.html", folder=os.path.basename(app.config["OUTPUT_FOLDER"]), files=get_files())


@app.route('/results')
def results():
    if not app.config["GRAPH_DIV"]:
        return redirect(url_for("index"))
    return render_template(
        "public/results.html",
        div_placeholder=Markup(app.config["GRAPH_DIV"]),
        folder=os.path.basename(app.config["OUTPUT_FOLDER"]),
        file=os.path.basename(app.config["PATH_GIF"])
    )


@socketio.on('connect', namespace='/index')
@app.route("/configure-process", methods=["GET", "POST"])
def configure_process():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        app.config['GENERATIONS'] = form.generations.data
        app.config['INDIVIDUALS'] = form.individuals.data
        app.config['TRIANGLES'] = form.triangles.data
        app.config['MUTATION_PERCENTAGE'] = form.mutation_rate.data
        app.config["SURVIVAL_PERCENTAGE"] = form.survival_rate.data
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
        dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        app.config["OUTPUT_FOLDER"] = os.path.join(Config().path_output, f"run_{dt}")
        if not os.path.isdir(app.config["OUTPUT_FOLDER"]):
            Logger().info("Upload directory does not yet exist. Making it ...")
            os.makedirs(app.config["OUTPUT_FOLDER"])

        path_upload = os.path.join(app.config["OUTPUT_FOLDER"], filename)
        image.save(path_upload)
        image = cv2.imread(path_upload)
        image = resize_image(image)
        cv2.imwrite(path_upload, image)

        app.config["PATH_IMAGE"] = path_upload
        app.config["IMAGE_FILENAME"] = filename
        app.config["IMAGE_HEIGHT"], app.config["IMAGE_WIDTH"] = get_image_size()
        app.config["PATH_GIF"] = os.path.join(app.config["OUTPUT_FOLDER"], "evolutionary_triangles.gif")
        config_evo.n_population = app.config["INDIVIDUALS"]
        config_evo.n_triangles = app.config["TRIANGLES"]
        config_evo.n_generations = app.config["GENERATIONS"]
        config_evo.mutation_rate = app.config["MUTATION_PERCENTAGE"] / 100
        config_evo.survival_rate = app.config["SURVIVAL_PERCENTAGE"] / 100
        config_evo.triangulation_method = app.config["TRIANGULATION_METHOD"]
        path_image_ref = app.config["PATH_IMAGE"]
        config_evo.side_by_side = True
        app.config["FILES"] = get_files()

        # need visibility of the global thread object
        global thread
        Logger().info('Client connected')

        et = EvolutionaryTriangles(
            path_image=path_image_ref,
            config=config_evo,
            path_output=app.config["OUTPUT_FOLDER"],
            local=False
        )

        if not thread.is_alive():
            Logger().info("Starting Thread")
            thread = socketio.start_background_task(run_evolution, et=et)

    return redirect(url_for("index"))


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    Logger().info('Client disconnected')


def generate_waiting_image(generation: int) -> bool:
    image = Image.new('RGB', (app.config["IMAGE_WIDTH"], app.config["IMAGE_HEIGHT"]), (255, 255, 255))
    image = draw_text(
        image=image,
        text=f"Waiting for generation {generation + 1} ...",
        text_color=(0, 0, 0)
    )
    image_array = convert_pil_to_array(image_pil=image)
    path_img = os.path.join(app.config["OUTPUT_FOLDER"], f"waiting.png")
    cv2.imwrite(path_img, image_array)
    return True


def run_evolution(et: EvolutionaryTriangles) -> bool:
    Logger().info('Client connected')

    df_distances = pd.DataFrame({"Generation": [0], "Mean_squared_distance": [et.fitness_initial]})
    for generation in range(app.config["GENERATIONS"]):
        generate_waiting_image(generation=generation)
        if generation == 0:
            socketio.emit('reload', namespace='/index')
        df_distance = et.run_generation(generation=generation)
        socketio.emit('reload', namespace='/index')
        df_distances = df_distances.append(df_distance, ignore_index=True, sort=False)

    os.remove(os.path.join(app.config["OUTPUT_FOLDER"], f"waiting.png"))

    fig = et.plot_distances(df=df_distances)
    app.config["GRAPH_DIV"] = et.write_results(fig=fig, df_distances=df_distances)
    create_video(
        path_image_ref=app.config["PATH_IMAGE"],
        dir_images=app.config["OUTPUT_FOLDER"],
        path_video=app.config["PATH_GIF"],
        fps=et.config.fps
    )

    return True


@app.route('/upload/<folder>/<filename>')
def display_image(folder: str, filename: str):
    Logger().info(f'Displaying: {filename}')
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


def main():
    if platform.system() == "Windows":
        socketio.run(app, port=5000, debug=True)
    else:
        socketio.run(app, port=5000, host="0.0.0.0", debug=True)


if __name__ == "__main__":
    main()
