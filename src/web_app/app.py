"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 23/10/2020
"""

import datetime
import os

from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
from flask_caching import Cache
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from wtforms import Form, validators, IntegerField

from src.config import Config
from src.run_evolutionary_triangles import run_evolution
from src.utils.logger import Logger

app = Flask("evolutionary-triangles")
app.config['SECRET_KEY'] = 'your secret key'  # TODO: change to long random string and store in ENV
app.config["IMAGE_FILENAME"] = ""
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config['MAX_IMAGE_FILESIZE'] = 50 * 1024 * 1024
app.config["MAX_IMAGE_PIXELS"] = 256
app.config['GENERATIONS'] = 10
app.config['INDIVIDUALS'] = 10
app.config['TRIANGLES'] = 10
app.config['MUTATION_RATE'] = 95  # Percentage
app.config["SUCCESS"] = False
app.config["OUTPUT_FOLDER"] = os.path.join(
    Config().path_output, f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
app.config["CACHE_TYPE"] = "simple"
app.config["PORT"] = 5000
app.config["DEBUG"] = True
cache = Cache()

# turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)


class InputForm(Form):
    generations = IntegerField(
        label='# generations',
        default=app.config['GENERATIONS'],
        validators=[
            validators.NumberRange(min=1, max=100)
        ])
    individuals = IntegerField(
        label='# individuals',
        default=app.config['INDIVIDUALS'],
        validators=[
            validators.InputRequired(),
            validators.NumberRange(min=1, max=100)
        ])
    triangles = IntegerField(
        label='# triangles',
        default=app.config['TRIANGLES'],
        validators=[
            validators.InputRequired(),
            validators.NumberRange(min=1, max=100)
        ])
    mutation_rate = IntegerField(
        label='mutation rate',
        default=app.config['MUTATION_RATE'],
        validators=[
            validators.InputRequired(),
            validators.NumberRange(min=0, max=100)
        ])


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


def get_files() -> list:
    files = []
    if os.path.isdir(app.config["OUTPUT_FOLDER"]) and app.config["IMAGE_FILENAME"]:
        Logger().debug(app.config["OUTPUT_FOLDER"])
        files = [
            f for f in os.listdir(app.config["OUTPUT_FOLDER"]) if
            any(f.endswith(ext.lower()) for ext in app.config["ALLOWED_IMAGE_EXTENSIONS"])
        ]
        files = [app.config["IMAGE_FILENAME"]] + [f for f in files if f != app.config["IMAGE_FILENAME"]]
        Logger().debug(files)

    return files


@app.route("/home", methods=('GET', 'POST'))
@app.route("/", methods=('GET', 'POST'))
def index():

    folder = os.path.split(app.config["OUTPUT_FOLDER"])[-1]
    return render_template(
        "public/index.html", folder=folder, files=get_files()
    )


@app.route("/configure-process", methods=["GET", "POST"])
def configure_process():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        app.config['GENERATIONS'] = form.generations.data
        app.config['INDIVIDUALS'] = form.individuals.data
        app.config['TRIANGLES'] = form.triangles.data
        app.config['MUTATION_RATE'] = form.mutation_rate.data
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
    if not os.path.isdir(app.config["OUTPUT_FOLDER"]):
        Logger().info("Upload directory does not yet exist. Making it ...")
        os.makedirs(app.config["OUTPUT_FOLDER"])

    path_upload = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    image.save(path_upload)
    Logger().info(f"Uploaded {filename}")

    app.config["IMAGE_PATH"] = path_upload
    app.config["IMAGE_FILENAME"] = filename

    if request.method == 'POST' and request.form['submit_button'] == 'submit':
        config_evo = Config()
        config_evo.path_output = app.config["OUTPUT_FOLDER"]
        config_evo.n_population = app.config["INDIVIDUALS"]
        config_evo.n_triangles = app.config["TRIANGLES"]
        config_evo.n_generations = app.config["GENERATIONS"]
        config_evo.mutation_rate = app.config["MUTATION_RATE"] / 100
        path_image_ref = app.config["IMAGE_PATH"]
        config_evo.side_by_side = False
        app.config["FILES"] = get_files()

        run_evolution(path_to_image=path_image_ref, config=config_evo, web_app_handle=app)

    # return render_template("public/index.html", files=get_files())
    return redirect(url_for("index"))


@app.route('/upload/<folder>/<filename>')
def display_image(folder: str, filename: str):
    Logger().info(f'display_image filename: f{folder}/{filename}')
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


def main():
    # app.run()

    socketio.run(app, port=5000, debug=True)


if __name__ == "__main__":
    main()
