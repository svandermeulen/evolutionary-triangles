"""
-*- coding: utf-8 -*-
Written by: sme30393
Date: 23/10/2020
"""
import atexit
import os

from PIL import Image
from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from wtforms import Form, validators, IntegerField

from src.config import Config
from src.run_evolutionary_triangles import run_evolution
from src.utils.logger import Logger

config = Config()
app = Flask("evolutionary-triangles")
app.config['SECRET_KEY'] = 'your secret key'  # TODO: change to long random string and store in ENV
app.config["IMAGE_UPLOADS"] = config.path_output
app.config["IMAGE_FILENAME"] = ""
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config['MAX_IMAGE_FILESIZE'] = 50 * 1024  # * 1024
app.config["MAX_IMAGE_PIXELS"] = 256
app.config['GENERATIONS'] = 10
app.config['INDIVIDUALS'] = 10
app.config['TRIANGLES'] = 10
app.config['MUTATION_RATE'] = 95  # Percentage
app.config["SUCCESS"] = False


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


def resize_image(path_image: str) -> str:
    def _get_new_dimensions(dim1: int, dim2: int) -> (int, int):
        dim1_new = app.config["MAX_IMAGE_PIXELS"]
        dim2_new = int((dim1_new / dim1) * dim2)
        return dim1_new, dim2_new

    with open(path_image, 'r+b') as f:
        with Image.open(f) as image:
            width, height = image.size
            Logger().info(f"{width}, {height}")

            if not width >= app.config["MAX_IMAGE_PIXELS"] or not height >= app.config["MAX_IMAGE_PIXELS"]:
                return path_image

            if width > height:
                width_new, height_new = _get_new_dimensions(dim1=width, dim2=height)
            else:
                height_new, width_new = _get_new_dimensions(dim1=height, dim2=width)

            Logger().info(f"{width_new}, {height_new}")
            image_resized = image.resize((width_new, height_new))

    path_image_resized = os.path.splitext(path_image)[0] + "_resized" + os.path.splitext(path_image)[1]
    image_resized.save(path_image_resized)
    return path_image_resized


@app.route("/home", methods=('GET', 'POST'))
@app.route("/", methods=('GET', 'POST'))
def index():
    files = []
    if os.path.isdir(app.config["IMAGE_UPLOADS"]) and app.config["IMAGE_FILENAME"]:
        files = [
            f for f in os.listdir(app.config["IMAGE_UPLOADS"]) if
            any(f.endswith(ext.lower()) for ext in app.config["ALLOWED_IMAGE_EXTENSIONS"])
        ]
        files = [app.config["IMAGE_FILENAME"]] + [f for f in files if f != app.config["IMAGE_FILENAME"]]
        Logger().info(files)

    if app.config["SUCCESS"]:
        config.n_population = app.config["INDIVIDUALS"]
        config.n_triangles = app.config["TRIANGLES"]
        config.n_generations = app.config["GENERATIONS"]
        config.mutation_rate = app.config["MUTATION_RATE"] / 100
        path_image_ref = app.config["IMAGE_PATH"]
        config.side_by_side = False

        success = run_evolution(path_to_image=path_image_ref, config=config, web_app_handle=app)
        if success:
            Logger().info("Evolutionary triangles finished successfully")
            render_template("public/index.html", files=files)

    return render_template("public/index.html", files=files)


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
    if not os.path.isdir(app.config["IMAGE_UPLOADS"]):
        Logger().info("Upload directory does not yet exist. Making it ...")
        os.makedirs(app.config["IMAGE_UPLOADS"])

    path_upload = os.path.join(app.config["IMAGE_UPLOADS"], filename)
    image.save(path_upload)
    Logger().info(f"Uploaded {filename}")

    app.config["SUCCESS"] = True
    app.config["IMAGE_PATH"] = path_upload
    app.config["IMAGE_FILENAME"] = filename
    return redirect(url_for('index'))


@app.route('/upload/<filename>')
def display_image(filename):
    Logger().info('display_image filename: ' + filename)
    return send_from_directory(app.config["IMAGE_UPLOADS"], filename)


def cleanup() -> bool:
    try:
        Logger().info(f"Trying to remove remove: {app.config['IMAGE_PATH']}")
        os.remove(app.config["IMAGE_PATH"])
    except BaseException as e:
        Logger().error(f"{e}. Removal of {app.config['IMAGE_PATH']} failed, continue exiting without cleanup.")
        pass
    return True


def main():
    pass


if __name__ == "__main__":
    app.run(port=5000, debug=True)
    atexit.register(cleanup)
