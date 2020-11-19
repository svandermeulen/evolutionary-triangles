FROM continuumio/miniconda3

MAINTAINER Stef van der Meulen "stefvandermeulen@gmail.com"

RUN apt-get update -y && apt-get install -y python-pip && apt-get install -y libgl1-mesa-glx && apt-get install -y bash-completion

# Create the environment:
COPY requirements.txt evolutionary-triangles/requirements.txt

WORKDIR evolutionary-triangles

RUN pip install -r requirements.txt

COPY ./src ./src

ENV PYTHONPATH=/evolutionary-triangles:/evolutionary-triangles/src

WORKDIR src/web_app

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]