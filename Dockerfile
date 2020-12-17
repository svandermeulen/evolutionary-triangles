FROM continuumio/miniconda3

MAINTAINER Stef van der Meulen "stefvandermeulen@gmail.com"

RUN apt-get update -y && apt-get install -y python-pip && apt-get install -y libgl1-mesa-glx && apt-get install -y bash-completion

# Create the environment:
COPY requirements.txt evolutionary-triangles/requirements.txt

WORKDIR evolutionary-triangles

RUN pip install -r requirements.txt

COPY ./ ./
ENV PYTHONPATH=/evolutionary-triangles:/evolutionary-triangles/src
# ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT python app.py