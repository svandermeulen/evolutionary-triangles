# evolutionary-triangles
Represent an image using only triangles which coordinates and colors are tuned using an Evolutionary Algorithm. 

## Introduction
I created this web application primarily for myself to learn more about genetic algorithms and building/deploying web 
applications using Flask, Docker and Azure App Services.
With the web app you can upload an image and start the Evolutionary Algorithm to obtain a triangulated representation of
that image

### Approach
<p>
    In short, the algorithm's objective is to take an input image and to try to draw a close representation using only
    triangles, also referred to as triangulation.
    This is achieved by applying a genetic approach, where a set of predefined number of individuals are mutated and
    crossbred to create new generations, of which only those with the highest fitness are kept.
    The main sequence of operations is:
</p>
<ul>
    <li>Generate individuals.
    <li>Compute the fitness of each of the individuals,
    <li>Crossbreed pairs of individuals, slightly favoring individuals with higher fitness scores.
    <li>Mutate the outcoming children
    <li>Only keep the top n of the population based on their fitness scores for the next generation
    <li>Repeat above steps for m number of generations
</ul>

## Installation

Application is built on Python 3.8 or higher

Create virtual environment: `python -m venv path\to\venv`

Activate virtual environment: `path\to\venv\Scripts\activate`

Install packages: `pip install -r requirements.txt`

Run the app on a local server: `python app.py`



