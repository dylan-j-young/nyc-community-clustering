# Mapping NYC Communities with Demographic Data

Welcome! This project studies demographic, housing, and economic data from the 2020 Decennial US Census and the 2023 5-Year American Community Survey (ACS) and uses spatial clustering to identify communities of similar people and households. This repository holds my code and notebooks, which you can replicate by following the setup instructions below. To read a high-level summary of the project, check out my GitHub Pages site [here](https://dylan-j-young.github.io/projects/nyc-community-clustering/).

DISCLAIMER: This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau.

## Project structure

The project has the following simplified root-level structure, roughly inspired by [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/):

- `config/` (variable names for Census data tables)
- `data/` (*CREATED UPON RUNNING NOTEBOOK 00. Holds Census data and geometries)
- `models/` (*CREATED UPON RUNNING NOTEBOOK 00. Holds cluster labels for various models and hyperparameters)
- `notebooks/` (main notebooks for the project. Running these in order will walk you through the entire project)
- `src/` (Python scripts used in the notebooks, treated as a module to import)
- `.env` (*MUST CREATE BEFORE RUNNING. Stores the user's Census API key)
- `requirements.txt` (List of Python modules required to run the project)

## Setup

1. Clone the repository using your preferred method.
2. Install any dependencies in `requirements.txt`. Either of the following will work:
    - Run the command `pip install -r requirements.txt`.
    - Install each module using your preferred package manager, such as `conda`. Some packages may only be available on `conda-forge`. 
3. To pull data from the Census API, it's recommended to request an API key (limited queries are available without one, but frequent queries require a key). You can make this request at [https://www.census.gov/data/developers.html](https://www.census.gov/data/developers.html).
4. Once you get your key, create a file called `.env` in the project root with the following line:
    ```
    CENSUS_API_KEY=your_key_here
    ```
5. Run the script `run_pipeline.py` to complete the directory structure, download data, and build the SQLite database.
<!-- Run all cells in the notebook `notebooks/00_preprocessing.ipynb` to complete the directory structure and download all relevant data and geometries. -->
6. If you want to replicate my work, first run through notebook `notebooks/01_feature-engineering.ipynb` to generate the dataset I used for clustering. Then run through notebook `notebooks/02_clustering.ipynb` to look at different clustering algorithms and evaluation metrics.