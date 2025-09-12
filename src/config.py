import numpy as np
import pandas as pd

import os
from pathlib import Path
import yaml

#### 
# GLOBAL SETTINGS
####

# Pandas settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

#### 
# GEOGRAPHY
####

# Coordinate system codes
WEB_MERCATOR_EPSG = 3857 # Best for display
WGS84_EPSG = 4326 # Default CRS for interoperability

# FIPS county codes
FIPS_DICT = {
    "36005": "The Bronx",
    "36047": "Brooklyn",
    "36061": "Manhattan",
    "36081": "Queens",
    "36085": "Staten Island"
}

# Boroughs
BOROUGHS = ["The Bronx",
            "Brooklyn",
            "Manhattan",
            "Queens",
            "Staten Island"]

# # Distance
# MEAN_EARTH_RADIUS = 6371 * 1000 # m
# KM = 1000 # m
# DEGREE_TO_M = np.pi/180 * MEAN_EARTH_RADIUS

####
# DATA PATHS
####

# Data directories
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Shapefiles: NYC census tracts
RAW_SHAPEFILES_DIR = RAW_DATA_DIR / "shapefiles"
TRACTS_RAW = (
    RAW_SHAPEFILES_DIR 
    / "tl_2024_36_tract" 
    / "tl_2024_36_tract.shp"
)
TRACTS_CLEAN = INTERIM_DATA_DIR / "tracts_clean.parquet"

# Shapefiles: Water area polygons for each county/borough
AREAWATER = [(
    RAW_SHAPEFILES_DIR
    / f"tl_2024_{a}_areawater"
    / f"tl_2024_{a}_areawater.shp"
    ) for a in FIPS_DICT.keys()
]

# 2020 Census Demographic Profile
DECENNIAL2020_DP_RAW = RAW_DATA_DIR / "decennial2020_dp.json"
DECENNIAL2020_DP_CLEAN = INTERIM_DATA_DIR / "decennial2020_dp_clean.parquet"

####
# CENSUS VARIABLES
####

CONFIG_DIR = Path("config")

def load_yaml(name: str):
    """
    Given the name of a YAML file in config/, returns the dict, list, or scalar associated with the data in the file.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
    
CENSUS_VARS = load_yaml("census_variables")