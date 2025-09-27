import numpy as np
import pandas as pd
import matplotlib as mpl

import os
from pathlib import Path
import yaml

#### 
# GLOBAL SETTINGS
####

# Pandas settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

mpl.rc("hatch", color="k", linewidth=0.5)

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

# Distance
METERS_PER_MILE = 1609.344
# MEAN_EARTH_RADIUS = 6371 * 1000 # m
# KM = 1000 # m
# DEGREE_TO_M = np.pi/180 * MEAN_EARTH_RADIUS

####
# DATA PATHS
####

# Model directories
MODEL_DIR = Path("models")

# Data directories
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Feature selections for clustering
FEATURES_ALL = PROCESSED_DATA_DIR / "features_all.parquet"
FEATURES_CUT1 = PROCESSED_DATA_DIR / "features_cut1.parquet"
FEATURES_CUT2 = PROCESSED_DATA_DIR / "features_cut2.parquet"
FEATURES_CUT3 = PROCESSED_DATA_DIR / "features_cut3.parquet"

# Shapefiles: NYC census tracts
RAW_SHAPEFILES_DIR = RAW_DATA_DIR / "shapefiles"
TRACTS_RAW = (
    RAW_SHAPEFILES_DIR 
    / "tl_2024_36_tract" 
    / "tl_2024_36_tract.shp"
)
TRACTS_CLEAN = INTERIM_DATA_DIR / "tracts_clean.parquet"
TRACTS_HIGHPOP = INTERIM_DATA_DIR / "tracts_highpop.parquet"

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
DECENNIAL2020_DP_HIGHPOP = INTERIM_DATA_DIR / "decennial2020_dp_highpop.parquet"

# 2023 ACS (5-year)
ACS5YR2023_RAW = RAW_DATA_DIR / "acs5yr2023.json"
ACS5YR2023_CLEAN = INTERIM_DATA_DIR / "acs5yr2023_clean.parquet"
ACS5YR2023_HIGHPOP = INTERIM_DATA_DIR / "acs5yr2023_highpop.parquet"

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