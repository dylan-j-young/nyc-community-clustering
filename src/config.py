import pandas as pd
import os

#### 
# GLOBAL SETTINGS
####

# Pandas settings
pd.set_option("display.max_columns", None)

####
# DATA PATHS
####

# Data directories
RAW_DATA_DIR = "data/raw"
INTERIM_DATA_DIR = "data/interim"
PROCESSED_DATA_DIR = "data/processed"

# Shapefiles
CTRACT_SHAPEFILE = os.path.join(RAW_DATA_DIR, "shapefiles", "census-tracts", "tl_2024_36_tract.shp")

# Tables
P5_TABLE = os.path.join(RAW_DATA_DIR, "P5_race-hispanic", "DECENNIALDHC2020.P5-Data.csv")

#### 
# GEOGRAPHY
####

# FIPS county codes
FIPS_DICT = {
    "36005": "The Bronx",
    "36047": "Brooklyn",
    "36061": "Manhattan",
    "36081": "Queens",
    "36085": "Staten Island"
}