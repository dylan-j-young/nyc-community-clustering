import pandas as pd
import os

#### 
# GLOBAL SETTINGS
####

# Pandas settings
pd.set_option("display.max_columns", None)

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

####
# DATA PATHS
####

# Data directories
RAW_DATA_DIR = "data/raw"
INTERIM_DATA_DIR = "data/interim"
PROCESSED_DATA_DIR = "data/processed"

# Shapefiles: NYC census tracts (land and water)
TRACT_SHAPEFILE = os.path.join(RAW_DATA_DIR, "shapefiles", "tl_2024_36_tract", "tl_2024_36_tract.shp")

# Shapefiles: Water area polygons for each county/borough
AREAWATER_SHAPEFILES = [
    os.path.join(
        RAW_DATA_DIR, "shapefiles", f"tl_2024_{a}_areawater",
        f"tl_2024_{a}_areawater.shp"
    )
    for a in FIPS_DICT.keys()
]

# Tables
P5_TABLE = os.path.join(RAW_DATA_DIR, "P5_race-hispanic", "DECENNIALDHC2020.P5-Data.csv")