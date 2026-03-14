import numpy as np
import pandas as pd
import geopandas as gpd

import json
import logging
from typing import Sequence
from shapely.geometry import shape

from .. import config, database, utils

def run():
    logging.info("--- 3) TRANSFORM PHASE ---")

    logging.info("Generating analysis tract geometries...")
    transform_tracts()

def transform_tracts():
    gdf = database.query_db("SELECT * FROM clean_tracts;")
    gdf_water = database.query_db("SELECT * FROM clean_areawater;")

    gdf = _simplify_rename_tracts(gdf)
    gdf = _remove_water(gdf, gdf_water)
    gdf = gdf.set_index("geoid")
    gdf = _remove_docks(gdf)
    gdf = gdf.reset_index()

    database.save_to_db(gdf, "analysis_tracts", spatial=True)


def _remove_water(gdf, gdf_water):
    # Subtract areawater polygons from the census tracts
    # There should be 2324 tracts (-3 water-only tracts)
    gdf = gdf.overlay(gdf_water, how='difference')
    return( gdf )

def _simplify_rename_tracts(gdf):
    # Remove unnecessary columns
    gdf = gdf.drop(columns=["geoidfq", "mtfcc", "funcstat", "statefp", "countyfp", "tractce", "namelsad", "awater"])

    # Rename remaining columns to be more intuitive
    gdf = gdf.rename(columns={
        "name": "tract",
        "aland": "area",
        "intptlat": "lat",
        "intptlon": "long"
    })

    #  Add boroughs column
    boroughs = utils.get_borough(gdf["geoid"])
    gdf.insert(1, "borough", boroughs)

    # Remove census tracts (rows) not in the five boroughs
    gdf = gdf.dropna(subset=["borough"])

    return( gdf )

def _remove_docks(gdf):
    # Hard-coded problem tracts (could fix this algorithmically if necessary)
    dock_tract_ids = ["36061" + s for s in ["000900","000700","001502","000202","006200","008601"]]
    dock_tracts = gdf.loc[dock_tract_ids]

    # -- Explode out MultiPolygons and keep only largest chunks --
    dock_tracts_clipped = dock_tracts.explode().iloc[[0,3,5,7,9,11]]

    # -- Replace original tracts with clipped ones --
    for id in dock_tract_ids:
        gdf.loc[id] = dock_tracts_clipped.loc[id]
    
    return(gdf)

# Restrict columns (this can be a SQL thing)

#   # dp
    ## TODO : move to analysis cleaning
    # # Keep only the columns listed in CENSUS_VARS
    # # (Only pure counts, removing redundant columns)
    # census_var_renames = config.CENSUS_VARS["2020_census_dp"]
    # cols_to_keep = list( census_var_renames.keys() )
    # df = df[df.columns.intersection(cols_to_keep)]

    # # Rename columns
    # df = df.rename( columns = census_var_renames )

#   # acs
    ## TODO : move to analysis cleaning
    # # Rename columns
    # census_var_renames = config.CENSUS_VARS["2023_acs_5yr_select"]
    # df = df.rename( columns = census_var_renames )

# Filter rows (this can be SQL'd)

# Combine tables? Or not?


