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

    logging.info("Transforming tract geometries...")
    transform_tracts()

    logging.info("Transforming 2020 Demographic Profile data...")
    transform_decennial2020()

    logging.info("Transforming 2023 ACS data...")
    transform_acs2023()    

    # These do nothing, just pushes tables through transform
    logging.info("Transforming Neighborhood Tabulation Areas...")
    transform_ntas()    

    logging.info("Transforming 2017 Zillow neighborhood boundaries...")
    transform_zillow()    

def transform_tracts():
    """
    Perform the following transformations:
    1. Remove water area from tract geometries
    2. Remove fictitious adjacencies through docks of MH tracts in BK and Q
    3. Drop low-population rows
    """
    gdf = database.query_db("SELECT * FROM clean_tracts;")
    gdf_water = database.query_db("SELECT * FROM clean_areawater;")

    gdf = _simplify_rename_tracts(gdf)
    gdf = _remove_water(gdf, gdf_water)
    gdf = gdf.set_index("geoid")
    gdf = _remove_docks(gdf)
    gdf = gdf.reset_index()

    gdf = _filter_geoids(gdf,
        _get_low_population_geoids()
    )

    database.save_to_db(gdf, "analysis_tracts", spatial=True)

def transform_decennial2020():
    """
    Perform the following transformations:
    1. Restrict columns to those provided in config/census_variables.yaml
    2. Rename columns according to config/census_variables.yaml
    3. Drop low-population rows
    """
    df = database.query_db("SELECT * FROM clean_decennial2020;")
    
    # Keep only the columns listed in CENSUS_VARS
    # (Only pure counts, removing redundant columns)
    df = df.set_index("geoid")
    census_var_renames = {
        key.lower(): value \
        for key, value in config.CENSUS_VARS["2020_census_dp"].items()
    }
    cols_to_keep = list( census_var_renames.keys() )
    df = df[df.columns.intersection(cols_to_keep)]
    df = df.reset_index()

    # Rename columns
    df = df.rename( columns = census_var_renames )

    # Drop low-pop rows
    df = _filter_geoids(df,
        _get_low_population_geoids()
    )

    database.save_to_db(df, "analysis_decennial2020")

def transform_acs2023():
    """
    Perform the following transformations:
    1. Rename ACS columns according to config/census_variables.yaml
    2. Drop columns representing margin of error
    3. Drop low-population rows
    4. *TODO* Geographic interpolation of select columns
    """
    df = database.query_db("SELECT * FROM clean_acs2023;")

    # Rename columns
    census_var_renames = {
        key.lower(): value \
        for key, value in config.CENSUS_VARS["2023_acs_5yr_select"].items()
    }
    df = df.rename( columns = census_var_renames )

    # Drop columns with margins of error
    all_cols = df.columns.to_numpy()
    margin_cols = all_cols[[(col[:4] == "err_") for col in all_cols]]
    df = df.drop(columns=margin_cols)

    # Drop low-pop rows
    df = _filter_geoids(df,
        _get_low_population_geoids()
    )

    # TODO : Geographic interpolation
    
    database.save_to_db(df, "analysis_acs2023")

def transform_ntas():
    """Does nothing."""

    gdf = database.query_db("SELECT * FROM clean_ntas;")

    database.save_to_db(gdf, "analysis_ntas", spatial=True)

def transform_zillow():
    """Does nothing."""
    
    gdf = database.query_db("SELECT * FROM clean_zillow;")

    database.save_to_db(gdf, "analysis_zillow", spatial=True)

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
    
    return gdf

def _get_low_population_geoids():
    """Returns all geoids in New York State with low population or household number."""
    # hardcoded from census_variables.yaml
    pop_column = "DP1_0001C".lower() 
    household_column = "DP1_0113C".lower()

    df = database.query_db(f"""
        SELECT geoid FROM clean_decennial2020
        WHERE {pop_column} < 200
            OR {household_column} < 100
    """)
    return set(df['geoid'])

def _filter_geoids(df, exclude):
    return df[~df['geoid'].isin(exclude)]

# Filter rows (this can be SQL'd)