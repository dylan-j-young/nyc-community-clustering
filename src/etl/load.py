import numpy as np
import pandas as pd
import geopandas as gpd

import json
import logging
from typing import Sequence
from shapely.geometry import shape

from .. import config, database, utils

def run():
    """Run the full load sequence, which reads in raw JSON files and shapefiles from the extract phase and loads into bronze tables in the SQLite database."""

    logging.info("--- 2) LOAD PHASE ---")

    logging.info("Cleaning 2020 Demographic Profile...")
    clean_decennial2020()

    logging.info("Cleaning 2023 5-Year ACS...")
    clean_acs2023()

    logging.info("Cleaning tract geographies...")
    # clean_nyc_tracts()
    clean_tracts_draft()
    clean_areawater_draft()

    logging.info("Cleaning Neighborhood Tabulation Areas...")
    clean_nyc_NTAs()

    logging.info("Cleaning 2017 Zillow neighborhood boundaries...")
    clean_zillow_nbds()

def clean_raw_types(df):
    """
    Smart type conversion before saving to the SQL database.
    """
    # 1. Identify ID columns (these stay as strings)
    id_cols = ['GEOID']
    
    # 2. Get the list of columns to try and convert
    cols_to_convert = [c for c in df.columns if c not in id_cols]
    
    # 3. Convert to numeric if not an ID column
    for col in cols_to_convert:
        # errors='coerce' turns weird stuff into NaN (NULL)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 4. Explicitly ensure IDs are strings
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return(df)

# DEPRECATED
def clean_nyc_tracts():
    """
    Cleans NYC census tracts using clean_tracts, and writes the cleaned GeoDataFrame to the table "clean_tracts" in the SQLite database.
    
    Parameters
    ----------
    
    Returns
    -------
    """
    gdf = clean_tracts(config.TRACTS_RAW, config.AREAWATER)

    # Write to SQLite table
    database.save_to_db(gdf, "clean_tracts", spatial=True)

# DEPRECATED
def clean_tracts(input_shapefile, areawater_shapefile):
    """
    Given valid TIGER/Line census tract shapefiles, loads a GeoDataFrame using geopandas, clean its entries, and return it.

    Parameters
    ----------
    input_shapefile : str or Path
        Location of the .shp file for the desired tracts. Note that other auxiliary files (.shx, .dbf, .prj) are required in the same directory for the shapefile to successfully load.

    areawater_shapefile : str or Path (or list thereof), optional
        Location of the .shp file or .shp files for water areas to subtract from the census tract geometries. Note that other auxiliary files (.shx, .dbf, .prj) are required in the same directory for each shapefile to successfully load. Default is None.
    
    Returns
    -------
    gdf : gpd.GeoDataFrame
        GeoDataFrame of census tracts in NYC with the columns:
        GEOID : str, 11-digit GEOID for tract
        borough : str, representing the name of the borough
        tract : str, Census tract number
        area : int64, land area of tract in square meters
        lat : str, latitude of the tract's internal point
        long : str, longitude of the tract's internal point
        geometry : Polygon, representing the tract in WGS84
    """

    # Load census tract shapefile
    gdf = gpd.read_file(input_shapefile)

    # Preprocessing: clean GEOIDs
    geoids = utils.clean_geoid(gdf["GEOID"])
    gdf["GEOID"] = geoids

    # Remove unnecessary columns
    gdf = gdf.drop(columns=["GEOIDFQ", "MTFCC", "FUNCSTAT", "STATEFP", "COUNTYFP", "TRACTCE", "NAMELSAD", "AWATER"])

    # Rename remaining columns to be more intuitive
    gdf = gdf.rename(columns={
        "NAME": "TRACT",
        "ALAND": "AREA",
        "INTPTLAT": "LAT",
        "INTPTLON": "LONG"
    })

    # Add new columns, and convert numeric columns to numbers
    boroughs = utils.get_borough(geoids)
    gdf.insert(1, "BOROUGH", boroughs)
    
    gdf[["LAT","LONG","AREA"]] = gdf[["LAT","LONG","AREA"]].apply(pd.to_numeric)

    # Remove census tracts (rows) not in the five boroughs
    gdf = gdf.dropna(subset=["BOROUGH"])

    # Convert coordinate reference to WGS84 
    gdf = gdf.to_crs(epsg=config.WGS84_EPSG)

    # Load in an areawater GeoDataFrame
    if not areawater_shapefile is None:
        if isinstance(areawater_shapefile, Sequence):
            # Multiple areawater shapefiles to combine
            gdfs_water = [
                gpd.read_file(f).to_crs(epsg=config.WGS84_EPSG) \
                for f in config.AREAWATER
            ]
            gdf_water = gpd.GeoDataFrame( pd.concat(gdfs_water) )
        else:
            # Just a single shapefile
            gdf_water = gpd.read_file(areawater_shapefile) \
                        .to_crs(epsg=config.WGS84_EPSG)

        # Subtract areawater polygons from the census tracts
        gdf = gdf.overlay(gdf_water, how='difference')

    # # Set index to GEOID
    # gdf = gdf.set_index("GEOID")

    return( gdf )

def clean_areawater_draft():
    # Multiple areawater shapefiles to combine
    gdfs_water = [
        gpd.read_file(f).to_crs(epsg=config.WGS84_EPSG) \
        for f in config.AREAWATER
    ]
    gdf_water = gpd.GeoDataFrame( pd.concat(gdfs_water) )

    # Write to SQLite table
    database.save_to_db(gdf_water, "clean_areawater", spatial=True)

def clean_tracts_draft():
    # Load census tract shapefile
    gdf = gpd.read_file(config.TRACTS_RAW)

    # Preprocessing: clean GEOIDs
    geoids = utils.clean_geoid(gdf["GEOID"])
    gdf["GEOID"] = geoids

    # Keep only tracts in the NYC five boroughs (this is a state-level dataset)
    # There should be 2327 of these
    gdf = gdf[
        (gdf["STATEFP"] + gdf["COUNTYFP"]).isin(config.FIPS_DICT)
    ]

    # Make lat/long coordinates numeric
    gdf[["INTPTLAT","INTPTLON"]] = gdf[["INTPTLAT","INTPTLON"]].apply(pd.to_numeric)

    # Convert coordinate reference to WGS84 
    gdf = gdf.to_crs(epsg=config.WGS84_EPSG)

    # Write to SQLite table
    database.save_to_db(gdf, "clean_tracts", spatial=True)

def clean_decennial2020():
    """
    Performs an initial cleaning of the 2020 DP data. Removes redundant columns and loads into the SQLite database as the table "clean_decennial2020".

    Parameters
    ----------
    
    Returns
    -------
    """
    # Load raw data from file and convert to a dataframe
    with open(config.DECENNIAL2020_DP_RAW, "r") as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data[1:], columns=raw_data[0])

    # Remove duplicate columns
    df = df.loc[:,~df.columns.duplicated()]

    # Clean GEO_ID, rename to GEOID, and set as index
    df["GEO_ID"] = utils.clean_geoid(df["GEO_ID"])
    df = df.rename(columns={"GEO_ID": "GEOID"})
    # df = df.set_index("GEOID")

    # Remove end columns that are redundant
    df = df.drop(columns=["NAME","state","county","tract"])

    # Convert non-ID columns to numeric
    df = clean_raw_types(df)

    # Export cleaned DataFrame to file
    database.save_to_db(df, "clean_decennial2020")

def clean_acs2023():
    """
    Performs an initial cleaning of the 2023 ACS 5yr data. Loads the data into the SQLite database as the table "clean_acs2023".

    Parameters
    ----------
    
    Returns
    -------
    """
    # Load raw data from file and convert to a dataframe
    with open(config.ACS5YR2023_RAW, "r") as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data[1:], columns=raw_data[0])

    # Clean GEO_ID, rename to GEOID, and set as index
    df["GEO_ID"] = utils.clean_geoid(df["GEO_ID"])
    df = df.rename(columns={"GEO_ID": "GEOID"})
    # df = df.set_index("GEOID")

    # Remove end columns that are redundant
    df = df.drop(columns=["state","county","tract"])
    
    # Convert strings of numbers to numbers
    # for col in df:
    #     df[col] = pd.to_numeric(df[col], errors="raise")
    df = clean_raw_types(df)

    # Label unfilled entries with nan
    df = df.replace(-888888888, np.nan)
    df = df.replace(-666666666, np.nan)
    df = df.replace(-222222222, np.nan)

    ## TODO: move to analysis cleaning
    # # Drop columns with margins of error
    # all_cols = df.columns.to_numpy()
    # margin_cols = all_cols[[(col[:4] == "err_") for col in all_cols]]
    # df = df.drop(columns=margin_cols)

    # Export cleaned DataFrame to file
    database.save_to_db(df, "clean_acs2023")

def clean_nyc_NTAs():
    """
    Load in JSON from fetch_nyc_NTAs() and convert it to a GeoDataFrame. Save to the table "clean_ntas" in the SQLite database.

    Parameters
    ----------
    
    Returns
    -------
    """
    # Read in JSON
    df = pd.read_json(config.NYC_NTAS_RAW)

    # Make a proper geometry column from provided GeoJSON column
    df = df.rename({"the_geom": "geometry"}, axis=1)
    df["geometry"] = df.apply(lambda row: shape(row["geometry"]), axis=1)

    # Initialize the GeoDataFrame and save
    gdf = gpd.GeoDataFrame(df, crs=config.WGS84_EPSG, geometry="geometry")
    database.save_to_db(gdf, "clean_ntas", spatial=True)

def clean_zillow_nbds():
    """
    Load in JSON from fetch_zillow_nbds() and convert it to a GeoDataFrame. Save to a table called "clean_zillow" in the SQLite table.

    Parameters
    ----------
    
    Returns
    -------
    """
    # Read in JSON
    import json
    with open(config.ZILLOW_RAW, 'r') as f:
        geojson = json.load(f)

    # Convert features into GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geojson, crs=config.WGS84_EPSG)

    # Export
    database.save_to_db(gdf, "clean_zillow", spatial=True)