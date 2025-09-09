## PREAMBLE
import numpy as np
import pandas as pd
import geopandas as gpd

import os
from pathlib import Path

from src import config
from src.utils import clean_geoid, get_borough

# Testbed in scratch_01 notebook
def clean_tracts(input_shapefile: str | Path, 
                 output_path: str | Path) -> gpd.GeoDataFrame:
    """
    Given valid TIGER/Line census tract shapefiles, loads a GeoDataFrame using geopandas, clean its entries, and return it. Also exports the cleaned GeoDataFrame to a parquet file (pyarrow or equivalent library required).

    Parameters
    ----------
    input_shapefile : str or Path
        Location of the .shp file for the desired tracts. Note that other auxiliary files (.shx, .dbf, .prj) are required in the same directory for the shapefile to successfully load.

    output_path : str or Path
        Desired output location for the parquet file.
    
    Returns
    -------
    gdf : gpd.GeoDataFrame
        GeoDataFrame of census tracts in NYC with the columns:
        GEOID : str, 11-digit GEOID for tract
        BOROUGH : str, representing the name of the borough
        TRACT : str, Census tract number
        AREA : int64, land area of tract in square meters
        LAT : str, latitude of the tract's internal point
        LONG : str, longitude of the tract's internal point
        geometry : Polygon, representing the tract in WGS84
    """

    # Load census tract shapefile
    gdf = gpd.read_file(input_shapefile)

    # Preprocessing: clean GEOIDs
    geoids = clean_geoid(gdf["GEOID"])
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

    # Add new columns
    boroughs = get_borough(geoids)
    gdf.insert(1, "BOROUGH", boroughs)

    # Remove census tracts (rows) not in the five boroughs
    gdf = gdf.dropna(subset=["BOROUGH"]).reset_index(drop=True)

    # Convert coordinate reference to WGS84 
    gdf = gdf.to_crs(epsg=config.WGS84_EPSG)

    # Export cleaned GeoDataFrame to output_path
    gdf.to_parquet(output_path)

    return(gdf)

