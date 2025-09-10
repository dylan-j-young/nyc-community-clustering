## PREAMBLE
import numpy as np
import pandas as pd
import geopandas as gpd

import os
from pathlib import Path
from typing import Optional, Sequence

from src import config
from src.utils import clean_geoid, get_borough

# Testbed in scratch_01 notebook
def clean_tracts(input_shapefile: str | Path,
                 output_path: str | Path,
                 areawater_shapefile: 
                    Optional[str | Path | Sequence[str|Path]] = None
                ) -> gpd.GeoDataFrame:
    """
    Given valid TIGER/Line census tract shapefiles, loads a GeoDataFrame using geopandas, clean its entries, and return it. Also exports the cleaned GeoDataFrame to a parquet file (pyarrow or equivalent library required).

    Parameters
    ----------
    input_shapefile : str or Path
        Location of the .shp file for the desired tracts. Note that other auxiliary files (.shx, .dbf, .prj) are required in the same directory for the shapefile to successfully load.

    output_path : str or Path
        Desired output location for the parquet file.

    areawater_shapefile : str or Path (or list thereof), optional
        Location of the .shp file or .shp files for water areas to subtract from the census tract geometries. Note that other auxiliary files (.shx, .dbf, .prj) are required in the same directory for each shapefile to successfully load. Default is None.
    
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

    # Add new columns, and convert numeric columns to numbers
    boroughs = get_borough(geoids)
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

    # Set index to GEOID
    gdf = gdf.set_index("GEOID")

    # Export cleaned GeoDataFrame to output_path
    gdf.to_parquet(output_path)

    return(gdf)

if __name__ == "__main__":
    # Quick test when running the script directly
    gdf = clean_tracts(config.TRACTS_RAW, config.TRACTS_CLEAN,
                       areawater_shapefile=config.AREAWATER)
    
    # Check that water was subtracted properly
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax,
             facecolor="none",
             edgecolor="black",
             linewidth=1
            )
    plt.show()