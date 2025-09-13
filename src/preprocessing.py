## PREAMBLE
import numpy as np
import pandas as pd
import geopandas as gpd

import os
import requests
import json
from pathlib import Path
from typing import Optional, Sequence
from dotenv import load_dotenv

from src import config, utils

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

    # Set index to GEOID
    gdf = gdf.set_index("GEOID")

    # Export cleaned GeoDataFrame to output_path
    gdf.to_parquet(output_path)

    return(gdf)

def fetch_2020_demographic_profile():
    """ 
    Calls the US Census API with a GET query for the 2020 Census Demographic Profile, for each census tract in NYC. Saves the returned data as a JSON file in config.DECENNIAL2020_DP_RAW.

    Parameters
    ----------
    
    Returns
    -------
    raw_data : list
        The raw data returned from the GET request and saved to file
    """
    # Get API key from .env (user-specific local secrets)
    load_dotenv()
    API_KEY = os.getenv("CENSUS_API_KEY")
    
    # Parameters of Census API query
    year = 2020
    source = "dec" # Decennial Census
    dataset = "dp" # Demographic Profile
    cols = ",".join(["GEO_ID","group(DP1)"]) # GEO_ID and all of the DP
    borough_fips = list(config.FIPS_DICT.keys())
    borough_codes = [fips[2:] for fips in borough_fips]
    boroughs = ",".join(borough_codes)
    state = "36" # NY
    tracts = "*" # all

    # Construct URL query
    url = f"https://api.census.gov/data/{year}/{source}/{dataset}" \
        + f"?get={cols}" \
        + f"&for=tract:{tracts}" \
        + f"&in=county:{boroughs}" \
        + f"&in=state:{state}" \
        + f"&key={API_KEY}"

    # Send GET request and retrieve a JSON-formatted response
    response = requests.get(url)
    print(f"GET request status: {response.status_code}")
    raw_data = response.json()

    # Write to file
    with open(config.DECENNIAL2020_DP_RAW, "w") as f:
        json.dump(raw_data, f)
    
    return(raw_data)

def clean_2020_demographic_profile():
    """
    Performs an initial cleaning of the 2020 DP data. Selects out only pure counts (not percentages or annotations) and removes redundant columns.

    Parameters
    ----------
    
    Returns
    -------
    df_clean : pd.DataFrame
        The cleaned dataframe saved to file.
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
    df = df.set_index("GEOID")

    # Remove end columns that are redundant
    df = df.drop(columns=["NAME","state","county","tract"])

    # Keep only the columns listed in CENSUS_VARS
    # (Only pure counts, removing redundant columns)
    census_var_renames = config.CENSUS_VARS["2020_census_dp"]
    cols_to_keep = list( census_var_renames.keys() )
    df = df[df.columns.intersection(cols_to_keep)]

    # Rename columns
    df = df.rename( columns = census_var_renames )

    # Convert strings of numbers to numbers
    for col in df:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Remove rows with no people
    df = df[df["pop"] > 0]

    # Export cleaned DataFrame to file
    df.to_parquet(config.DECENNIAL2020_DP_CLEAN)

    return( df )

if __name__ == "__main__":
    # # Test clean_tracts()
    # gdf = clean_tracts(config.TRACTS_RAW, config.TRACTS_CLEAN,
    #                    areawater_shapefile=config.AREAWATER)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 8))
    # gdf.plot(ax=ax,
    #          facecolor="none",
    #          edgecolor="black",
    #          linewidth=1
    #         )
    # plt.show()

    # # Test fetch_2020_demographic_profile()
    # decennial2020_dp_raw = fetch_2020_demographic_profile()

    # # Test clean_2020_demographic_profile()
    # decennial2020_dp_clean = clean_2020_demographic_profile()

    pass

def remove_unpopulated_tracts():
    """
    Reads in the NYC census tracts as a GeoDataFrame and cleaned data from the 2020 Census Demographic Profile. Keeps only tracts that are also in the demographic profile and have a nonzero number of people.

    Parameters
    ----------

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
    gdf = gpd.read_parquet(config.TRACTS_CLEAN)
    df = pd.read_parquet(config.DECENNIAL2020_DP_CLEAN)


    gdf = gdf[ gdf.index.isin(df.index) ]
    
    # Export cleaned GeoDataFrame
    gdf.to_parquet(config.TRACTS_CLEAN)

    return( gdf )