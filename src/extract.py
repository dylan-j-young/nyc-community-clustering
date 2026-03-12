## PREAMBLE
import numpy as np
import pandas as pd
import geopandas as gpd

import os
import logging
import requests
import shutil
import zipfile
import json
import sqlite3
from pathlib import Path
from typing import Optional, Sequence
from dotenv import load_dotenv
from sodapy import Socrata
from shapely.geometry import shape

from src import config, utils

def save_to_db(df, table_name, if_exists="replace", spatial=False):
    """
    Saves a dataframe to the raw SQLite database.
    """
    if spatial == False:
        conn = sqlite3.connect(config.DATABASE_DIR)
        try:
            # if_exists="replace" ensures we don't add duplicates
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            logging.info(f"Successfully wrote {len(df)} rows to table: {table_name}")
        except Exception as e:
            logging.error(f"Failed to write to table {table_name}: {e}")
            raise
        finally:
            conn.close()
    else:
        # Enforce CRS
        if df.crs is None:
            df = df.set_crs(config.WGS84_EPSG)
        else:
            df = df.to_crs(config.WGS84_EPSG)

        # Write to file
        df.to_file(config.DATABASE_DIR,
                   driver="SQLite",
                   spatialite=True,
                   layer=table_name)
        
def query_db(sql_query):
    df = gpd.read_file(config.DATABASE_DIR, sql=sql_query)

    if isinstance(df, gpd.GeoDataFrame):
        # Reattach CRS
        df = df.set_crs(config.WGS84_EPSG)

    return( df )

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

def fetch_shapefiles(timeout=300):
    """
    Get 2023 shapefiles for NYC census tracts and areawater geometries.

    Parameters
    ----------
    timeout : float, optional
        Time in seconds to wait for a response from requests.get(url). Default is 300.
    
    Returns
    -------
    """

    fips_codes = list( config.FIPS_DICT.keys() )
    state = fips_codes[0][:2]

    # URLs to pull 2023 shapefiles from
    urls = [config.TRACTS_URL] + config.AREAWATER_URLS

    for url in urls:
        zip_path = config.SHAPEFILES_DIR / os.path.basename(url)
        extract_dir = os.path.splitext(zip_path)[0]

        # Send GET request and retrieve a JSON-formatted response
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status() # Turns 4xx and 5xx errors into exceptions
        except Exception as e:
            logging.error(f"Network error: {e}")
            raise
        
        # No error in GET request
        logging.info(f"GET request for {os.path.basename(url)} succeeded")

        # Write zip file to config.RAW_DATA_DIR
        with open(zip_path, "wb") as f:
            f.write(response.content)

        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)  # delete folder and contents
            logging.info(f"Extract directory {extract_dir} already exists. Deleting and replacing...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

def clean_tracts(input_shapefile: str | Path,
                 output_path: str | Path,
                 areawater_shapefile: 
                    Optional[str | Path | Sequence[str|Path]] = None
                ) -> gpd.GeoDataFrame:
    """
    Given valid TIGER/Line census tract shapefiles, loads a GeoDataFrame using geopandas, clean its entries, and return it. Also writes the cleaned GeoDataFrame to the table "tracts" in the SQLite database.

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
        geoid : str, 11-digit GEOID for tract
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

    # Set index to GEOID
    gdf = gdf.set_index("GEOID")

    # Export cleaned GeoDataFrame to output_path
    save_to_db(gdf, "tracts", spatial=True)

    return(gdf)

def fetch_2020_demographic_profile():
    """ 
    Calls the US Census API with a GET query for the 2020 Census Demographic Profile, for each census tract in NYC. Saves the returned data as raw JSON in config.DECENNIAL2020_DP_RAW.

    Parameters
    ----------
    
    Returns
    -------
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
    try:
        response = requests.get(url)
        response.raise_for_status() # Turns 4xx and 5xx errors into exceptions
    except Exception as e:
        logging.error(f"Network error: {e}")
        raise
    else:
        # No error in GET request
        logging.info("GET request succeeded")
        raw_data = response.json()
    
        # Write to file
        with open(config.DECENNIAL2020_DP_RAW, "w") as f:
            json.dump(raw_data, f)

        # # Write to SQLite database
        # # # --- Clean up for SQL ---
        # df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        # # # Remove duplicate columns
        # # df = df.loc[:,~df.columns.duplicated()]

        # # # Remove end columns that are redundant in other tables
        # # df = df.drop(columns=["NAME","state","county","tract"])

        # # # Clean GEO_ID and rename to GEOID
        # # df["GEO_ID"] = utils.clean_geoid(df["GEO_ID"])
        # # df = df.rename(columns={"GEO_ID": "GEOID"})

        # # # Convert non-ID columns to numeric
        # # df = clean_raw_types(df)

        # # Save to SQLite database
        # save_to_db(df, "raw_decennial2020")
        # logging.info("Saved data to table raw_decennial2020")

def fetch_2023_acs_5yr_select():
    """ 
    Calls the US Census API with a GET query for the 2023 ACS (5-year), grabbing only the variables specified in config.CENSUS_VARS, for each census tract in NYC. Saves the returned data as a JSON file in config.ACS5YR2023_RAW.

    Parameters
    ----------
    
    Returns
    -------
    """
    # Get API key from .env (user-specific local secrets)
    load_dotenv()
    API_KEY = os.getenv("CENSUS_API_KEY")
    
    # Parameters of Census API query
    year = 2023
    source = "acs/acs5" # American Community Survey
    dataset = "profile" # Demographic Profile
    vars_to_get = list( config.CENSUS_VARS["2023_acs_5yr_select"].keys() )
    cols = ",".join(["GEO_ID"] + vars_to_get) # GEO_ID and all of the DP
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
    try:
        response = requests.get(url)
        response.raise_for_status() # Turns 4xx and 5xx errors into exceptions
    except Exception as e:
        logging.error(f"Network error: {e}")
        raise
    else:
        # No error in GET request
        logging.info("GET request succeeded")
        raw_data = response.json()
    
        # Write to file
        with open(config.ACS5YR2023_RAW, "w") as f:
            json.dump(raw_data, f)


def initial_clean_2020_demographic_profile():
    """
    Performs an initial cleaning of the 2020 DP data. Selects out only pure counts (not percentages or annotations) and removes redundant columns.

    Parameters
    ----------
    
    Returns
    -------
    df : pd.DataFrame
        The cleaned dataframe saved to the SQLite table.
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

    # Export cleaned DataFrame to file
    save_to_db(df, "decennial2020_dp")

    return( df )

def initial_clean_2023_acs_5yr_select():
    """
    Performs an initial cleaning of the 2023 ACS 5yr data.

    Parameters
    ----------
    
    Returns
    -------
    df : pd.DataFrame
        The cleaned dataframe saved to the SQLite table.
    """
    # Load raw data from file and convert to a dataframe
    with open(config.ACS5YR2023_RAW, "r") as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data[1:], columns=raw_data[0])

    # Clean GEO_ID, rename to GEOID, and set as index
    df["GEO_ID"] = utils.clean_geoid(df["GEO_ID"])
    df = df.rename(columns={"GEO_ID": "GEOID"})
    df = df.set_index("GEOID")

    # Remove end columns that are redundant
    df = df.drop(columns=["state","county","tract"])

    # Rename columns
    census_var_renames = config.CENSUS_VARS["2023_acs_5yr_select"]
    df = df.rename( columns = census_var_renames )

    # Convert strings of numbers to numbers
    for col in df:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Drop columns with margins of error
    all_cols = df.columns.to_numpy()
    margin_cols = all_cols[[(col[:4] == "err_") for col in all_cols]]
    df = df.drop(columns=margin_cols)

    # Label unfilled entries with nan
    df = df.replace(-888888888, np.nan)
    df = df.replace(-666666666, np.nan)

    # Export cleaned DataFrame to file
    save_to_db(df, "acs5yr2023")

    return( df )

def fetch_nyc_NTAs():
    """
    Get 2020 Neighborhood Tabulation Areas (NTAs) from data.cityofnewyork.us for reference. Save raw JSON at the location specified in config.NYC_NTAS_RAW.

    Parameters
    ----------
    
    Returns
    -------
    """

    # Unauthenticated client only works with public data sets. Note 'None'
    # in place of application token, and no username or password:
    client = Socrata("data.cityofnewyork.us", None)

    # Example authenticated client (needed for non-public datasets):
    # client = Socrata(data.cityofnewyork.us,
    #                  MyAppToken,
    #                  username="user@example.com",
    #                  password="AFakePassword")

    # First 2000 results, returned as JSON from API / converted to Python list of
    # dictionaries by sodapy.
    results = client.get("9nt8-h7nd", limit=2000)

    # Write to file
    import json
    with open(config.NYC_NTAS_RAW, "w") as f:
        json.dump(results, f)

def clean_nyc_NTAs():
    """
    Load in JSON from fetch_nyc_NTAs() and convert it to a GeoDataFrame. Save at the location specified by config.NYC_NTAS_CLEAN.

    Parameters
    ----------
    
    Returns
    -------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame constructed from the raw data, which was saved to file.
    """
    # Read in JSON
    df = pd.read_json(config.NYC_NTAS_RAW)

    # Make a proper geometry column from provided GeoJSON column
    df = df.rename({"the_geom": "geometry"}, axis=1)
    df["geometry"] = df.apply(lambda row: shape(row["geometry"]), axis=1)

    # Initialize the GeoDataFrame and save
    gdf = gpd.GeoDataFrame(df, crs=config.WGS84_EPSG, geometry="geometry")
    gdf.to_parquet(config.NYC_NTAS_CLEAN)
    # save_to_db(gdf, "ntas", spatial=True)

    return(gdf)

def fetch_zillow_nbds():
    """
    Get 2017 Zillow neighborhood boundaries from data.cityofnewyork.us for reference. Save raw JSON at the location specified in config.NYC_NTAS_RAW.

    Parameters
    ----------
    
    Returns
    -------
    """
    # REST server URL
    BASE_URL = "https://gispub.epa.gov/arcgis/rest/services/OEI/Zillow_Neighborhoods/MapServer/0/query"

    # where clause for request
    counties = ["Bronx", "Kings", "New York", "Queens", "Richmond"]
    where_clause = (
        "State = 'NY' AND "
        "County IN ({})".format(
            ",".join(f"'{c}'" for c in counties)
        )
    )

    # Make request
    features = []
    offset = 0
    page_size = 1000
    while True:
        params = {
            "where": where_clause,
            "outFields": "*",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size
        }

        # Send GET request and retrieve a JSON-formatted response
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status() # Turns 4xx and 5xx errors into exceptions
        except Exception as e:
            logging.error(f"Network error: {e}")
            raise

        logging.info("GET request succeeded")
        data = response.json()

        batch = data.get("features", [])
        if not batch:
            break

        features.extend(batch)
        offset += page_size

    # GeoJSON object
    results = {
        "type": "FeatureCollection",
        "features": features
    }

    # Write to file
    import json
    with open(config.ZILLOW_RAW, "w") as f:
        json.dump(results, f)

def clean_zillow_nbds():
    """
    Load in JSON from fetch_zillow_nbds() and convert it to a GeoDataFrame. Save to a table called "zillow_nbds" in the SQLite table.

    Parameters
    ----------
    
    Returns
    -------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame constructed from the raw data, which was saved to file.
    """
    # Read in JSON
    import json
    with open(config.ZILLOW_RAW, 'r') as f:
        geojson = json.load(f)

    # Convert features into GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geojson, crs=config.WGS84_EPSG)

    # Export
    save_to_db(gdf, "zillow_nbds", spatial=True)

    return(gdf)

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
    # decennial2020_dp_clean = initial_clean_2020_demographic_profile()

    pass