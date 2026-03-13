## PREAMBLE
import os
import logging
import requests
import shutil
import zipfile
import json
from dotenv import load_dotenv
from sodapy import Socrata

from .. import config

def run():
    """Run the full extract sequence, which obtains raw data and dumps as JSON and shapefiles."""

    logging.info("--- 1) EXTRACT PHASE ---")
    
    logging.info("Fetching 2020 Demographic Profile...")
    fetch_2020_demographic_profile()

    logging.info("Fetching 2023 5-Year ACS...")
    fetch_2023_acs_5yr_select()

    logging.info("Fetching shapefiles...")
    fetch_shapefiles()

    logging.info("Fetching Neighborhood Tabulation Areas...")
    fetch_nyc_NTAs()

    logging.info("Fetching 2017 Zillow neighborhood boundaries...")
    fetch_zillow_nbds()

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

if __name__ == "__main__":
    pass