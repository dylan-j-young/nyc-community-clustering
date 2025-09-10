## PREAMBLE
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree

import os
import random
from typing import Optional

from src import config

## Helper functions for other scripts and notebooks
def clean_geoid(geoid_raw: str | pd.Series) -> str | pd.Series:
    """
    Inputs a raw GEOID, which can either be 11 or 20 digits,
    and returns the 11 digit version (also as a str or pd.Series).

    Parameters
    ----------
    geoid_raw : str or pd.Series
        Input GEOID(s) expected as string of digits, either 11 or
        20 digits.
    
    Returns
    -------
    geoid : str or pd.Series
        GEOID(s) standardized to 11-character string(s).
    """
    if isinstance(geoid_raw, str):
        geoid = geoid_raw[-11:]
    elif isinstance(geoid_raw, pd.Series):
        geoid = geoid_raw.str.slice(-11)
    else:
        raise TypeError("Expected str or pd.Series.")
    
    return( geoid )

def get_fips(geoid: str | pd.Series) -> str | pd.Series:
    """ 
    Reads a GEOID and returns its 5-digit FIPS code.

    Parameters
    ----------
    geoid : str or pd.Series
        Input GEOID(s) standardized to 11-digit string(s)
    
    Returns
    -------
    fips : str or pd.Series
        County FIPS code standardized to 5-digit string(s)
    """

    if isinstance(geoid, str):
        fips = geoid[:5]
    elif isinstance(geoid, pd.Series):
        fips = geoid.str.slice(0,5)
    else:
        raise TypeError("Expected str or pd.Series.")
    
    return( fips )


def get_borough(geoid: str | pd.Series) -> str | pd.Series:
    """
    Reads a GEOID's county FIPS code and returns a string representing
    the name of the proper NYC borough.

    Parameters
    ----------
    geoid : str or pd.Series
        Input GEOID(s) standardized to 11-digit string(s)
    
    Returns
    -------
    borough : str or pd.Series
        County FIPS code standardize to 5-digit string(s)
    """

    # Extract FIPS code
    fips = get_fips(geoid)

    if isinstance(fips, str):
        borough = config.FIPS_DICT[fips]
    elif isinstance(fips, pd.Series):
        borough = fips.map( config.FIPS_DICT )
    else:
        raise TypeError("Expected str or pd.Series.")
    
    return( borough )

def get_local_subregion(gdf : gpd.GeoDataFrame,
                        k : int = 100, 
                        borough : str = "random",
                        tract_name : str = "random",
                        seed : Optional[int] = None
                        ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """ 
    Pulls a local subregion of k nearest census tracts of a chosen borough to a chosen center tract. By default, the borough and tract are randomly picked. This function uses scipy.spatial.KDTree to generate the subset.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input dataset (assumed of NYC census tracts)
    
    k : int, optional
        The number of tracts to pull into a subregion. Defualt is 100.
    
    borough : str, optional
        The name of the borough to work in, as written in src.config.BOROUGHS, or "random", which asks for one to be randomly chosen. Defualt is "random".
    
    tract_name : str, optional
        The standardized code for the census tract (e.g. "501", "449.01") or "random", which asks for one to be randomly chosen. Default is "random".
    
    seed : int or None, optional
        A random seed used to initialize RNG instances. Default is none.

    Returns
    -------
    local_tracts : GeoDataFrame
        The chosen k-tract subregion.

    borough_tracts : GeoDataFrame
        The census tracts only in the chosen borough.
    """
    # Initialize RNGs
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    # Pull borough tracts
    if borough == "random":
        # Get random borough
        borough = rng_py.choice(config.BOROUGHS)
    borough_tracts = gdf.loc[gdf["BOROUGH"] == borough]

    # Pull the desired tract
    if tract_name == "random":
        # Get a random tract
        tract0 = borough_tracts.sample(n=1, random_state=rng_np)
    else:
        tract0 = borough_tracts.loc[
            borough_tracts["TRACT"] == tract_name
        ].head(1)
    local_tracts = None

    # Get coordinates of each tract and the center tract
    points = borough_tracts[["LAT","LONG"]].values # shape = (n, 2)
    point0 = tract0[["LAT","LONG"]]

    # Use a kd-tree to find nearest neighbors
    tree = KDTree(points)
    __, indices = tree.query(point0, k)

    # Get GEOIDs of nearest neighbors
    geoids = borough_tracts.index # shape = (n, 1)
    local_ids = geoids[indices.flatten()]

    # Use GEOIDs to pull the correct df subset
    local_tracts = borough_tracts.loc[local_ids]

    return( local_tracts, borough_tracts )