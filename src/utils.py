## PREAMBLE
import numpy as np
import pandas as pd
import geopandas as gpd

import os

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
