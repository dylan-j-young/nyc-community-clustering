## PREAMBLE
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from libpysal.weights import Rook, Queen

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
        Input dataset (assumed to be NYC census tracts)
    
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

def CLR(X, pseudocount=1e-5):
    """ 
    Performs a centered log-ratio (CLR) transformation on an array-like which is assumed to represent compositional data (each row's entries sum to a a constant value, normalized to 1). 

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input compositional data (rows sum to a constant).

    pseudocount : float, optional
        Converts 0 values to this to avoid dividing by 0. Recommended to input a small value that is not much smaller than the smallest possible nonzero value. Default is 1e-5.
    
    Returns
    -------
    X_clr : np.ndarray, shape (n_samples, n_features)
        CLR-transformed data.
    """
    # Normalize by sum of first row
    total = X[0].sum()
    X = X / total

    # Clip with pseudocount and 1
    X = np.clip(X, pseudocount, 1)

    # Calculate geometric mean and CLR
    log_gm = np.mean(np.log(X), axis=1, keepdims=True)
    X_clr = np.log(X) - log_gm

    return(X_clr)

def contiguity_matrix(gdf, type="queen"):
    """ 
    Given a GeoDataFrame of polygons (e.g., census tracts), generates a contiguity matrix using the libpysal library.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input dataset. Must have a geometry column with polygon information.

    type : str, optional
        Type of contiguity to calculate ("rook" or "queen"). Default is "queen".
    
    Returns
    -------
    w : np.ndarray, shape (n_rows, n_rows)
        The contiguity matrix. wij = 1 if tract i and tract j are (rook- or queen-) contiguous and 0 otherwise, with diagonal entries 0.

    ids : np.ndarray, shape (n_rows)
        Array of indices corresponding to the rows and columns of the contiguity matrix.
    """
    # Generate matrix
    if type == "queen":
        W = Queen.from_dataframe(gdf, use_index=True)
    elif type == "rook":
        W = Rook.from_dataframe(gdf, use_index=True)
    else:
        raise
    
    # Clean up and return information
    w, ids = W.full()
    ids = np.array(ids).flatten()
    return( w, ids )

def normalized_neighbor_weights(gdf):
    """
    Given a GeoDataFrame of polygons with common boundaries, returns a weights matrix w, where w[i][j] is the length of the border between tracts i and j (integer-indexed) divided by the total length of common borders of tract i.

    Warning! While the absolute border length matrix is symmetric, this normalized matrix is NOT. This is because tracts i and j may have different total border lengths, so the same border might be a larger fraction of the total length for one over the other.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input dataset. Must have a geometry column with polygon information.
    
    Returns
    -------
    w : np.ndarray, shape (n_rows, n_rows)
        The weights matrix. wij = border_length(i,j) / total_common_border_length(i).

    ids : np.ndarray, shape (n_rows)
        Array of indices in the original GeoDataFrame corresponding to the rows and columns of the contiguity matrix.
    """
    # Get rook neighbors
    rook, ids = contiguity_matrix(gdf, type="rook")

    # Set up weights matrix
    n = rook.shape[0]
    w = np.zeros((n,n))
    index_of_id = {ids[i]: i for i in range(n)}

    # Iterate over polygons
    for i in range(n):
        # Geometry of polygon i
        geometry0 = gdf.loc[ids[i]]["geometry"]

        # Geometry of neighbors j of polygon i
        id_neighbors = ids[ rook[i] == 1 ]
        geometry_neighbors = gdf.loc[id_neighbors]["geometry"]

        # Must have at least one neighbor to attempt normalization
        if len(geometry_neighbors) > 0:
            # Calculate normalized border lengths
            borders = geometry0.intersection(geometry_neighbors)
            norm_border_lengths = borders.length / (borders.length.sum())

            # Plug in values into spot (i, jk) for the kth neighbor of i
            js = [index_of_id[id] for id in id_neighbors]
            for k, jk in enumerate(js):
                w[i][jk] = norm_border_lengths.iloc[k]
    
    return( w, ids )

def pretty_round(x, p):
    """ 
    Given a number x and number of significant digits p, returns a nicely rounded version of x. Only works on single numbers (not numpy arrays).
    """
    order_of_magnitude = round(np.ceil(np.log10(x)))
    y = x * 10**(-order_of_magnitude+p)
    x_prettyround = round(y) * 10**(order_of_magnitude-p)
    return( x_prettyround )

def interpolate_from_neighbors(gdf, col, remainder_fill=0, verbose=False):
    """ 
    Given a GeoDataFrame of polygons and an associated data column with nan values, fill in nan values using a weighted interpolation based on the lengths of borders with neighboring polygons.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input dataset. Must have a geometry column with polygon information.
    
    col : pd.Series
        A data column (Series) with GEOIDs correlated with the GeoDataFrame.

    remainder_fill : float or int, optional
        Value to fill any remaining nan entries after interpolation. Default is 0.
    
    verbose : bool, optional
        If verbose is True, the function outputs status messages about the interpolation. Default is False.
    
    Returns
    -------
    interpolated_col : pd.Series
        The data column with nan values interpolated or filled in.
    """
    # Get neighbor weights matrix and id lookup dict once
    neighbor_weights, ids = normalized_neighbor_weights(gdf)
    id_to_iloc = {ids[i]: i for i in range(len(gdf))}

    # Neighbor weights are potentially in a different order, so reorder col
    original_index = col.index
    col = col.reindex(ids)

    # Initial status message
    if verbose:
        nan_count = len( col[col.isna()] )
        total_count = len( col )
        print(f"Starting interpolation with {nan_count} NaN entries out of {total_count} total rows:")

    # Recursively call interpolation function
    threshold0, dthresh = 1.0, 0.05
    interpolated_col = _recurse_interpolation(col, neighbor_weights, 
                               id_to_iloc, threshold0, dthresh, verbose)
    
    # Check if there are any nans left
    remainder_nan_count = len(interpolated_col[interpolated_col.isna()])
    if remainder_nan_count > 0:
        # If so, fill remaining nan cells with remainder_fill and notify
        interpolated_col = interpolated_col.fillna(remainder_fill)
        
        if verbose:
            print(f"Unable to interpolate {remainder_nan_count} rows; filled with {remainder_fill} instead.")
    
    # Reorder col to match original
    interpolated_col = interpolated_col.reindex(original_index)
    return( interpolated_col )


def _recurse_interpolation(col, neighbor_weights,
                           id_to_iloc, threshold, dthresh, verbose):
    """ 
    A helper function for interpolate_from_neighbors(), not meant to be called separately.
    """

    # Get integer locations of all nan values in col
    nan_rows = col[col.isna()]
    ilocs_nan = [id_to_iloc[nan_idx] for nan_idx in nan_rows.index]

    # If no nans, return the col as-is
    if len(nan_rows) == 0:
        if verbose:
            print("Successfully interpolated all tracts!")
        return( col )

    # Remove weights of neighbors that are nan
    notnan_projector = col.notna().astype(int).to_numpy()[None,:]
    notnan_neighbor_ws = neighbor_weights * notnan_projector

    # Fraction of each border that is not nan (as a numpy array)
    frac_border_notnan = notnan_neighbor_ws.sum(axis=1)

    # Get all nan rows and sort descending by frac_border_notnan
    rows_in_interpolation_order = pd.Series(
        index = nan_rows.index,
        data = frac_border_notnan[ilocs_nan]
    ).sort_values(ascending=False)

    # Interpolate rows one by one in order of frac_border_notnan
    col_nanremoved = col.fillna(0)
    for k, row_k in enumerate(rows_in_interpolation_order):
        # Stop interpolating when frac_border_notnan goes below threshold
        if (row_k < threshold) or (row_k <= 0):
            break

        # Get integer location of the kth row
        geoid = rows_in_interpolation_order.index[k]
        iloc = id_to_iloc[geoid]
        
        # Interpolate using an average weighted by notnan_neighbor_ws
        interpolated_value = np.average(
            col_nanremoved, weights=notnan_neighbor_ws[iloc]
        )
        col[geoid] = interpolated_value

        # Increment k one more time if all rows successfully filled
        if (k+1 == len(rows_in_interpolation_order)):
            k += 1

    # If no more tracts can be interpolated at threshold=0, give up
    if (threshold < 0) and (k == 0):
        return( col )

    # Print recursion status messages
    if verbose:
        if k == 0:
            print(f"Reducing threshold from {threshold:.2f} to {(threshold-dthresh):.2f}.")
            threshold -= dthresh
        elif k == 1:
            print(f"Interpolated {k} tract at a threshold of {threshold:.2f}.")
        else:
            print(f"Interpolated {k} tracts at a threshold of {threshold:.2f}.")

    # Continue recursion
    return( _recurse_interpolation(col, neighbor_weights,
                                   id_to_iloc, threshold, dthresh, verbose)
    )

if __name__ == "__main__":
    from src import plotting
    import matplotlib.pyplot as plt

    # Get NYC census tracts
    nyc_tracts = gpd.read_parquet(config.TRACTS_CLEAN)

    # Test local region by plotting a random local subregion
    local_tracts, borough_tracts = get_local_subregion(nyc_tracts, seed=42)
    fig, ax = plotting.plot_local_subregion(local_tracts, borough_tracts)
    plt.show()