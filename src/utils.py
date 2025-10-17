## PREAMBLE
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from scipy.optimize import curve_fit

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from libpysal.weights import Rook, Queen
from esda import Moran
import networkx as nx

import os
import random
from typing import Optional
import warnings

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
    X : np.ndarray or pd.DataFrame, shape (n_samples, n_features)
        Input compositional data (rows sum to a constant).

    pseudocount : float, optional
        Converts 0 values to this to avoid dividing by 0. Recommended to input a small value that is not much smaller than the smallest possible nonzero value. Default is 1e-5.
    
    Returns
    -------
    X_clr : np.ndarray or pd.DataFrame, shape (n_samples, n_features)
        CLR-transformed data.
    """
    # If DataFrame, convert to ndarray
    is_df = False
    if isinstance(X, pd.DataFrame):
        is_df = True
        df = X.copy()
        X = df.to_numpy()

    # Normalize by sum of first row
    total = X[0].sum()
    X = X / total

    # Clip with pseudocount and 1
    X = np.clip(X, pseudocount, 1)

    # Calculate geometric mean and CLR
    log_gm = np.mean(np.log(X), axis=1, keepdims=True)
    X_clr = np.log(X) - log_gm

    # If DataFrame, convert back to DataFrame
    if is_df:
        df_clr = pd.DataFrame(X_clr,
                              columns=[col + "_clr" for col in df.columns],
                              index=df.index)
        X_clr = df_clr
    
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
        W = Queen.from_dataframe(gdf, use_index=True, silence_warnings=True)
    elif type == "rook":
        W = Rook.from_dataframe(gdf, use_index=True, silence_warnings=True)
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

    # If no tracts were interpolated, reduce threshold (or give up if < 0)
    if k == 0:
        if threshold < 0:
            return( col )
        else:
            threshold -= dthresh

    # Print recursion status messages
    if verbose:
        if k == 0:
            print(f"Reducing threshold from {threshold+dthresh:.2f} to {threshold:.2f}.")
        elif k == 1:
            print(f"Interpolated {k} tract at a threshold of {threshold:.2f}.")
        else:
            print(f"Interpolated {k} tracts at a threshold of {threshold:.2f}.")

    # Continue recursion
    return( _recurse_interpolation(col, neighbor_weights,
                                   id_to_iloc, threshold, dthresh, verbose)
    )

def _feature_r2(df_train, df_test, series_train, series_test):
    """ 
    Internal-facing function that essentially acts as a wrapper for scikit-learn's LinearRegression. Given a df and series with common indices, perform a regression on training values and return R2 for the test values.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training values of the input dataset. All columns used in regression.
    
    df_test : pd.DataFrame
        Test values of input dataset. 

    series_train : pd.Series
        Training values of the input Y series.

    series_test : pd.Series
        Test values of the input Y series.
    
    Returns
    -------
    r2 : float
        The R2 value for the linear regression attempting to predict values in series_test using the data in df_test.
    """
    model = LinearRegression()
    model.fit(df_train, series_train)
    r2 = model.score(df_test, series_test)
    return(r2)

def get_internal_feature_r2s(df, test_size=0.2, n_runs=1, random_state=None):
    """ 
    Given a DataFrame df, attempt to predict each column of values using a linear regression on all the other columns. Returns R2 values for each column in a list.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    test_size : float, optional
        The fraction of rows used in the test dataset. Default is 0.2.
    
    n_runs : int, optional
        The number of times to perform the train/test split and regression. If 1, the returned DataFrame has one column: R2. If greater than 1, the returned DataFrame has R2_mean and R2_std statistics.
    
    random_state : float or None, optional
        Random seed for the train-test split. If None, no random seed is supplied. Default is None.
    
    Returns
    -------
    r2s : pd.DataFrame
        A DataFrame with indices set by the columns of df and values set by the resulting R2s from the linear regression. Includes both mean and std if n_runs > 1.
    """
    # scale features by z score
    cols = df.columns.values
    scaler = StandardScaler()
    df_zscore = df.copy()
    df_zscore[cols] = scaler.fit_transform(df_zscore[cols])

    r2s_all = []
    for i in range(n_runs):
        X_train, X_test = train_test_split(df_zscore, 
                                        test_size=test_size,
                                        random_state=random_state)

        # For each attribute, get an r2 score
        r2s = [
            _feature_r2(
                X_train.drop(columns=[col]), 
                X_test.drop(columns=[col]),
                X_train[col],
                X_test[col]
            ) for col in cols
        ]
        r2s_all.append(r2s)
    r2s_all = np.array(r2s_all).T

    if n_runs == 1:
        return(pd.DataFrame(
            r2s_all.flatten(),
            index=cols, columns=["R2"]
        ))
    else:
        return(pd.DataFrame(
            np.column_stack(( r2s_all.mean(axis=1), r2s_all.std(axis=1) )),
            index=cols, columns=["R2_mean","R2_std"]
        ))

def get_feature_r2s(df_X, df_Y, test_size=0.2, n_runs=1, random_state=None):
    """
    Perform a linear regression on each of the columns in df_Y using the data in df_X (both DataFrames assumed to share common indices). Return a pd.Series object of R2 values for each column in df_Y.

    Parameters
    ----------
    df_X : pd.DataFrame
        Dataset used to predict values in df_Y

    df_Y : pd.DataFrame
        Dataset with potentially multiple columns, sharing indices with df_X.

    test_size : float, optional
        The fraction of rows used in the test dataset. Default is 0.2.
    
    n_runs : int, optional
        The number of times to perform the train/test split and regression. If 1, the returned DataFrame has one column: R2. If greater than 1, the returned DataFrame has R2_mean and R2_std statistics.

    random_state : float or None, optional
        Random seed for the train-test split. If None, no random seed is supplied. Default is None.
    
    Returns
    -------
    r2s : pd.DataFrame
        A DataFrame with indices set by the columns of df_Y and values set by the resulting R2s from the linear regression. Includes both mean and std if n_runs > 1.
    """
    if isinstance(df_Y, pd.Series):
        if df_Y.name is None:
            df_Y.name = "Y"
        df_Y = pd.DataFrame(df_Y)
    
    # Standard scale for regression
    scaler = StandardScaler()
    X_cols = df_X.columns.values
    X_scaled = df_X.copy()
    X_scaled[X_cols] = scaler.fit_transform(X_scaled[X_cols])
    Y_cols = df_Y.columns.values
    Y_scaled = df_Y.copy()
    Y_scaled[Y_cols] = scaler.fit_transform(Y_scaled[Y_cols])

    r2s_all = []
    for i in range(n_runs):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y_scaled, test_size=test_size, random_state=random_state
        )

        r2s = [
            _feature_r2(
                X_train, X_test,
                Y_train[attr], Y_test[attr]
            ) for attr in df_Y
        ]
        r2s_all.append(r2s)
    r2s_all = np.array(r2s_all).T

    if n_runs == 1:
        return(pd.DataFrame(
            r2s_all.flatten(),
            index=df_Y.columns.values, columns=["R2"]
        ))
    else:
        return(pd.DataFrame(
            np.column_stack(( r2s_all.mean(axis=1), r2s_all.std(axis=1) )),
            index=df_Y.columns.values, columns=["R2_mean","R2_std"]
        ))

def remove_small_islands(gdf, n_min=5):
    """ 
    Given a GeoDataFrame and a minimum number of regions n_min to fit in a cluster, identify, isolate, and remove any disconnected islands with fewer than 2*n_min regions. This threshold is the point above which the island could be split into two minimally sized clusters. 

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input dataset with a geometry column for identifying connectivity.
    
    n_min : int, optional
        The minimum size of a desired cluster. Default is 5.
    
    Returns
    -------
    gdf_pruned : gpd.GeoDataFrame
        The input dataset with rows removed that belong to any small islands.
    
    small_islands : list of gpd.GeoDataFrame
        List of GeoDataFrame objects (row slices of the original gdf), with each object corresponding to a connected small island.
    """
    w = Rook.from_dataframe(gdf, use_index=False, silence_warnings=True)
    G = w.to_networkx()
    connected_components = list( nx.connected_components(G) )

    gdf_pruned = gdf.copy()
    small_islands = []
    for component in connected_components:
        if len(component) <= 2*n_min:
            island = gdf.iloc[list(component)].copy()

            small_islands.append(island)
            gdf_pruned = gdf_pruned.drop(island.index)
    
    return(gdf_pruned, small_islands)

def add_small_island_regions(gdf, cluster_attr, small_islands):
    """ 
    Given a list (small_islands) of GeoDataFrames of disconnected regions and a larger GeoDataFrame with the same columns plus an extra column representing region labels, add the small_islands rows back into the larger frame with new region designations.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input dataset with a geometry column for identifying connectivity.
    
    cluster_attr : int, optional
        The name of the categorical column defining each sample's cluster label.

    small_islands : list of gpd.GeoDataFrame
        List of GeoDataFrame objects (row slices of the original gdf), with each object corresponding to a connected small island.
    
    Returns
    -------
    gdf_full : gpd.GeoDataFrame
        The input dataset with islands added back in.   
    """

    n_regions = len(gdf[cluster_attr].unique())

    for i, island in enumerate(small_islands):
        island[cluster_attr] = n_regions + i
    
    gdf_full = gpd.GeoDataFrame(
        pd.concat([gdf] + small_islands),
        crs=gdf.crs
    )

    return(gdf_full)

def get_Morans_I(gdf, attrs, contiguity="rook"):
    """
    Computes Moran's I for each column of gdf specified in attrs, given contiguity set by the geometry column in gdf.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing a geometry column of Polygons or MultiPolygons with valid contiguity (to be calculated).

    attrs : list
        List of columns of gdf to calculate Moran's I on.
    
    contiguity : str, optional
        Specifies Rook or Queen contiguity. Default is Rook.
    
    Returns
    -------
    moran_I : pd.Series
        Series of the Moran's I values, indexed by columns in attrs.
    """
    # Calculate contiguity
    w = Rook.from_dataframe(gdf, use_index=False, silence_warnings=True) \
            if (contiguity == "rook") else \
            Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    
    # Calculate Moran's I
    morans_I = pd.Series(
        [Moran(gdf[col], w).I for col in attrs],
        index=attrs,
        name="morans_I"
    )

    return( morans_I )

def get_nonlinear_feature_r2(df, attr_x, attr_y, model, **kwargs):
    """
    Given two columns in a dataframe and a model, perform a fit using scipy.optimize.curve_fit (with all kwargs passed into this function), and return the R2 value of the fit.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both attrx and attry as columns.

    attr_x : str
        Name of the independent variable column of df.

    attr_y : str
        Name of the dependent variable column of df.

    model : function
        The proposed model, taking the form attr_y = model(attr_x, *params).
    
    Returns
    -------
    r2 : float
        The R2 value of the fit.

    pOpt : np.ndarray, shape (n_params,)
        The optimal parameters of model.
    
    pCov : np.ndarray, shape (n_params, n_params)
        The covariance matrix of the fit.
    """
    # Get X, Y columns as numpy arrays
    X, Y = df[attr_x].to_numpy(), df[attr_y].to_numpy()

    # Perform the curve fit and get predicted Y values
    pOpt, pCov = curve_fit(model, X, Y, **kwargs)
    Y_pred = model(X, *pOpt)

    # Evaluate R2
    SS_total = np.sum( (Y - Y.mean())**2 )
    SS_res = np.sum( (Y - Y_pred)**2 )
    R2 = 1 - SS_res/SS_total

    return( R2, pOpt, pCov )

def get_residual_spatial_structure(gdf, attrs):
    """
    Given a DataFrame df, perform a simple linear regression on each column vs. all the other columns and calculate residuals. Then, calculate Moran's I on those residuals to obtain the degree of spatial structure contained in the residuals. Return all such Moran's I values in a pd.Series object.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input dataset with a geometry column for adjacency.

    attrs : list of str
        List of columns containing feature data.
    
    Returns
    -------
    residual_mIs : pd.Series
        A Series with indices set by attrs and values set by the Moran's I of the residuals on each linear regression.
    """

    residual_mI_list = []
    for attr_y in attrs:
        # Get X, y data arrays
        X = gdf[attrs].drop(columns=[attr_y]).to_numpy()
        y = gdf[attr_y].to_numpy()

        # Perform the regression and calculate residuals
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = pd.Series(
            y - y_pred,
            index=gdf.index,
            name="residuals"
        )

        # Calculate Moran's I
        mI = get_Morans_I(gdf.join(residuals), ["residuals"])
        residual_mI_val = float(mI.loc["residuals"])
        residual_mI_list.append(residual_mI_val)

    residual_mI = pd.Series(
        residual_mI_list, index=attrs, name="residual_mI"
    )
    return( residual_mI )

def aggregate_clusters(gdf, feature_attrs, cluster_attr):
    """ 
    Given a GeoDataFrame with cluster labels, dissolve the individual tracts into a single cluster, and calculate characteristic feature values

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to plot. Must have a geometry column and a column with a name defined by cluster_attr, consisting of cluster labels.

    feature_attrs : list of str
        List of feature variable names to include in the dissolved GeoDataFrame. Aggregated value is calculated using a median.

    cluster_attr : str
        The name of the cluster label to group by.

    Returns
    -------
    clusters : gpd.GeoDataFrame
        GeoDataFrame where each row corresponds to a single cluster. Feature attributes calculated using a median of the representative tracts.
    """

    # Aggregation function for combining geographical and feature data
    aggfunc = {"AREA": "sum", "LAT": "median", "LONG": "median"} | {feature: "median" for feature in feature_attrs}

    # Dissolve clusters
    clusters = gdf.dissolve(by=cluster_attr, aggfunc=aggfunc)

    return( clusters )

def _proportionally_assign_clusters(nc, component_nodes):
    """ 
    Internal method to assign nc total clusters to different connected components in the full contiguity graph.

    Parameters
    ----------
    nc : int
        The total number of clusters.

    component_nodes : list of int
        List of the number of nodes in each component. 
    
    Returns
    -------
    component_ncs : list of int
        List of the number of clusters assigned to each component. Each component must receive a minimum of 1.
    """
    # Get the fraction of total nodes present in each component
    total_nodes = sum(component_nodes)
    component_frac = np.array([nodes/total_nodes for nodes in component_nodes])

    # Find nc_k such that nc_k/nc is as close as possible to component_frac
    rounded = np.round(nc * component_frac)

    # Are any entries equal to 0? If so, set to 1 and coarse assign to component_ncs
    component_ncs = np.array([(1 if nc_k == 0 else nc_k) for nc_k in rounded])
    # Total count may be greater than nc. If so, we have to subtract assigned clusters. Do so iteratively.
    while sum(component_ncs) > nc:
        # How far off is our fraction for each component?
        frac_devs = (component_ncs == 1) * -100 + \
                    (component_ncs != 1) * (component_ncs/nc - component_frac)
        
        # Find largest positive fractional deviation and subtract 1 from it
        i = np.argmax(frac_devs)
        component_ncs[i] -= 1

    # Convert to list of ints
    component_ncs = [int(n) for n in component_ncs.tolist()]

    return( component_ncs )

def perform_multicomponent_cluster(gdf, attrs, nc, alg, name, seed=0):
    """ 
    Given a GeoDataFrame of multiple discontiguous components, perform spatial clustering using the algorithm specified in the model wrapper on each connected component, and then combine into a single series.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input dataset with a geometry column for adjacency.

    attrs : list of str
        List of columns containing feature data.

    nc : int
        The total number of clusters. Must be at least as large as the number of connected components.

    alg : func
        A wrapper from src/algorithms.py that performs a clustering algorithm on a connected component.
    
    name : str
        The desired name for the column of cluster labels.

    seed : int, optional
        A random seed to feed into the clustering algorithm. Default is 0.

    Returns
    -------
    clusters : pd.Series
        Series of cluster labels (from 0 to nc-1), indexed by the index of gdf.
    """
    # Get list of connected components
    w = Rook.from_dataframe(gdf,
            use_index=True, silence_warnings=True
    )
    g = w.to_networkx()
    ccs = list( nx.connected_components(g) )
    component_nodes = [len(cc) for cc in ccs]

    # Designate cluster counts to each component
    component_ncs = _proportionally_assign_clusters(nc, component_nodes)

    # Iterate over connected components
    clusters = pd.Series()
    for i, component in enumerate(ccs):
        # Keep track of nc and label counts
        nc = component_ncs[i]
        prev_ncs = sum(component_ncs[:i])

        # Get connected component gdf
        ids = gdf.index[list(component)]
        connected_gdf = gdf.loc[ids]

        # Cluster using the provided clustering algorithm
        labels = alg(connected_gdf, attrs, nc, seed=seed) + prev_ncs
        if i > 0: 
            clusters = pd.concat([clusters, labels], axis=0)
        else:
            clusters = labels
    
    clusters.name = name
    return( clusters )

if __name__ == "__main__":
    from src import plotting
    import matplotlib.pyplot as plt

    # Get NYC census tracts
    nyc_tracts = gpd.read_parquet(config.TRACTS_CLEAN)

    # Test local region by plotting a random local subregion
    local_tracts, borough_tracts = get_local_subregion(nyc_tracts, seed=42)
    fig, ax = plotting.plot_local_subregion(local_tracts, borough_tracts)
    plt.show()