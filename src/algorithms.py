## PREAMBLE
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.cluster import KMeans, AgglomerativeClustering
from libpysal.weights import Rook
from spopt.region import RegionKMeansHeuristic

def kmeans(gdf, attrs, n, seed=None):
    """ 
    Wrapper for sklearn's kmeans algorithm. Given a GeoDataFrame of fully connected tracts, perform a spatially constrained k-means algorithm and return a series of labels.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to cluster.

    attrs : list of str
        List of the column names of gdf to be used in clustering.

    n : int
        Number of clusters to form. Must be at least 1.

    seed : int or None, optional
        Random seed to ensure reproducible results. If None, no seed is input into the algorithm. Default is NOne.

    Returns
    -------
    labels : pd.Series
        Series of the resulting cluster labels, indexed by the index in gdf. Name of the series is "labels".
    """

    # Options for k-means. Default unless otherwise specified
    init = "k-means++"
    n_init = "auto"
    max_iter = 300
    tol = 0.0001
    verbose = 0
    random_state = seed
    copy_x = True
    algorithm = "lloyd"

    # Instantiate model
    kmeans = KMeans(n_clusters=n, 
               init=init, n_init=n_init, max_iter=max_iter, tol=tol,
               verbose=verbose,
               random_state=random_state, copy_x=copy_x, algorithm=algorithm)
    
    # Get data and fit
    X = gdf[attrs].to_numpy()
    kmeans.fit(X)

    # Predict cluster labels
    labels = pd.Series(
        kmeans.predict(X),
        index=gdf.index,
        name="labels"
    )

    return( labels )

def connected_subset_sckmeans(gdf, attrs, n, seed=0):
    """ 
    Wrapper for spopt's RegionKMeansHeuristic algorithm. Given a GeoDataFrame of fully connected tracts, perform a spatially constrained k-means algorithm and return a series of labels.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to cluster. Must contain a valid geometry column in order to calculate contiguity.

    attrs : list of str
        List of the column names of gdf to be used in clustering.

    n : int
        Number of clusters to form. Must be at least 1.

    seed : int, optional
        Random seed to ensure reproducible results. If None, no seed is input into the algorithm. Default is 0.

    Returns
    -------
    labels : pd.Series
        Series of the resulting cluster labels, indexed by the index in gdf. Name of the series is "labels".
    """
    X = gdf[attrs].to_numpy()
    w = Rook.from_dataframe(gdf, use_index=False, silence_warnings=True)

    # Default options unless otherwise specified
    drop_islands = True
    
    sc_kmeans = RegionKMeansHeuristic(X, n, w,
                    drop_islands=drop_islands, seed=seed)
    sc_kmeans.solve()

    labels = pd.Series(sc_kmeans.labels_, index=gdf.index, name="labels")
    return( labels )