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
    model = KMeans(n_clusters=n, 
               init=init, n_init=n_init, max_iter=max_iter, tol=tol,
               verbose=verbose,
               random_state=random_state, copy_x=copy_x, algorithm=algorithm)
    
    # Get data and fit
    X = gdf[attrs].to_numpy()
    model.fit(X)

    # Predict cluster labels
    labels = pd.Series(
        model.predict(X),
        index=gdf.index,
        name="labels"
    )

    return( labels )

def sc_kmeans(gdf, attrs, n, seed=0):
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
    # Data and contiguity matrix
    X = gdf[attrs].to_numpy()
    w = Rook.from_dataframe(gdf, use_index=False, silence_warnings=True)

    # Default options unless otherwise specified
    drop_islands = True
    
    # Instantiate and solve model
    model = RegionKMeansHeuristic(X, n, w,
                    drop_islands=drop_islands, seed=seed)
    model.solve()

    # Extract labels
    labels = pd.Series(model.labels_, index=gdf.index, name="labels")
    return( labels )

def sc_agg(gdf, attrs, n, seed=None):
    """ 
    Wrapper for sklearn's AgglomerativeClustering algorithm. Given a GeoDataFrame of fully connected tracts, perform agglomerative clustering with a spatial connectivity constraint, using a Ward linkage.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to cluster. Must contain a valid geometry column in order to calculate contiguity.

    attrs : list of str
        List of the column names of gdf to be used in clustering.

    n : int
        Number of clusters to form. Must be at least 1.

    seed : int or None, optional
        Not used by this model.

    Returns
    -------
    labels : pd.Series
        Series of the resulting cluster labels, indexed by the index in gdf. Name of the series is "labels".
    """
    # Data and contiguity matrix
    X = gdf[attrs].to_numpy()
    w = Rook.from_dataframe(gdf, use_index=False, silence_warnings=True)

    # Agglomerative clustering options. Default unless otherwise specified
    connectivity = w.sparse # Enforce spatial connectivity constraint
    linkage = "ward"
    metric = "euclidean" # Must be Euclidean for Ward linkage
    memory = None
    distance_threshold = None
    compute_distances = False
    
    # Set up model and fit
    model = AgglomerativeClustering(n_clusters = n,
        linkage=linkage, connectivity=connectivity, metric=metric,
        memory=memory,
        distance_threshold=distance_threshold, compute_distances=compute_distances
    )
    model.fit(X)

    labels = pd.Series(model.labels_, index=gdf.index, name="labels")
    return( labels )