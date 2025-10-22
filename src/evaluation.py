import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.sparse as sp
from scipy.sparse import csgraph as cg
from scipy.sparse import SparseEfficiencyWarning
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score

from libpysal.weights import Rook
from libpysal.weights.util import WSP
from esda import path_silhouette
import networkx as nx

import warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from src import config, utils

def _calculate_truncated_sample_silhouettes(dist_matrix, labels):
    """ 
    Internal method for computing a custom-made, modified form of the path silhouette score. This function is a simplified and modified implementation of `sklearn.silhouette_samples()`.
    
    The original path silhouette is defined in the library `esda` under `esda.path_silhouette()`, and it measures a "path-weighted" distance between each sample (summing the distances between neighboring nodes on a shortest-path connection on the graph connecting the two samples in question) before calculating the silhouette in the normal way:
    
    .. math::
        s(i) = \\frac{ \\min_{k \\neq c}(\\overline{d}_{k}(i)) - \\overline{d}_{c}(i) }{ \\mathscr{N} },

    where sample i is in cluster c, :math:`\\overline{d}_{k}(i)` is the average (here, path-weighted) distance between sample i and samples in cluster k, and :math:`\\mathscr{N}` is a normalization factor that makes the silhouette range from -1 (bad clustering) to 1 (ideal clustering).

    This version uses the same path-weighted distance, but when comparing :math:`\\overline{d}_{k}(i)` to :math:`\\overline{d}_{c}(i)` for each cluster k, it only averages over the smallest m(c,k) distances, where m(c,k) is the number of samples in the smaller of the two clusters. This modification attempts to reduce the penalty that large clusters tend to experience in the path silhouette score when bordering much smaller clusters.

    Parameters
    ----------
    dist_matrix : {array-like, sparse matrix} of shape (n_samples, n_samples)
        An array of pairwise distances between samples. If a sparse matrix is provided, CSR format should be favoured avoiding an additional copy.

    labels : array-like of shape (n_samples,)
        Label values for each sample.

    Returns
    -------
    scores : array-like of shape (n_samples,)
        Truncated path silhouette scores for each sample.
    """
    # Get label names and counts
    label_names, label_freqs = np.unique(labels, return_counts=True)

    # Ensure an equal number of samples considered in intra and intercluster distances
    max_samples = np.minimum(label_freqs[:,None]-1, label_freqs)

    # Iterate over each sample i (in cluster c with label lc)
    scores = np.zeros(len(labels))
    for i, lc in enumerate(labels):
        c = np.argmax(label_names == lc)

        # If cluster has one sample, set score to 0
        if label_freqs[c] == 1:
            scores[i] = 0
            continue

        # All distances between samples i and other samples
        dists = dist_matrix[i]

        # avg_dist_c[n-1] is the average distance of the shortest n intracluster distances
        intra_dists = np.sort(dists[labels==lc])[1:] # Remove (i,i) distance
        avg_dist_c = intra_dists.cumsum() / np.arange(1, len(intra_dists)+1)

        # Iterate over other cluster labels
        min_avgdist = np.inf
        for k, lk in enumerate(label_names):
            if lk == lc:
                continue
            n_max = max_samples[c, k]
            
            # Intercluster distances
            inter_dists = np.sort(dists[labels==lk])
            avg_dist_k = np.mean(inter_dists[:n_max])
            diff_kc = avg_dist_k - avg_dist_c[n_max-1]
            
            # Get smallest average intercluster distance
            if diff_kc < min_avgdist:
                min_avgdist = diff_kc
                min_score = diff_kc / max(avg_dist_k, avg_dist_c[n_max-1])
        
        scores[i] = min_score
    
    return(scores)

def truncated_path_silhouette(data, labels, W,
                             D=None, metric=euclidean_distances):
    """ 
    Computes a custom-made, modified form of the path silhouette score. This function is a simplified and modified implementation of `esda.path_silhouette()`.

    The original path silhouette measures a "path-weighted" distance between each sample (summing the distances between neighboring nodes on a shortest-path connection on the graph connecting the two samples in question) before calculating the silhouette in the normal way:
    
    .. math::
        s(i) = \\frac{ \\min_{k \\neq c}(\\overline{d}_{k}(i)) - \\overline{d}_{c}(i) }{ \\mathscr{N} },

    where sample i is in cluster c, :math:`\\overline{d}_{k}(i)` is the average (here, path-weighted) distance between sample i and samples in cluster k, and :math:`\\mathscr{N}` is a normalization factor that makes the silhouette range from -1 (bad clustering) to 1 (ideal clustering).

    This version uses the same path-weighted distance, but when comparing :math:`\\overline{d}_{k}(i)` to :math:`\\overline{d}_{c}(i)` for each cluster k, it only averages over the smallest m(c,k) distances, where m(c,k) is the number of samples in the smaller of the two clusters. This modification attempts to reduce the penalty that large clusters tend to experience in the path silhouette score when bordering much smaller clusters.

    Parameters
    ----------
    data : np.ndarray (N,P)
        Matrix of data with N observations and P covariates.

    labels : np.ndarray (N,)
        Flat vector of the L labels assigned over N observations.

    W : libpysal.weights.W | libpysal.graph.Graph
        Spatial weights object reflecting the spatial connectivity in the problem under analysis.

    D : np.ndarray (N,N) | None, optional
        A precomputed distance matrix to apply over W. If passed, takes precedence over data, and data is ignored. Default is None.

    metric : callable, optional
        Function mapping the (N,P) data into an (N,N) dissimilarity matrix, like that found in sklearn.metrics.pairwise or scipy.spatial.distance. Default is sklearn.metrics.pairwise.euclidean_distances

    Returns
    -------
    scores : array-like of shape (n_samples,)
        Truncated path silhouette scores for each sample.    
    """
    # -- From esda.path_silhouette: get path-weighted distances (all_pairs) --
    if D is None:
        D = metric(data)
    # polymorphic for sparse & dense input
    assert (D < 0).sum() == 0, (
        "Distance metric has negative values, which is not supported."
    )
    off_diag_zeros = (D + np.eye(D.shape[0])) == 0
    D[off_diag_zeros] = -1
    Wm = sp.csr_matrix(W.sparse)
    DW = sp.csr_matrix(Wm.multiply(D))
    DW.eliminate_zeros()
    DW[DW < 0] = 0
    assert (DW < 0).sum() == 0
    all_pairs = cg.shortest_path(DW, directed=False)
    labels = np.asarray(labels)

    # -- From esda.path_silhouette: recurse on connected subcomponents
    if W.n_components > 1:
        psils_ = np.empty(W.n, dtype=float)
        for component in np.unique(W.component_labels):
            this_component_mask = np.nonzero(W.component_labels == component)[0]
            subgraph = W.sparse[
                this_component_mask.reshape(-1, 1),  # these rows
                this_component_mask.reshape(1, -1),
            ]  # these columns
            subgraph_W = WSP(subgraph).to_W()
            assert subgraph_W.n_components == 1
            # DW operation is idempotent
            subgraph_D = DW[
                this_component_mask.reshape(-1, 1),  # these rows
                this_component_mask.reshape(1, -1),
            ]  # these columns
            subgraph_labels = labels[this_component_mask]
            n_subgraph_labels = len(np.unique(subgraph_labels))
            if not (2 < n_subgraph_labels < (subgraph_W.n - 1)):
                psils = subgraph_solutions = [0] * subgraph_W.n
            else:
                subgraph_solutions = truncated_path_silhouette(
                    data=None,
                    labels=subgraph_labels,
                    W=subgraph_W,
                    D=subgraph_D,
                    metric=metric
                )
                psils = subgraph_solutions
            psils_[this_component_mask] = psils
        psils = psils_
    else:
        # Fully connected graph: compute a modified form of the silhouette that truncates the number of samples used per cluster
        psils = _calculate_truncated_sample_silhouettes(all_pairs, labels)
    
    scores = psils
    return( scores )

# A contiguity-constrained Davies-Bouldin score
def contiguous_dbs(gdf, feature_attrs, label_attr):
    """
    Computes a custom-made, modified form of the Davies-Bouldin score that only compares contiguous clusters. This function is a simplified and modified implementation of `sklearn.metrics.davies_bouldin_score()`.

    The original Davies-Bouldin score is an evaluation metric not specific to spatial clustering. It is computed by first evaluating the within-cluster variation S_c for cluster c, defined here as the average Euclidean distance between each sample i in the cluster and the cluster centroid. Then, the similarity between each pair of clusters c and k, given by R_ck = (S_c + S_k)/M_ck, where M_ck is the Euclidean distance between centroids of clusters c and k. Each cluster c is assigned its "worst-case" similarity R_c = max(R_ck) over all other clusters k, and the total Davies-Bouldin score is then given by the average of R_c over all clusters c in the set. A smaller score indicates more well-separated clusters.

    This version modifies the original definition by setting R_c as the maximum of R_ck over all *neighboring* clusters k, rather than all other clusters. The modification attempts to provide a spatially-aware measure of cluster similarity by only considering cluster separation at the local level (and ignoring highly similar but spatially distant pairs of clusters).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The dataset to study. Must have a valid geometry column for contiguity processing, as well as a column of cluster labels.

    feature_attrs : list of str
        A list of all the column names used to calculate the distance metric.
    
    label_attr : str
        The name of the column of cluster labels.

    Returns
    -------
    overall_score : float
        The contiguous Davies-Boudin score, averaged over each cluster.
    """

    X_df = gdf.loc[:,feature_attrs]
    X = X_df.values

    # -- Generate adjacency matrix for labels --
    clusters = gdf.dissolve(by=label_attr)
    nc = len(clusters)
    W = Rook.from_dataframe(clusters, 
                            use_index=True, silence_warnings=True)
    cluster_names = clusters.index.values.tolist()

    # -- Calculate centroids and intra-dists (from sklearn implementation) --
    intra_dists = np.zeros(nc)
    centroids = np.zeros((nc, len(X[0])), dtype=float)
    for k in range(nc):
        cluster_name = cluster_names[k]
        cluster_k = X_df.loc[gdf[label_attr] == cluster_name].values
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))

    # -- Calculate all pairwise scores (modified from sklearn implementation) --
    centroid_distances = pairwise_distances(centroids)
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return(0.0)
    
    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    all_pair_scores = combined_intra_dists / centroid_distances

    # -- Restrict to contiguous pairs before maximizing score --
    scores = np.zeros(nc)
    for k in range(nc):
        neighbors = W.neighbors[cluster_names[k]]
        idx_neighbors = [cluster_names.index(nbr) for nbr in neighbors]
    
        if len(idx_neighbors) == 0:
            scores[k] = 0
        else:
            scores[k] = np.max(all_pair_scores[k, idx_neighbors])

    overall_score = float(np.mean(scores))
    return( overall_score )

def polsby_popper(gdf, geometry="geometry"):
    """
    Given a GeoDataFrame with a valid geometry column of polygons, calculate the Polsby-Popper score of each sample. This score quantifies geometric compactness on a scale from 0 to 1, with 1 being maximally compact (a circle), and lower scores indicating less compactness.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The dataset. Must contain a valid geometry column.

    geometry : str, optional
        The name of the geometry column. Default is "geometry".

    Returns
    -------
    overall_score : float
        Average Polsby-Popper score over all clusters.
    """
    # Convert to Web Mercator. Won't be absolutely accurate, but we only really care about relative score. And NYC is not so geographically extended that the distortion should change significantly across the city
    gdf = gdf.to_crs(config.WEB_MERCATOR_EPSG)
    geom = gdf[geometry]

    # Calculate Polsby-Popper score for each row in gdf
    p = geom.length
    a = geom.area    
    score = (4*np.pi*a)/(p*p)

    # Average to get the overall score
    overall_score = float( np.mean(score) )
    return( overall_score )

def heterogeneity(df, feature_attrs, label_attr):
    H = 0
    clustered = df.groupby(label_attr)
    for label, cluster in clustered:
        # Get within-cluster SSD (heterogeneity)
        X = cluster[feature_attrs].to_numpy()
        xbar = np.mean(X, axis=0, keepdims=True)
        ssd = np.sum((X - xbar)**2)

        # Add to overall heterogeneity
        H += ssd
    return(H)

def tabulate_evaluation_metrics(gdf, feature_attrs, label_attrs,
                                metric_names="all", verbose=False):
    """ 
    Given a dataset with (potentially multiple) columns with cluster labels, evaluate these clusterings using various metrics defined by metric_names. Possible metrics include:

    - "polsby-popper"
    - "davies-bouldin"
    - "contiguous-dbs"
    - "silhouette"
    - "path-silhouette"
    - "truncated-psil"
    - "truncated-psil-bdy"
    - "heterogeneity"

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The dataset. Must contain a valid geometry column for contiguity generation, as well as columns containing cluster labels.

    feature_attrs : list of str
        A list of all the column names used to calculate the distance metric.
    
    label_attrs : list of str
        List of the names of columns containing cluster labels to evaluate.

    metric_names : str or list of str, optional
        Specifies what metrics to evaluate. If "all", evaluates all listed metrics. Otherwise, this must be a list of valid metric names as defined above. Default is "all"

    verbose : bool, optional
        Specifies whether or not to print update flags as different cluster columns are evaluated. Default is False.

    Returns
    -------
    eval_metrics : pd.DataFrame
        DataFrame with columns given by the evaluation metrics used and indices given by the cluster column names.
    """
    if metric_names == "all":
        metric_names = ["polsby-popper", "davies-bouldin", "contiguous-dbs", "silhouette", "path-silhouette", "truncated-psil", "truncated-psil-bdy", "heterogeneity"]

    # Iterate over cluster label columns
    all_scores = []
    for label_attr in label_attrs:
        if verbose:
            print(f"Evaluating {label_attr}...")
        scores = []

        # Process data, labels, clusters, and contiguity objects
        X = gdf[feature_attrs].to_numpy()
        labels = gdf[label_attr].to_numpy()
        clusters = utils.aggregate_clusters(gdf, feature_attrs, label_attr)
        W = Rook.from_dataframe(gdf,
                                use_index=False, silence_warnings=True)
        n_labels = len(set(labels))

        # -- Polsby-Popper --
        if "polsby-popper" in metric_names:
            scores.append( polsby_popper(clusters) )

        # -- Davies-Bouldin --
        if "davies-bouldin" in metric_names:
            if n_labels > 1:
                scores.append( davies_bouldin_score(X, labels) )
            else:
                scores.append( 0 )

        # -- Contiguous Davies-Boudin score --
        if "contiguous-dbs" in metric_names:
            scores.append( contiguous_dbs(gdf, feature_attrs, label_attr) )
        
        # -- Average path silhouette --
        if "silhouette" in metric_names:
            if n_labels > 1:
                sil = silhouette_score(
                    X, labels,
                    metric="euclidean"
                )
            else:
                sil = 0
            scores.append( sil )
        
        # -- Average path silhouette --
        if "path-silhouette" in metric_names:
            if n_labels > 1: 
                psil_arr = path_silhouette(
                    X, labels, W, 
                    metric=euclidean_distances,
                    closest=False
                )
                scores.append( float(np.mean(psil_arr)) )
            else:
                scores.append( 0 )

        # -- Truncated path silhouette --
        if ("truncated-psil" in metric_names) \
            or ("truncated-psil-bdy" in metric_names):
            if n_labels > 1:
                mpsil_arr = truncated_path_silhouette(X, labels, W,
                                                    D=None, metric=euclidean_distances)
            else:
                mpsil_arr = np.zeros(len(labels))
        
            # Average over all tracts
            if "truncated-psil" in metric_names:
                scores.append( float(np.mean(mpsil_arr)) )

            # Only average over tracts on a cluster boundary
            if "truncated-psil-bdy" in metric_names:
                weights = np.ones(len(labels)).astype(bool)
                for i in range(len(labels)):
                    lc = labels[i]
                    lc_neighbors = set(labels[W.neighbors[i]])
                    lc_neighbors.discard(lc)
                    weights[i] = (len(lc_neighbors) > 0)
                
                if n_labels > 1:
                    scores.append( np.average(mpsil_arr, weights=weights) )
                else:
                    scores.append( 0 )

        # -- SSD heterogeneity --
        if "heterogeneity" in metric_names:
            scores.append( heterogeneity(gdf, feature_attrs, label_attr) )

        all_scores.append(scores)

    # Construct DataFrame out of scores
    eval_metrics = pd.DataFrame(
        all_scores, index=label_attrs, columns=metric_names
    )

    return( eval_metrics )

def scan_component_n(gdf, feature_attrs, alg, nc0, scan_size_max=20, seed=0):
    """
    Given a GeoDataFrame with features to cluster and a passed-in spatial clustering algorithm, scan the number of clusters for each connected component of samples, and compute evaluation metrics.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input dataset with a geometry column for adjacency and feature columns given by feature_attrs.

    feature_attrs : list of str
        List of columns containing feature data.

    alg : func
        A wrapper from src/algorithms.py that performs a clustering algorithm on a connected component.

    nc0 : int
        Initial guess for the total number of clusters across all components. Algorithm will scan approximately from 0 to 2*nc0, proportionally distributed across components.

    scan_size_max : int, optional
        The maximum number of cluster numbers to try for each connected component. The function samples linearly from 0 to twice the proportionally assigned cluster number from nc0.

    seed : int, optional
        A random seed to feed into the clustering algorithm. Default is 0.

    Returns
    -------
    all_eval_metrics : pd.DataFrame
        DataFrame summarizing the evaluation metrics for each clustering run. Columns correspond to the evaluation metrics, and rows have a MultiIndex of the form ("c", "n") where "c" is the component index, and "n" is the number of clusters.
    """
    # Get list of connected components
    ccs = utils.get_connected_components(gdf)
    component_nodes = [len(cc) for cc in ccs]

    # Get nc scans for each connected component
    component_nc_scans = []
    component_nc0s = utils._proportionally_assign_clusters(
        nc0, component_nodes
    )
    for component_nc0 in component_nc0s:
        if component_nc0 == 1:
            scan = [1,2,3]
        else:
            min_scan_step = int(np.ceil(2*component_nc0/scan_size_max))
            scan = [
                *range(min_scan_step, 2*component_nc0+min_scan_step, min_scan_step)
            ]
        component_nc_scans.append(scan)

    # Iterate over connected components
    all_eval_metrics = pd.DataFrame()
    for i, component in enumerate(ccs):
        # Get component-specific data
        component_nc_scan = component_nc_scans[i]
        ids = gdf.index[list(component)]
        connected_gdf = gdf.loc[ids]

        # Scan over desired nc values
        clusters = pd.Series()
        for nc in component_nc_scan:
            # Perform clustering
            cluster_attr = f"c{i}_n{nc}"
            component_labels = alg(connected_gdf, feature_attrs, nc, seed=seed)
            component_labels.name = cluster_attr
            
            # Combine results into a dataframe
            if len(clusters) == 0: 
                clusters = component_labels
            else:
                clusters = pd.concat([clusters, component_labels], axis=1)
            
        # Run evaluation metrics on df
        cluster_attrs = clusters.columns
        merged = connected_gdf.join(clusters)

        # Get eval metrics and add columns for eventual MultiIndex
        eval_metrics = tabulate_evaluation_metrics(
            merged, feature_attrs, cluster_attrs,
            metric_names="all"
        )
        eval_metrics["c"] = i
        eval_metrics["n"] = component_nc_scan

        if i == 0:
            all_eval_metrics = eval_metrics
        else:
            all_eval_metrics = pd.concat([all_eval_metrics, eval_metrics], axis=0)

    # Set up MultiIndex and return
    all_eval_metrics = all_eval_metrics.set_index(["c","n"])
    return( all_eval_metrics )