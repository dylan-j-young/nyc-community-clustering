import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import jenkspy

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplcolors
import contextily as ctx

from src import config, utils

def _plot_and_annotate(ax, tracts,
                       linewidth: float = 1,
                       fontsize: int = 6):
    """
    Basic wrapper for census tract plotting. Plot the polygons and annotate them.
    """
    # Add tracts to figure
    tracts.plot(ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=linewidth
               )

    # Annotate with census tract numbers
    for idx, row in tracts.iterrows():
        # Label is Census Tract number
        label = row["TRACT"]

        # Label location is a representative point within the tract
        point = row.geometry.representative_point()

        ax.annotate(text=label, xy=(point.x, point.y),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=fontsize)
    
    return( ax )

def _zoom(ax, zoom_out):
    """
    Basic wrapper for zooming out (or in) on a matplotlib axis.
    
    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object.

    zoom_out : float
        Sets a scale factor for zooming out the map from the census tract (higher = zoomed farther out). Useful if you want to see the broader region of a tract or small collection of tracts.

    Returns
    -------
    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object.
    """    
    # Zoom out: get current bounds and center
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0, y0 = (xmin+xmax)/2, (ymin+ymax)/2
    dx, dy = (xmax-xmin)/2, (ymax-ymin)/2

    # Zoom out: set new bounds
    xmin_new, xmax_new = x0 - dx*zoom_out, x0 + dx*zoom_out
    ymin_new, ymax_new = y0 - dy*zoom_out, y0 + dy*zoom_out
    ax.set_xlim(xmin_new, xmax_new)
    ax.set_ylim(ymin_new, ymax_new)
    
    return( ax )

def plot_tracts(tracts, 
                zoom_out: float | str = "auto",
                zoom_adjust: int = 0):
    """
    Quickly visualizes a subset of census tracts (or a single tract). For plotting, GeoDataFrames are locally converted to Web Mercator and then plotted on top of a contextily basemap.

    Parameters
    ----------
    tracts : gpd.GeoDataFrame or pd.Series
        The selected tract(s) to plot. If a single tract, the CRS is assumed to be in WGS84 to start (and then converted to Web Mercator for plotting)

    zoom_out : float or "auto", optional
        Sets a scale factor for zooming out the map from the census tract (higher = zoomed farther out). Useful if you want to see the broader region of a tract or small collection of tracts. Default is "auto", which sets to 1 for multiple tracts and 5 for a single tract.

    zoom_adjust : int, optional
        Sets an adjustment to the level of detail for the contextily basemap. A higher number increases the resolution of the map. Recommended not to go outside -1, 0, or 1 due to excessive load times. Default is 0.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """

    # Convert single-tract pd.Series to a single-row GeoDataFrame
    if isinstance(tracts, pd.Series):
        tracts = gpd.GeoDataFrame(tracts.to_frame().T,
                                  crs=config.WGS84_EPSG)
    
    # Convert to Web Mercator (for compatibility with basemaps)
    tracts = tracts.to_crs(config.WEB_MERCATOR_EPSG)

    # Create fig, ax
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Add tracts and census labels to figure
    ax = _plot_and_annotate(ax, tracts)
    
    # Zoom out: set defaults, or zoom out 
    if zoom_out == "auto":
        zoom_out = 1 if (len(tracts) > 1) else 5
    else:
        assert not isinstance(zoom_out, str)
        ax = _zoom(ax, zoom_out)
    
    # Improve visualization
    ax.set_title("Census tracts", fontsize=12)
    ax.axis("off")  # Remove axis labels for a clean map
    ax.set_aspect("equal")
    fig.tight_layout()

    # Add basemap
    ctx.add_basemap(ax, 
                    crs=config.WEB_MERCATOR_EPSG,
                    source=ctx.providers.CartoDB.Positron,
                    reset_extent=True,
                    zoom_adjust=zoom_adjust
                   )

    return( fig, ax )

def plot_local_subregion(local_tracts, borough_tracts):
    """ 
    Quickly visualizes a choice of local subregion for testing purposes. For plotting, GeoDataFrames are locally converted to Web Mercator and then plotted on top of a contextily basemap.

    Parameters
    ----------
    local_tracts : GeoDataFrame
        The local subregion of census tracts.

    borough_tracts : GeoDataFrame
        A wider subregion of tracts, for context.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Convert to Web Mercator (for compatibility with basemaps)
    local_tracts = local_tracts.to_crs(config.WEB_MERCATOR_EPSG)
    borough_tracts = borough_tracts.to_crs(config.WEB_MERCATOR_EPSG)

    # Create fig, ax
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Shade in local tracts
    local_tracts.plot(ax=ax,
                facecolor="red",
                edgecolor="none",
                alpha=0.25
                )
    
    # Zoom out to set xlim, ylim based off local_tracts
    ax = _zoom(ax, 1.25)
    
    # Add borough tracts and census labels to figure
    ax = _plot_and_annotate(ax, borough_tracts,
                            linewidth=0.5,
                            fontsize=4)
    
    # Improve visualization
    ax.set_title("Local census tracts", fontsize=12)
    ax.axis("off")  # Remove axis labels for a clean map
    ax.set_aspect("equal")
    fig.tight_layout()

    # Add basemap
    ctx.add_basemap(ax, 
                    crs=config.WEB_MERCATOR_EPSG,
                    source=ctx.providers.CartoDB.Positron,
                    reset_extent=True,
                    zoom_adjust=1
                   )
    
    return( fig, ax )

def plot_pca_summary(df, idx=None, signed_weights=False):
    """ 
    Perform PCA on the input df and plot a summary of the results. This consists of two plots. The first graphs the unexplained fractional variance against the number of principal components, and the second graphs the normalized component weights along each feature direction (weights sum to 1).

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe to perform PCA on.
    
    idx : np.ndarray or list or slice or None, optional
        Subset of component indices to plot. If None, plots all components. Default is None.

    signed_weights : bool, optional
        Sets whether to plot the bars as strictly positive weights or to also plot the weights with a relative sign. Defualt is False.
    
    Returns
    -------
    pca : sklearn.decomposition.PCA
        The PCA object

    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes array with shape (1,2).
    """
    # Perform PCA and get feature names
    pca = PCA().fit(df.to_numpy())
    cols = df.columns.values

    # Create fig, ax objects
    fig, ax = plt.subplots(1, 2, figsize=(15,5),
                           gridspec_kw={"width_ratios": [1,3]})

    # ax[0]: Quick variance plot
    n_eigs = 1 + np.arange(len(cols))
    ax[0].plot(n_eigs, 1 - pca.explained_variance_ratio_.cumsum(), '.-')
    ax[0].set_xticks(n_eigs)
    ax[0].set_ylim((-0.01, None))
    ax[0].grid()
    ax[0].set_xlabel("Number of components")
    ax[0].set_ylabel("Unexplained fractional variance")
    ax[0].set_title("Variance vs. # Components")

    # ax[1]: Component weights
    n_bars = len(cols) if (idx is None) else len(cols[idx])
    bar_width = min(0.1, 0.67/n_bars)
    __, __ = plot_pca_weights(
        pca.components_, cols,
        idx=idx, signed_weights=signed_weights, bar_width=bar_width,
        figax=(fig, ax[1])
    )

    fig.suptitle("PCA Summary")
    fig.tight_layout()
    return( pca, fig, ax )

def plot_pca_weights(pca_components, cols, idx=None, 
                     bar_width=0.1, signed_weights=False,
                     figax=None, **kwargs):
    """ 
    Given a set of eigenvectors from pca.components_,
    plot the relative weights of each axis in a bar plot, defined as the magnitude squared of the eigenvector components along each axis (possibly signed).

    Parameters
    ----------
    pca_components : np.ndarray, shape (n_eigs, n_cols)
        Array-like of PCA components (eigenvectors of the covariance matrix).

    cols : np.ndarray or list
        Names of the different columns.
    
    idx : np.ndarray or list or None, optional
        Subset of component indices to plot. If None, plots all components. Default is None.

    bar_width : float, optional
        Sets the width of each bar (each column is spaced apart by 1). Default is 0.1.

    signed_weights : bool, optional
        Sets whether to plot the bars as strictly positive weights or to also plot the weights with a relative sign. Defualt is False.

    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Params
    intrabar_width = 0
    n_comps = len(cols)

    # Define indices
    if idx is None:
        idx = [i for i in range(len(cols))]
    n_bars = pca_components[idx].shape[0]

    # Make plot
    if figax is None:
        figsize = (8,6) if not ("figsize" in kwargs) else kwargs["figsize"]
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    # Plot bars
    x = np.arange(n_comps)
    for i, component in enumerate(pca_components[idx]):
        k = idx[i]
        offset = (-(n_bars-1)/2 + i) * (bar_width+intrabar_width)
        sign_corr = np.sign(component) if signed_weights else 1
        ax.bar(x + offset, sign_corr*component**2, 
               width = bar_width,
               label=f"eig{k}")
    
    ax.set_xticks(x, cols, fontsize=8, rotation=45, ha="right")
    ax.set_ylim((-0.5,1) if signed_weights else (0,1))
    ax.set_ylabel("Weights (signed)" if signed_weights else "Weights")
    ax.grid()
    ax.legend()
    ax.set_title("PCA Component Weights")

    return(fig, ax)

def discrete_nonuniform_colormap(name, boundaries, colors, N=256):
    """ 
    Returns a mpl.colors.LinearSegmentedColormap with N segments, separated into k = len(colors) regions bounded by the k+1 values in boundaries (assumed from 0 to 1).

    The LinearSegmentedColormap is generated using a dict with four keys: "red", "green", "blue", and "alpha". The values are a list of 3-tuples of the form (x_i, y0_i, y1_i), where x monotonically increases from 0 to 1 (the interpolation endpoints), y0_i is the right color value endpoint of the previous segment, and y1_i is the left color value endpoint of the next segment. To generate the discrete colormap with nonuniform boundaries, we define breakpoints set by boundaries and set y0_i = y1_{i-1} to get the colormap to interpolate between two equal endpoints within each segment.

    Parameters
    ----------
    name : str
        The desired name of this colormap.
    
    boundaries : list or np.array, shape k+1
        The boundaries of the k color regions. The first element must be 0, and the last must be 1. Values in boundaries should monotonically increase.

    colors : list or np.array
        The k colors. Each color should be a 3 or 4 element tuple representing (r,g,b) or (r,g,b,a) values, normalized from 0 to 1.
    
    N : int, optional
        The number of segments for the LinearSegmentedColormap. Defualt is 256.

    Returns
    -------
    cmap : mpl.colors.LinearSegmentedColormap
        The desired colormap. 
    """
    # Initialize vars
    k = len(colors)
    rlist, glist, blist, alist = [], [], [], []

    # Pull out r, g, b, a lists from the list of color tuples
    r, g, b, a = mplcolors.to_rgba_array(colors).T

    # Enumerate over each boundary point
    for i, bdy in enumerate(boundaries):
        # Define interpolation indices to form flat segments. First j0 and last j1 do nothing.
        if i == 0:
            j0, j1 = 0, 0
        elif i == k:
            j0, j1 = k-1, k-1
        else:
            j0, j1 = i-1, i
        
        # Add interpolation endpoint to each list
        rlist.append((bdy, r[j0], r[j1]))
        glist.append((bdy, g[j0], g[j1]))
        blist.append((bdy, b[j0], b[j1]))
        alist.append((bdy, a[j0], a[j1]))
    
    # Construct the dict and then make the colormap
    cdict = {"red": rlist,
             "green": glist,
             "blue": blist,
             "alpha": alist}
    cmap = mplcolors.LinearSegmentedColormap(name, cdict, N=N)

    return( cmap )

def plot_choropleth(col, tracts, n_classes = 6, cmap = mpl.cm.Reds,
                    figax = None):
    """ 
    Given a column of data and a set of tracts, construct a choropleth map with breaks defined using the Jenks natural breaks algorithm (1D K-means clustering). Colors are evenly spaced from white to the value specified by `color`.

    Parameters
    ----------
    col : pd.Series or np.ndarray
        Array-like of the data to plot. Assumed to be in the proper order to correlate with the tracts GeoDataFrame.

    tracts : gpd.GeoDataFrame
        The various tracts to plot.

    n_classes : int, optional
        The number of color classes to plot in the choropleth. Defualt is 6.

    cmap : mpl.colors.Colormap, optional
        The (continuous) colormap to pull colors from. Default is mpl.colors.Red.
    
    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Find Jenks natural breaks (cluster 1D data into nc clusters)
    nc = n_classes
    jenks_bdys = np.array( 
        jenkspy.jenks_breaks(col[col.notna()], n_classes=nc)
    )
    
    # Normalize boundaries to go from 0 to 1 for LinearSegmentedColormap
    vmin, vmax = col.min(), col.max()
    jenks_bdys_norm = (jenks_bdys - vmin) / (vmax - vmin) # length nc+1

    # Generate nc colors from base colormap
    colors = [cmap((i+1) / nc) for i in range(nc)] # length nc

    # Get quantized colormap and normalization
    cmap_quant = discrete_nonuniform_colormap("test", jenks_bdys_norm, colors)
    norm = mplcolors.Normalize(vmin=vmin, vmax=vmax)

    # Make plot
    if figax is None:
        figsize = (8,6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    merged = tracts.copy()
    merged = merged.join(col)

    # Convert to Web Mercator
    merged = merged.to_crs(config.WEB_MERCATOR_EPSG)
    merged.plot(
        column=col.name,
        cmap=cmap_quant,
        norm=norm,
        linewidth=0,
        ax=ax,
        missing_kwds={
            "color": "#505050",  # Color for NaN values
            "hatch": "////",  # Texture to the polygons for clarity
            "linewidth": 0,
            "label": "No Data"  # Label for NaN values in the legend
        },
        legend=True,
        legend_kwds={
            "ticks": list(jenks_bdys)
        }
    )

    # Improve visualization
    ax.set_title(col.name, fontsize=12)
    ax.axis("off")  # Remove axis labels for a clean map
    ax.set_aspect("equal")

    # Add basemap
    ctx.add_basemap(ax, 
                    crs=config.WEB_MERCATOR_EPSG,
                    source=ctx.providers.CartoDB.Positron,
                    reset_extent=True
                    )

    fig.tight_layout()

    return( fig, ax )

def plot_feature_comparison(df, attr1, attr2, figax=None):
    """
    Given two columns from a DataFrame, plot a scatter plot comparing the two in order to look for correlations, clusters, or other interesting structure.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.

    attr1 : str
        The name of the DataFrame column to be plotted on the X axis.

    attr2 : str
        The name of the DataFrame column to be plotted on the Y axis.

    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object containing the scatter plot.
    """
    # Make plot
    if figax is None:
        figsize = (4.5, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    ax.plot(df[attr1], df[attr2], '.', markersize=2)
    ax.grid()
    ax.set_xlabel(attr1)
    ax.set_ylabel(attr2)
    ax.set_title(f"Feature comparison scatter plot")

    fig.tight_layout()

    return(fig, ax)

def plot_nonlinear_fit_summary(gdf, attr_x, attr_y, model, **kwargs):
    """
    Given two columns in a dataframe and a model, perform a fit using scipy.optimize.curve_fit (with all kwargs passed into this function) as well as a spatial autocorrelation analysis on the residuals with Moran's I. Plot the results in two panels: in the left panel, a scatter plot with the fitted curve and R2. In the right panel, a choropleth map of the residuals with Moran's I.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing both attrx and attry as columns.

    attr_x : str
        Name of the independent variable column of df.

    attr_y : str
        Name of the dependent variable column of df.

    model : function
        The proposed model, taking the form attr_y = model(attr_x, *params).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plots.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes array with shape (1,2).
    """
    # Perform R2 analysis
    r2, pOpt, pCov = utils.get_nonlinear_feature_r2(
        gdf, attr_x, attr_y, model, **kwargs
    )

    # Calculate residuals
    X, Y = gdf[attr_x].to_numpy(), gdf[attr_y].to_numpy()
    Y_pred = model(X, *pOpt)
    residuals = pd.Series(Y - Y_pred, index=gdf.index, name="residuals")

    # Perform Moran's I on residuals
    gdf_augmented = gdf.copy()
    gdf_augmented["residuals"] = residuals
    mI = utils.get_Morans_I(gdf_augmented, ["residuals", attr_x, attr_y])
    mI_res, mI_x, mI_y = [
        float(mI.loc[attr]) for attr in ["residuals", attr_x, attr_y]
    ]

    # Create fig, ax objects
    fig, ax = plt.subplots(1, 2, figsize=(10,5),
                           gridspec_kw={"width_ratios": [2,3]})

    # ax[0]: Curve fit plot
    ax[0].scatter(X, Y, s=5, marker=".", alpha=0.25)
    ax[0].plot(np.sort(X), model(np.sort(X), *pOpt), '-k',
               linewidth=1, label=fr"$R^2$ = {round(r2, 3)}")
    ax[0].grid()
    ax[0].set_xlabel(attr_x)
    ax[0].set_ylabel(attr_y)
    ax[0].set_title("Nonlinear curve fit")
    ax[0].legend()

    # ax[1]: Moran's I spatial autocorrelation
    __, __ = plot_choropleth(residuals, gdf, figax=(fig, ax[1]))
    ax[1].set_title(fr"$I_\mathrm{{{"Moran"}}}$(residuals) = {round(mI_res,2)} (vs. X: {round(mI_x,2)}, Y: {round(mI_y,2)})", pad=-15)

    fig.suptitle(f"Nonlinear fit summary: Y ({attr_y}) vs. X ({attr_x})")
    fig.tight_layout()
    return( fig, ax )

def plot_residual_structure(gdf, attrs, attr_y):
    """
    Given a GeoDataFrame with feature attributes given by attrs, perform a linear regression to predict attr_y vs. all other attributes using sklearn.linear_model.LinearRegression as well as a spatial autocorrelation analysis on the residuals with Moran's I. Plot the residuals and their Moran's I using a choropleth map.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing both attrx and attry as columns.

    attrs : list of str
        List of columns used in the model. Includes attr_y.

    attr_y : str
        Name of the dependent variable column of df.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object.
    """
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
    mI = utils.get_Morans_I(gdf.join(residuals), ["residuals", attr_y])
    mI_res, mI_y = [
        float(mI.loc[attr]) for attr in ["residuals", attr_y]
    ]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    __, __ = plot_choropleth(residuals, gdf, figax=(fig, ax))
    ax.set_title(fr"$I_\mathrm{{{"Moran"}}}$(residuals) = {round(mI_res,2)} (vs. Y: {round(mI_y,2)})", pad=-15)

    return( fig, ax )

if __name__ == "__main__":
    from src import utils

    # Get NYC census tracts
    nyc_tracts = gpd.read_parquet(config.TRACTS_CLEAN)

    # # Test plot_local_subregion()
    # local_tracts, borough_tracts = utils.get_local_subregion(
    #     nyc_tracts, seed=42
    #     )
    # fig, ax = plot_local_subregion(local_tracts, borough_tracts)
    # plt.show()

    # # Test plot_tracts()
    # tract_id = "36061024100"
    # tract = nyc_tracts.loc[tract_id]
    # fig, ax = plot_tracts(tract, zoom_adjust=1)
    # plt.show()

    # Test plot_choropleth()
    decennial2020_dp = pd.read_parquet(config.DECENNIAL2020_DP_CLEAN)
    col = pd.Series(decennial2020_dp["pop_hispanic"]/decennial2020_dp["pop"],
                    index=decennial2020_dp.index, name="pct_hispanic")
    fig, ax = plot_choropleth(col, nyc_tracts)
    plt.show()