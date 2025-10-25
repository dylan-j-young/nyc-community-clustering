import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_samples

from libpysal.weights import Rook, Queen
import esda
import networkx as nx
import jenkspy

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplcolors
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import colorsys
import contextily as ctx

import warnings

from src import config, utils, evaluation

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

def _zoom(ax, zoom_out, bbox_aspect=1):
    """
    Basic wrapper for zooming out (or in) on a matplotlib axis.
    
    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object.

    zoom_out : float
        Sets a scale factor for zooming out the map from the census tract (higher = zoomed farther out). Useful if you want to see the broader region of a tract or small collection of tracts.

    bbox_aspect : float, optional
        Sets the aspect ratio for the axis bounding box. Defined as width/height. Default is 1 (square).

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

    # Grow extent to match bbox aspect
    if bbox_aspect * dy > dx:
        dx = dy * bbox_aspect
    else:
        dy = dx / bbox_aspect

    # Zoom out: set new bounds
    xmin_new, xmax_new = x0 - dx*zoom_out, x0 + dx*zoom_out
    ymin_new, ymax_new = y0 - dy*zoom_out, y0 + dy*zoom_out
    ax.set_xlim(xmin_new, xmax_new)
    ax.set_ylim(ymin_new, ymax_new)
    
    return( ax )

def plot_tracts(tracts, 
                zoom_out = "auto",
                zoom_adjust = 0,
                figax = None):
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

    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
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

    # Make plot
    if figax is None:
        figsize = (6, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    # Add tracts and census labels to figure
    ax = _plot_and_annotate(ax, tracts)
    
    # Zoom out: set defaults, or zoom out 
    if zoom_out == "auto":
        if len(tracts) == 1:
            zoom_out = 4
        elif len(tracts) <= 5:
            zoom_out = 2
        else:
            zoom_out = 1
        ax = _zoom(ax, zoom_out)
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
                    source=config.DEFAULT_CTX_PROVIDER,
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
                    source=config.DEFAULT_CTX_PROVIDER,
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
    col : pd.Series
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
    if col.name not in merged.columns:
        merged = merged.join(col)

    # Convert to Web Mercator
    merged = merged.to_crs(config.WEB_MERCATOR_EPSG)
    merged.plot(
        column=col.name,
        cmap=cmap_quant,
        norm=norm,
        alpha=0.75,
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
                    source=config.DEFAULT_CTX_PROVIDER,
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

def _adjust_lightness(color, amount=0.5):
    try:
        c = mplcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mplcolors.to_rgb(c))
    return( colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2]) )

def _cmap_30():
    """
    Internal method to produce a 30-tone categorical colormap, based off the 10 default matplotlib colors (the "tab10" cmap).
    """
    # Get 10 color default map
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Lighten, then darken
    lighter = [_adjust_lightness(color, amount=1.25) \
               for color in default_colors]
    darker = [_adjust_lightness(color, amount=0.8) \
              for color in default_colors]

    colors_30 = default_colors + lighter + darker
    cmap_30 = mplcolors.ListedColormap(colors_30)
    return( cmap_30 )

def _cmap_60():
    """
    Internal method to produce a 60-tone categorical colormap, based off the 10 default matplotlib colors (the "tab10" cmap).
    """
    # Get 10 color default map
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Lighten, then darken
    lighter = [_adjust_lightness(color, amount=1.25) \
               for color in default_colors]
    darker = [_adjust_lightness(color, amount=0.8) \
              for color in default_colors]
    lighter_2 = [_adjust_lightness(color, amount=1.43) \
                 for color in default_colors]
    darker_2 = [_adjust_lightness(color, amount=0.6) \
                for color in default_colors]
    lighter_3 = [_adjust_lightness(color, amount=1.6) \
                 for color in default_colors]

    colors_60 = default_colors + lighter + darker + lighter_2 + darker_2 + lighter_3
    cmap_60 = mplcolors.ListedColormap(colors_60)
    return( cmap_60 )

def plot_categorical(gdf, attr, cmap="default", figax=None):
    """ 
    Given a GeoDataFrame, and a column specified by attr, construct a categorical map.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to plot. Must have a geometry column and a column with a name defined by attr, consisting of categorical values.

    attr : str
        The name of the categorical column to plot.

    cmap : mpl.colors.Colormap, optional
        The (categorical) colormap to use, or the string "default". If "default", uses either "tab10" (the matplotlib default), _cmap_30() (a custom 30-tone variant), or _cmap_60() (a custom 60-tone variant), depending on the number of labels. Default is "default". 
    
    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Determine cmap
    n_labels = gdf[attr].nunique()
    if n_labels <= 10:
        cmap = "tab10"
    elif n_labels <= 30:
        cmap = _cmap_30()
    else:
        cmap = _cmap_60()

    # Make plot
    if figax is None:
        figsize = (9,7)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    # Convert to Web Mercator
    gdf = gdf.to_crs(config.WEB_MERCATOR_EPSG)

    gdf.plot(
        ax=ax, column=attr, categorical=True, edgecolor="w",
        linewidth=0.25, cmap=cmap, alpha=0.75,
        legend=False
    )

    # Improve visualization
    ax.set_title(attr, fontsize=12)
    ax.axis("off")  # Remove axis labels for a clean map
    ax.set_aspect("equal")
    fig.tight_layout()

    # Add basemap
    ctx.add_basemap(ax, 
                    crs=config.WEB_MERCATOR_EPSG,
                    source=config.DEFAULT_CTX_PROVIDER,
                    reset_extent=True,
                    zoom_adjust=0
                    )
    
    return( fig, ax )

def plot_feature(gdf, attr, cmap="inferno", vbounds=(None, None), figax=None):
    """ 
    Given a GeoDataFrame, and a column specified by attr, plot the feature using a continuous (sequential) colormap.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to plot. Must have a geometry column and a column with a name defined by attr, consisting of categorical values.

    attr : str
        The name of the categorical column to plot.

    cmap : mpl.colors.Colormap or str, optional
        The colormap to use, or a string accepted by matplotlib as a colormap. Default is "inferno".

    vbounds : tuple, optional
        Sets the normalization of the colormap in the form (vmin, vmax). If either is None, set automatically. "frac" attributes are automatically scaled to (0, 1) unless otherwise specified. Default is (None, None).
    
    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Make plot
    if figax is None:
        figsize = (9,7)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    # Convert to Web Mercator
    gdf = gdf.to_crs(config.WEB_MERCATOR_EPSG)

    # Normalize colormap for fraction attributes (otherwise do nothing)
    if attr[:4] == "frac":
        vmin, vmax = 0, 1
    else:
        vmin, vmax = vbounds

    gdf.plot(
        ax=ax, column=attr, edgecolor="w",
        linewidth=0.25, cmap=cmap, alpha=0.75,
        vmin=vmin, vmax=vmax,
        legend=True
    )

    # Improve visualization
    ax.set_title(attr)
    ax.axis("off")  # Remove axis labels for a clean map
    ax.set_aspect("equal")
    fig.tight_layout()

    # Add basemap
    ctx.add_basemap(ax, 
                    crs=config.WEB_MERCATOR_EPSG,
                    source=config.DEFAULT_CTX_PROVIDER,
                    reset_extent=True,
                    zoom_adjust=0
                    )
    
    return( fig, ax )

def plot_clusters(gdf, cluster_attr,
                  color_clusters=True, which_clusters="all",
                  colormap="greedy", figax=None):
    """ 
    Given a GeoDataFrame of tracts grouped into clusters, plot the cluster geometries.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to plot. Must have a geometry column and a column with a name defined by cluster_attr, consisting of cluster labels.

    cluster_attr : str
        The name of the cluster label to group by.
    
    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.

    color_clusters : bool, optional
        Flag that sets whether or not to color in the cluster geometries. If True, uses a greedy coloring algorithm from the networkx library. If False, plots only the outlines.

    which_clusters : list or str, optional
        Determines which clusters to plot. If "all", plots all clusters. Otherwise, which_clusters is a list of the cluster label names to plot.

    colormap : str or pd.Series, optional
        Sets the colormap to color each cluster. If "greedy", uses a greedy coloring algorithm on the connectivity graph from the networkx package. Otherwise, expects a pd.Series object with indices matching the gdf and values corresponding to colors.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Aggregate clusters into a gdf
    clusters = utils.aggregate_clusters(gdf, [], cluster_attr)
    clusters = clusters.to_crs(config.WEB_MERCATOR_EPSG)

    # Restrict clusters gdf to those specified in which_clusters
    if which_clusters != "all":
        assert isinstance(which_clusters, list)
        clusters = clusters.loc[which_clusters]

    # Make plot
    if figax is None:
        figsize = (9,7)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    if color_clusters:
        if isinstance(colormap, str) and colormap == "greedy":
            # Greedy color using networkx
            cluster_labels = clusters.index.to_numpy()
            w = Queen.from_dataframe(clusters, 
                                    use_index=False, silence_warnings=True)
            G = w.to_networkx()
            coloring = nx.algorithms.coloring.greedy_color(G)

            # Assign colors to the coloring (using the default matplotlib colors)
            default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colormap = pd.Series({
                cluster_labels[idx]: default_colors[coloring[idx]] \
                for idx in coloring
            })
        else:
            # Expect a colormap to be provided
            assert isinstance(colormap, pd.Series)

        # Plot
        clusters.plot(
            ax=ax, color=colormap, edgecolor="k",
            linewidth=0, alpha=0.5
        )
    else:
        # Plot only outlines
        clusters.plot(
            ax=ax, facecolor="none", edgecolor="k",
            linewidth=1
        )

    # Annotate labels
    for idx, row in clusters.iterrows():
        ax.annotate(text=str(int(idx)), 
                    xy=row.geometry.representative_point().coords[0],
                    horizontalalignment='center', fontsize="small", color='black')

    # Improve visualization
    ax.set_title(cluster_attr, fontsize=12)
    ax.axis("off")  # Remove axis labels for a clean map
    ax.set_aspect("equal")

    # Zoom & bbox regularization
    if len(clusters) <= 2:
        zoom_out = 1
    else:
        zoom_out = 1
    ax = _zoom(ax, zoom_out)

    fig.tight_layout()

    # Add basemap
    ctx.add_basemap(ax, 
                    crs=config.WEB_MERCATOR_EPSG,
                    source=config.DEFAULT_CTX_PROVIDER,
                    reset_extent=True,
                    zoom_adjust=0
                    )
    
    return( fig, ax )

def plot_silhouette(gdf, feature_attrs, cluster_attr,
                    sil_type="path-silhouette",
                    figax=None):
    """ 
    Given a GeoDataFrame of tracts grouped into clusters, calculate the silhouette and plot it. There are a few options for the type of silhouette to plot:
    - "silhouette": the silhouette as defined in sklearn.metrics.silhouette_samples()
    - "path-silhouette": the path silhouette as defined in the esda package.
    - "truncated-psil": the custom-made path silhouette that equalizes the number of samples in each cluster when comparing the average distance.
    - "truncated-psil-bdy": the same as "truncated-psil", but only samples touching a cluster boundary are plotted.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to plot. Must have a geometry column, a set of feature columns, and a column with a name defined by cluster_attr, consisting of cluster labels.

    feature_attrs : list of str
        The list of column names corresponding to the feature space.

    cluster_attr : str
        The name of the cluster label to group by.
    
    sil_type : str, optional
        The type of silhouette to plot. Options are "silhouette", "path-silhouette", "truncated-psil", and "truncated-psil-bdy". Default is "path-silhouette".
    
    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Setup for path silhouette calculation
    X = gdf[feature_attrs].to_numpy()
    labels = gdf[cluster_attr].to_numpy()
    W = Rook.from_dataframe(gdf,
                            use_index=False, silence_warnings=True)

    # Calculate path silhouette
    with warnings.catch_warnings():
        # Warnings from scipy.sparse about efficiency
        warnings.filterwarnings("ignore")
        if sil_type == "silhouette":
            sils_arr = silhouette_samples(
                X, labels
            )
        elif sil_type == "path-silhouette":
            sils_arr = esda.path_silhouette(
                X, labels, W
            )
        else:
            sils_arr = evaluation.truncated_path_silhouette(
                X, labels, W
            )
        
    # Remove non-boundary tracts if sil_type == "truncated-psil-bdy"
    if sil_type == "truncated-psil-bdy":
        # Find boundary tracts
        is_bdy = np.zeros(len(labels)).astype(bool)      
        for i in range(len(labels)):
            lc = labels[i]
            lc_neighbors = set(labels[W.neighbors[i]])
            lc_neighbors.discard(lc)
            is_bdy[i] = (len(lc_neighbors) > 0)
        
        # Replace non boundary tracts wit np.nan
        sils_arr = np.where(~is_bdy, np.nan, sils_arr)

    # Make Series and compute psil average
    sils = pd.Series(sils_arr, 
                     index=gdf.index,
                     name=sil_type)
    sil_mean = np.nanmean(sils_arr)

    # Make plot
    if figax is None:
        figsize = (9,7)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    # Plot sils
    merged = gdf.join(sils)
    merged = merged.to_crs(config.WEB_MERCATOR_EPSG)
    merged.plot(
        column=sil_type,
        cmap="inferno", vmin=-1, vmax=1,
        alpha=0.75,
        linewidth=0,
        ax=ax,
        missing_kwds={
            "color": "#C0C0C0",  # Color for NaN values
            "linewidth": 0,
            "label": "No Data"  # Label for NaN values in the legend
        },
        legend=True,
        legend_kwds={
            "ticks": [-1,-0.5,0,0.5,1]
        }
    )

    # Add cluster outlines
    __, __ = plot_clusters(gdf, cluster_attr, figax=(fig,ax),
                           color_clusters=False)
    
    ax.set_title(f"{sil_type} for {cluster_attr}: mean={round(sil_mean,3)}", fontsize=12)

    return( fig, ax )

def plot_component_scans(metrics, component_names, suptitle):
    """
    Given a metrics dataframe for a dataset with multiple disconnected components, create a grid of plots with shape (n_metrics, n_components). 

    Parameters
    ----------
    metrics : pd.DataFrame
        Dataset of metrics to plot. Columns correspond to the evaluation metrics, and rows have a MultiIndex of the form ("c", "n") where "c" is the component index, and "n" is the number of clusters.

    component_names : list of str
        List of names provided to each component. Provided to give semantic information about which component is which. Example: ["The Bronx", "Brooklyn/Queens", "Manhattan", "Rockaway", "Staten Island"].
    
    suptitle : str
        Title of plot.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    metric_names = metrics.columns.to_numpy()

    components = set(metrics.index.get_level_values(0))
    n_components = len(components)
    n_metrics = metrics.shape[1]

    # Dynamic fig size
    x_size = (4*n_components) if (n_components <= 2) else 10
    y_size = (2*n_metrics) if (n_metrics <= 3) else 7
    fig, ax = plt.subplots(n_metrics, n_components, figsize=(x_size, y_size))

    # Fill in figure
    ymin, ymax = [np.inf] * n_metrics, [-np.inf] * n_metrics
    xlims = [None] * n_components
    for j, c in enumerate(components):
        metrics_j = metrics.loc[c]
        Y = metrics_j.to_numpy()
        X = metrics_j.index.to_numpy()

        # Plot data
        for i, m in enumerate(metric_names):
            ax[i,j].plot(X, Y[:,i], '.-k')
            
            # Update extents
            ymin[i] = min(ymin[i], np.min(Y[:,i]))
            ymax[i] = max(ymax[i], np.max(Y[:,i]))
        
        # Set x plot ranges
        dx = np.max(X) - np.min(X)
        xlims[j] = (0, np.max(X)+dx*0.05)

    # Set y plot ranges
    dy = [ymax[i]-ymin[i] for i in range(n_metrics)]
    ymin = [
        0 if (ymin[i] == 0) else (ymin[i] - dy[i]*0.05) \
        for i in range(n_metrics)
    ]
    ymax = [ymax[i] + dy[i]*0.05 for i in range(n_metrics)]
    ylims = [*zip(ymin, ymax)]        

    # Prettify plots
    for j, c in enumerate(components):
        locatorx = ticker.MaxNLocator(nbins = 5,
                                      steps=[1,2,4,5,10], min_n_ticks=3)
        for i, m in enumerate(metric_names):
            ax[i,j].set_xlim(xlims[j])
            ax[i,j].set_ylim(ylims[i])
            ax[i,j].grid()
            ax[i,j].xaxis.set_major_locator(locatorx)
            ax[i,j].tick_params(axis="both", direction="in")
            
            if i == 0:
                ax[i,j].set_title(component_names[j])

            if i == n_metrics-1:
                ax[i,j].set_xlabel("Number of clusters")
            else:
                ax[i,j].tick_params(axis="x", labelbottom=False)

            if j == 0:
                ax[i,j].set_ylabel(m)
            else:
                ax[i,j].tick_params(axis="y", labelleft=False)

    fig.suptitle(suptitle)
    fig.tight_layout()
    return(fig, ax)

def plot_cluster_scatter(df, feature_attrs, cluster_attr, which_clusters,
                            plotted_attrs="auto", figax=None):
    """
    Given a dataframe and two cluster labels, make a scatter plot comparing the two cluster samples along two feature directions. 
    
    By default the two feature directions chosen are the two with the greatest cluster separation, defined as the distance between the centroids divided by the sum of the standard deviations of each cluster (this is related to the Davies-Bouldin index, with a slightly different metric). However, custom feature attrs can be passed in through the optional parameter `plotted_attrs`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to plot. Must have a set of feature columns (they don't need to be standardized), and a column with a name defined by cluster_attr, consisting of cluster labels.

    feature_attrs : list of str
        The list of column names corresponding to the feature space. They don't need to be standardized.

    cluster_attr : str
        The name of the cluster label to group by.
    
    which_clusters : list of int
        Determines which clusters to plot. Assumed to be two elements long.
    
    plotted_attrs : str or list of str
        Determines which features to plot. If "auto", calculates the two features with the greatest separation index between the clusters. Otherwise, expects a list of two strings corresponding to the feature attributes to plot.
    
    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Get clusters
    clusters = df.groupby(by=cluster_attr)
    c0, c1 = which_clusters
    cluster_c0 = clusters.get_group(c0)
    cluster_c1 = clusters.get_group(c1)

    # Get matrices, centroids, and variances
    X0 = cluster_c0[feature_attrs].to_numpy()
    X1 = cluster_c1[feature_attrs].to_numpy()
    mu0, mu1 = np.mean(X0, axis=0), np.mean(X1, axis=0)
    std0, std1 = np.std(X0, axis=0), np.std(X1, axis=0)

    # Get attrs to plot
    if plotted_attrs == "auto":
        # Find two dimensions with maximum separation scores
        separations = np.abs(mu1 - mu0) / (std0 + std1)
        dim_y, dim_x = np.argsort(separations)[-2:]
        attr_y, attr_x = feature_attrs[dim_y], feature_attrs[dim_x]
    else:
        # Find two dimensions specified by plotted_attrs
        assert isinstance(plotted_attrs, list)
        assert len(plotted_attrs) == 2
        attr_x, attr_y = plotted_attrs
        dim_x = np.argmax(feature_attrs == attr_x)
        dim_y = np.argmax(feature_attrs == attr_y)

    # Make plot
    if figax is None:
        figsize = (6, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax
        
    # Plot data
    ax.scatter(X0[:,dim_x], X0[:,dim_y], s=5, c="C0", label=f"Cluster {c0}",
               zorder=1.7)
    ax.scatter(X1[:,dim_x], X1[:,dim_y], s=5, c="C1", label=f"Cluster {c1}",
               zorder=1.7)
    ax.scatter([mu0[dim_x]], [mu0[dim_y]], s=50, c="C0", edgecolor="k",
               zorder=1.8, label="Centroids")
    ax.scatter([mu1[dim_x]], [mu1[dim_y]], s=50, c="C1", edgecolor="k",
               zorder=1.8)
    
    # Confidence ellipses
    confidence_level = 0.8
    ellipse0_x, ellipse0_y = utils.get_confidence_ellipse(
        X0[:,[dim_x, dim_y]], confidence_level
    )
    ellipse1_x, ellipse1_y = utils.get_confidence_ellipse(
        X1[:,[dim_x, dim_y]], confidence_level
    )
    t = np.arange(0,1,0.001)
    ax.fill(ellipse0_x(t), ellipse0_y(t),
            color='C0', alpha=0.25, zorder=1.6,
            label="80% confid.")
    ax.fill(ellipse1_x(t), ellipse1_y(t),
            color='C1', alpha=0.25, zorder=1.6)

    # Prettify
    ax.set_xlabel(attr_x)
    ax.set_ylabel(attr_y)
    ax.set_title("Feature scatter plot")
    ax.legend(handlelength=1, labelspacing=0.25)
    ax.grid()
    ax.tick_params(axis="both", direction="in")
    fig.tight_layout()

    return(fig, ax)

def plot_cluster_bar(df, feature_attrs, cluster_attr, which_clusters,
                     figax=None):
    """
    Given a dataframe and two cluster labels, make a bar plot comparing the two cluster samples along all (z-scored) feature directions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to plot. Must have a set of feature columns (they don't need to be pre-standardized), and a column with a name defined by cluster_attr, consisting of cluster labels.

    feature_attrs : list of str
        The list of column names corresponding to the feature space. They don't need to be standardized.

    cluster_attr : str
        The name of the cluster label to group by.
    
    which_clusters : list of int
        Determines which clusters to plot. Assumed to be two elements long.
    
    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    
    # Standardize features for z-score plotting
    df_std = df.copy()
    df_std[feature_attrs] = StandardScaler().fit_transform(
        df_std[feature_attrs].to_numpy()
    )

    # Get centroids
    centroids = utils.aggregate_clusters(df_std, feature_attrs=feature_attrs, cluster_attr=cluster_attr)
    c0, c1 = which_clusters
    centroid0 = centroids[feature_attrs].loc[c0].to_numpy()
    centroid1 = centroids[feature_attrs].loc[c1].to_numpy()
    x = np.arange(len(centroid0))

    # Get standard deviations
    clusters = df_std.groupby(by=cluster_attr)
    cluster_c0 = clusters.get_group(c0)
    cluster_c1 = clusters.get_group(c1)
    std0 = cluster_c0[feature_attrs].std().to_numpy()
    std1 = cluster_c1[feature_attrs].std().to_numpy()

    # Make plot
    if figax is None:
        figsize = (8,2)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    # Make bars and label axes
    bar_width = 0.2
    ax.bar(x - bar_width/2, centroid0,
        width=bar_width, label=f"Cluster {c0}", zorder=2)
    ax.errorbar(x - bar_width/2, centroid0, yerr=std0,
                capsize=2, color="black", elinewidth=1, 
                fmt=".", markersize=2, zorder=3)
    ax.bar(x + bar_width/2, centroid1,
        width=bar_width, label=f"Cluster {c1}", zorder=2)
    ax.errorbar(x + bar_width/2, centroid1, yerr=std1,
                capsize=2, color="black", elinewidth=1, 
                fmt=".", markersize=2, zorder=3,
                label="Std. dev.")
    ax.set_xticks(x, feature_attrs, fontsize=8, rotation=15, ha="right", va="top")
    ax.set_ylabel("z-score")
    ax.set_title("Centroid feature bar plot")

    # Prettify and return
    locatory = ticker.MaxNLocator(nbins = 6, steps=[1, 5], min_n_ticks=4)
    ax.yaxis.set_major_locator(locatory)
    ax.grid()
    ax.legend(handlelength=1, labelspacing=0.25)
    fig.tight_layout()
    return(fig, ax)

def plot_ntas(gdf, annotate=True, annotation_col="ntaabbrev", figax=None):
    """ 
    Given a GeoDataFrame of NTAs, plot outlines.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to plot. Must have a geometry column and a column with a name defined by cluster_attr, consisting of cluster labels.

    annotate : bool, optional
        Sets whether or not to annotate the NTAs with names. Default is False.

    annotation_col : str, optional
        The name of the column to annotate with (only if annotate is True). Default is "ntaabbrev".
    
    figax : tuple of (fig, ax) or None, optional
        Optionally passes in an existing tuple of matplotlib fig and ax objects for the plot. If figax is None, the function generates a new fig and ax. Default is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Convert CRS
    gdf = gdf.to_crs(config.WEB_MERCATOR_EPSG)

    # Make plot
    if figax is None:
        figsize = (9,7)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    # Plot only outlines
    gdf.plot(
        ax=ax, facecolor="none", edgecolor="gray",
        linewidth=0.5, linestyle="dashed"
    )

    # Optional: annotate NTAs
    if annotate:
        for idx, row in gdf.iterrows():
            ax.annotate(text=row[annotation_col], 
                        xy=row.geometry.representative_point().coords[0],
                        horizontalalignment='center', fontsize="xx-small", color='gray', zorder=1.5, clip_on=True)

    # Improve visualization
    ax.axis("off")  # Remove axis labels for a clean map
    ax.set_aspect("equal")

    fig.tight_layout()

    # Add basemap
    ctx.add_basemap(ax, 
                    crs=config.WEB_MERCATOR_EPSG,
                    source=config.DEFAULT_CTX_PROVIDER,
                    reset_extent=True,
                    zoom_adjust=0
                    )
    
    return( fig, ax )

def plot_cluster_comparison_summary(gdf, feature_attrs, cluster_attr,
                                    which_clusters, plotted_attrs="auto"):
    """
    Given a spatially clustered dataset, compare the geography and features of two clusters and plot a summary consisting of three subplots. The first takes up the first row and shows a bar plot of each of the (z-scored) feature directions. The second, on the left of the second row, is a plot of the clusters. The third, on the right of the second row, is a scatter plot of the clusters in a 2D projection of feature space corresponding to the featured in plotted_attrs.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Dataset to plot. Must have a valid geometry column, a set of feature columns (non-standardized), a column with a name defined by cluster_attr, consisting of cluster labels.

    feature_attrs : list of str
        The list of column names corresponding to the feature space. They don't need to be standardized.

    cluster_attr : str
        The name of the cluster label to group by.
    
    which_clusters : list of int
        Determines which clusters to plot. Assumed to be two elements long.
    
    plotted_attrs : str or list of str
        Determines which features to plot. If "auto", calculates the two features with the greatest separation index between the clusters. Otherwise, expects a list of two strings corresponding to the feature attributes to plot.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Set up figure
    fig = plt.figure(figsize=(8, 6.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4,4], height_ratios=[2,4.5])
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    # Bar plot
    __, __ = plot_cluster_bar(gdf, feature_attrs, cluster_attr,
                              which_clusters, figax=(fig, ax1))

    # Geography plot
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colormap = pd.Series(default_colors[:2], index=which_clusters)
    __, __ = plot_clusters(gdf, cluster_attr,
                           which_clusters=which_clusters, colormap=colormap,
                           figax=(fig, ax2))
    ax2.set_title("Cluster geographies")

    # Scatter plot
    __, __ = plot_cluster_scatter(gdf, feature_attrs, cluster_attr,
                                  which_clusters, plotted_attrs=plotted_attrs,
                                  figax=(fig, ax3))
    
    fig.suptitle(f"Cluster comparison for: {cluster_attr}")    
    fig.tight_layout()
    ax = np.array([[ax1,ax1],[ax2,ax3]])
    return(fig, ax)

# Colormaps for access outside of this script
cmap_30 = _cmap_30()
cmap_60 = _cmap_60()

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