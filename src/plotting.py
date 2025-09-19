import numpy as np
import pandas as pd
import geopandas as gpd

import jenkspy

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplcolors
import contextily as ctx

from src import config

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

def plot_pca_weights(pca_components, cols, bar_width=0.1, signed_prob=False):
    """ 
    Given a set of eigenvectors from pca.components_,
    plot the relative weights of each axis in a bar plot, defined as the magnitude squared of the eigenvector components along each axis (possibly signed).

    Parameters
    ----------
    pca_components : np.ndarray, shape (n_eigs, n_cols)
        Array-like of PCA components (eigenvectors of the covariance matrix).

    cols : np.ndarray or list
        Names of the different columns.

    bar_width : float, optional
        Sets the width of each bar (each column is spaced apart by 1). Default is 0.1.

    signed_prob : bool, optional
        Sets whether to plot the bars as strictly positive weights or to also plot the weights with a relative sign. Defualt is False.
    
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

    # Make plot
    fig, ax = plt.subplots(figsize=(8,6))

    # Plot bars
    x = np.arange(n_comps)
    for i, component in enumerate(pca_components):
        offset = (-(n_comps-1)/2 + i) * (bar_width+intrabar_width)
        sign_corr = np.sign(component)/np.abs(component) \
            if signed_prob else 1
        ax.bar(x + offset, sign_corr*component**2, 
               width = bar_width,
               label=f"eig{i}")
    
    ax.set_xticks(x, cols, fontsize=8)
    ax.set_ylim((-1,1) if signed_prob else (0,1))
    ax.grid()
    ax.legend()

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

def plot_choropleth(col, tracts, n_classes = 6, color = mpl.cm.Reds(1.0)):
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

    color : tuple, optional
        The (r,g,b) or (r,g,b,a) values (from 0 to 1) of the brightest color to plot. Default is mpl.colors.Red(1.0).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object with the plotted GeoDataFrame and basemap.
    """
    # Find Jenks natural breaks (cluster 1D data into nc clusters)
    nc = n_classes
    jenks_bdys = np.array( jenkspy.jenks_breaks(col, n_classes=nc) )
    
    # Normalize boundaries to go from 0 to 1 for LinearSegmentedColormap
    vmin, vmax = col.min(), col.max()
    jenks_bdys_norm = (jenks_bdys - vmin) / (vmax - vmin) # length nc+1

    # Generate nc colors from white to color
    base_cmap = mplcolors.LinearSegmentedColormap.from_list("white_to_color_cmap", ["white", color])
    colors = [base_cmap((i+1) / nc) for i in range(nc)] # length nc

    # Get quantized colormap and normalization
    cmap_quant = discrete_nonuniform_colormap("test", jenks_bdys_norm, colors)
    norm = mplcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10,8))

    merged = tracts.copy()
    merged = merged.join(col)

    # Convert to Web Mercator
    merged = merged.to_crs(config.WEB_MERCATOR_EPSG)
    merged.plot(
        column=col.name,
        cmap=cmap_quant,
        norm=norm,
        linewidth=0,
        ax=ax
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

    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap_quant, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

    # force ticks at the actual quantile edges
    cbar.set_ticks(list(jenks_bdys))
    cbar.ax.set_yticklabels([f"{utils.pretty_round(v,3)}" for v in list(jenks_bdys)])
    # cbar.set_label(f"Quantile bins")

    fig.tight_layout()

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