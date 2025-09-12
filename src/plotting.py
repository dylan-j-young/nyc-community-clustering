import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    from src import utils

    # Get NYC census tracts
    nyc_tracts = gpd.read_parquet(config.TRACTS_CLEAN)

    # Test plot_local_subregion()
    local_tracts, borough_tracts = utils.get_local_subregion(
        nyc_tracts, seed=42
        )
    fig, ax = plot_local_subregion(local_tracts, borough_tracts)
    plt.show()

    # # Test plot_tracts()
    # tract_id = "36061024100"
    # tract = nyc_tracts.loc[tract_id]
    # fig, ax = plot_tracts(tract, zoom_adjust=1)
    # plt.show()