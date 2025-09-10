import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

from src import config

def plot_local_subregion(local_tracts, borough_tracts):
    """ 
    Quickly visualizes a choice of local subregion for testing purposes.
    For plotting, GeoDataFrames are locally converted to Web Mercator and then plotted on top of a contextily basemap.

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
    fig, ax = plt.subplots(figsize=(10, 8))

    # Add tracts to figure
    borough_tracts.plot(ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=1
                )
    local_tracts.plot(ax=ax,
                facecolor="red",
                edgecolor="none",
                alpha=0.25
                )

    # Annotate with census tract numbers
    for idx, row in borough_tracts.iterrows():
        # Label is Census Tract number
        label = row["TRACT"]

        # Label location is a representative point within the tract
        point = row.geometry.representative_point()

        ax.annotate(text=label, xy=(point.x, point.y),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=6)


    # Improve visualization
    ax.set_title("Subregion census tracts", fontsize=14)
    ax.axis("off")  # Remove axis labels for a clean map

    # No default zoom
    ax.set_aspect("equal")

    ctx.add_basemap(ax, 
                    crs=config.WEB_MERCATOR_EPSG,
                    source=ctx.providers.CartoDB.Positron,
                    reset_extent=True,
                    zoom_adjust=1)
    
    return( fig, ax )

if __name__ == "__main__":
    from src import utils

    # Get NYC census tracts
    nyc_tracts = gpd.read_parquet(config.TRACTS_CLEAN)

    # Test local region by plotting a random local subregion
    local_tracts, borough_tracts = utils.get_local_subregion(
        nyc_tracts, seed=42
        )
    fig, ax = plot_local_subregion(local_tracts, borough_tracts)
    plt.show()