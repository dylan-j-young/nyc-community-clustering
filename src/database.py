import geopandas as gpd
import sqlite3
import logging

from . import config

def get_connection():
    """Returns a connection to the SQLite database."""
    return( sqlite3.connect(config.DATABASE_DIR) )

def save_to_db(df, table_name, if_exists="replace", spatial=False):
    """
    Saves a dataframe to the raw SQLite database.
    """
    try:
        if spatial:
            if not isinstance(df, gpd.GeoDataFrame):
                raise TypeError(f"spatial=True requires a GeoDataFrame, got {type(df)}")

            # Enforce CRS
            if df.crs is None:
                df = df.set_crs(config.WGS84_EPSG)
            else:
                df = df.to_crs(config.WGS84_EPSG)

            df.to_file(config.DATABASE_DIR,
                    driver="SQLite", spatialite=True,
                    layer=table_name)
        else:    
            with get_connection() as conn:
                # if_exists="replace" ensures we don't add duplicates
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        logging.info(f"Successfully wrote {len(df)} rows to table: {table_name}")
    except Exception as e:
        logging.error(f"Failed to write to table {table_name}: {e}")
        raise

        
def query_db(sql_query):
    df = gpd.read_file(config.DATABASE_DIR, sql=sql_query)

    if isinstance(df, gpd.GeoDataFrame):
        # Reattach CRS
        df = df.set_crs(config.WGS84_EPSG)

    # enforce lowercase column name convention
    df.columns = df.columns.str.lower()

    return( df )