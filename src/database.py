import geopandas as gpd
import sqlite3
import logging

from . import config

def save_to_db(df, table_name, if_exists="replace", spatial=False):
    """
    Saves a dataframe to the raw SQLite database.
    """
    if spatial == False:
        conn = sqlite3.connect(config.DATABASE_DIR)
        try:
            # if_exists="replace" ensures we don't add duplicates
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            logging.info(f"Successfully wrote {len(df)} rows to table: {table_name}")
        except Exception as e:
            logging.error(f"Failed to write to table {table_name}: {e}")
            raise
        finally:
            conn.close()
    else:
        # Enforce CRS
        if df.crs is None:
            df = df.set_crs(config.WGS84_EPSG)
        else:
            df = df.to_crs(config.WGS84_EPSG)

        # Write to file
        try:
            df.to_file(config.DATABASE_DIR,
                    driver="SQLite",
                    spatialite=True,
                    layer=table_name)
            logging.info(f"Successfully wrote {len(df)} rows to table: {table_name}")
        except Exception as e:
            logging.error(f"Failed to write to table {table_name}: {e}")
            raise
        
def query_db(sql_query):
    df = gpd.read_file(config.DATABASE_DIR, sql=sql_query)

    if isinstance(df, gpd.GeoDataFrame):
        # Reattach CRS
        df = df.set_crs(config.WGS84_EPSG)

    return( df )