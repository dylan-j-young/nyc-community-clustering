import numpy as np
import pandas as pd
import geopandas as gpd

import os
import logging

from src import extract, config, plotting, utils

def ensure_directories():
    """Ensure that the necessary data and log directories exist."""
    dirs = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.SHAPEFILES_DIR,
        config.INTERIM_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODEL_DIR,
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logging.info(f"Created directory: {d}")

def setup_logging():
    """Set up pipeline log"""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = config.LOG_DIR / "pipeline.log"

    # Set up the configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),    # Writes to file
            logging.StreamHandler()           # Prints to terminal
        ]
    )
    logging.info("--- Pipeline Session Started ---")

def main():
    # --- 0. PROJECT SETUP ---
    setup_logging()
    logging.info("Checking all folders are created...")
    ensure_directories()

    # --- 1. EXTRACTION PHASE ---
    logging.info("Fetching 2020 Demographic Profile...")
    extract.fetch_2020_demographic_profile()

    logging.info("Fetching 2023 5-Year ACS...")
    extract.fetch_2023_acs_5yr_select()

    logging.info("Fetching shapefiles...")
    extract.fetch_shapefiles()

    logging.info("Fetching Neighborhood Tabulation Areas...")
    extract.fetch_nyc_NTAs()

    logging.info("Fetching 2017 Zillow neighborhood boundaries...")
    extract.fetch_zillow_nbds()


    logging.info("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()