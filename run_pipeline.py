import numpy as np
import pandas as pd
import geopandas as gpd

import os
import logging
import argparse

from src import config
from src.etl import extract, load

def ensure_directories():
    """Ensure that the necessary data and log directories exist."""
    logging.info("Checking all folders are created...")
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

# Different stages to run pipeline from
STAGES = ['extract', 'load', 'transform', 'features']

def main(start_from = "extract"):
    # Setup
    setup_logging()
    ensure_directories()

    run_from = STAGES.index(start_from)
    
    if run_from <= STAGES.index('extract'):
        # Get data and dump into raw files
        extract.run()
    if run_from <= STAGES.index('load'):
        # Minimal cleaning and type checks, then load to cleaned tables
        load.run() 
    if run_from <= STAGES.index('transform'):
        pass
        # # Row filtering, partial column filtering, geography transformation
        # transform.run()
    if run_from <= STAGES.index('features'):
        pass
        # # Generate final dataset for clustering
        # transform.to_gold()

    logging.info("--- Pipeline Completed ---")

if __name__ == "__main__":
    # Set up flag to start pipeline at different stages
    # python run_pipeline.py --from-stage extract
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-stage', 
                        choices=STAGES, default='extract')
    args = parser.parse_args()

    main(start_from=args.from_stage)