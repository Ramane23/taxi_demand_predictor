import sys
import os

# Adjust the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import zipfile 
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
#Plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    load_batch_of_features_from_store,
     load_predictions_from_store
)

from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout="wide")

# title
# current_date = datetime.strptime('2023-01-05 12:00:00', '%Y-%m-%d %H:%M:%S')
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Taxi demand prediction 🚕')
st.header(f'{current_date} UTC')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 7

def load_shape_data_file() -> gpd.geodataframe.GeoDataFrame:
    """
    Fetches remote file with shape data, that we later use to plot the
    different pickup_location_ids on the map of NYC.

    Raises:
        Exception: when we cannot connect to the external server where
        the file is.

    Returns:
        GeoDataFrame: columns -> (OBJECTID	Shape_Leng	Shape_Area	zone	LocationID	borough	geometry)
    """
    # download zip file
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f'{URL} is not available')

    # unzip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # load and return shape file
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')