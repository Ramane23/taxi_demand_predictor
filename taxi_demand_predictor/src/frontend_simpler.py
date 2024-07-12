import zipfile  # Importing the zipfile module for working with zip archives
from datetime import datetime, timedelta  # Importing datetime and timedelta for date and time manipulations
import requests  # Importing requests module for making HTTP requests
import numpy as np  # Importing numpy for numerical operations
import pandas as pd  # Importing pandas for data manipulation and analysis
import streamlit as st  # Importing streamlit for building interactive web applications
import geopandas as gpd  # Importing geopandas for working with geospatial data
import pydeck as pdk  # Importing pydeck for creating deck.gl visualizations
import sys
import os

# Adjust the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import (  # Importing custom functions from the src.inference module
    load_predictions_from_store,
    load_batch_of_features_from_store
)
from src.paths import DATA_DIR  # Importing DATA_DIR constant from the src.paths module
from src.plot import plot_one_sample  # Importing the plot_one_sample function from the src.plot module

st.set_page_config(layout="wide")  # Setting the layout of the Streamlit app to wide

# Setting the current date to the current UTC time, floored to the nearest hour
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Taxi demand prediction 🚕')  # Setting the title of the Streamlit app
st.header(f'{current_date} UTC')  # Setting the header to display the current date and time in UTC

# Setting up a progress bar in the sidebar
progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 6  # Number of steps in the progress bar

def load_shape_data_file() -> gpd.geodataframe.GeoDataFrame:
    """
    Fetches remote file with shape data, that we later use to plot the
    different pickup_location_ids on the map of NYC.

    Raises:
        Exception: when we cannot connect to the external server where
        the file is.

    Returns:
        GeoDataFrame: columns -> (OBJECTID Shape_Leng Shape_Area zone LocationID borough geometry)
    """
    # URL of the zip file containing shape data
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    # Sending a GET request to download the zip file
    response = requests.get(URL)
    # Path to save the downloaded zip file
    path = DATA_DIR / f'taxi_zones.zip'
    if response.status_code == 200:
        # Writing the content of the response to the zip file
        open(path, "wb").write(response.content)
    else:
        # Raising an exception if the file is not available
        raise Exception(f'{URL} is not available')

    # Unzipping the file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # Reading the shape file and converting the coordinate reference system to EPSG:4326
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')

@st.cache_data
def _load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """Wrapped version of src.inference.load_batch_of_features_from_store, so we can add Streamlit caching

    Args:
        current_date (datetime): The current date and time

    Returns:
        pd.DataFrame: n_features + 2 columns:
            - `rides_previous_N_hour`
            - `rides_previous_{N-1}_hour`
            - ...
            - `rides_previous_1_hour`
            - `pickup_hour`
            - `pickup_location_id`
    """
    return load_batch_of_features_from_store(current_date)  # Calling the load_batch_of_features_from_store function

@st.cache_data
def _load_predictions_from_store(
    from_pickup_hour: datetime,
    to_pickup_hour: datetime
) -> pd.DataFrame:
    """Wrapped version of src.inference.load_predictions_from_store, so we can add Streamlit caching

    Args:
        from_pickup_hour (datetime): Minimum datetime (rounded hour) for which we want to get predictions
        to_pickup_hour (datetime): Maximum datetime (rounded hour) for which we want to get predictions

    Returns:
        pd.DataFrame: 2 columns: pickup_location_id, predicted_demand
    """
    return load_predictions_from_store(from_pickup_hour, to_pickup_hour)  # Calling the load_predictions_from_store function

with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = load_shape_data_file()  # Loading shape data file
    st.sidebar.write('✅ Shape file was downloaded ')  # Displaying a message in the sidebar
    progress_bar.progress(1 / N_STEPS)  # Updating the progress bar

with st.spinner(text="Fetching model predictions from the store"):
    predictions_df = _load_predictions_from_store(
        from_pickup_hour=current_date - timedelta(hours=1),  # Setting the start time to one hour before the current time
        to_pickup_hour=current_date  # Setting the end time to the current time
    )
    st.sidebar.write('✅ Model predictions arrived')  # Displaying a message in the sidebar
    progress_bar.progress(2 / N_STEPS)  # Updating the progress bar

# Checking if predictions for the current hour have already been computed and are available
next_hour_predictions_ready = \
    False if predictions_df[predictions_df.pickup_hour == current_date].empty else True
prev_hour_predictions_ready = \
    False if predictions_df[predictions_df.pickup_hour == (current_date - timedelta(hours=1))].empty else True

if next_hour_predictions_ready:
    # Predictions for the current hour are available
    predictions_df = predictions_df[predictions_df.pickup_hour == current_date]

elif prev_hour_predictions_ready:
    # Predictions for the current hour are not available, so we use previous hour predictions
    predictions_df = predictions_df[predictions_df.pickup_hour == (current_date - timedelta(hours=1))]
    current_date = current_date - timedelta(hours=1)  # Adjusting the current date to the previous hour
    st.subheader('⚠️ The most recent data is not yet available. Using last hour predictions')

else:
    # Raising an exception if features are not available for the last 2 hours
    raise Exception('Features are not available for the last 2 hours. Is your feature pipeline up and running? 🤔')

with st.spinner(text="Preparing data to plot"):

    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        """Convert value in the range minval...maxval to a color in the range startcolor to stopcolor. The colors passed and the one returned are composed of a sequence of N component values.

        Credits to https://stackoverflow.com/a/10907855
        """
        f = float(val - minval) / (maxval - minval)
        return tuple(f * (b - a) + a for (a, b) in zip(startcolor, stopcolor))

    # Merging the geographical data with the prediction data
    df = pd.merge(geo_df, predictions_df,
                  right_on='pickup_location_id',
                  left_on='LocationID',
                  how='inner')

    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    progress_bar.progress(3 / N_STEPS)  # Updating the progress bar

with st.spinner(text="Generating NYC Map"):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,  # Latitude for the initial view
        longitude=-73.9712,  # Longitude for the initial view
        zoom=11,  # Zoom level
        max_zoom=16,  # Maximum zoom level
        pitch=45,  # Pitch angle
        bearing=0  # Bearing angle
    )

    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,  # Opacity of the layer
        stroked=False,  # Whether the layer has a stroke
        filled=True,  # Whether the layer is filled
        extruded=False,  # Whether the layer is extruded
        wireframe=True,  # Whether the layer has a wireframe
        get_elevation=10,  # Elevation value
        get_fill_color="fill_color",  # Fill color
        get_line_color=[255, 255, 255],  # Line color
        auto_highlight=True,  # Whether to highlight the layer on hover
        pickable=True,  # Whether the layer is pickable
    )

    tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_demand}"}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)  # Displaying the deck.gl visualization
    progress_bar.progress(4 / N_STEPS)  # Updating the progress bar

with st.spinner(text="Fetching batch of features used in the last run"):
    features_df = _load_batch_of_features_from_store(current_date)  # Loading the batch of features
    st.sidebar.write('✅ Inference features fetched from the store')  # Displaying a message in the sidebar
    progress_bar.progress(5 / N_STEPS)  # Updating the progress bar

with st.spinner(text="Plotting time-series data"):

    predictions_df = df

    row_indices = np.argsort(predictions_df['predicted_demand'].values)[::-1]
    n_to_plot = 10

    # Plot each time-series with the prediction
    for row_id in row_indices[:n_to_plot]:

        # Title
        location_id = predictions_df['pickup_location_id'].iloc[row_id]
        location_name = predictions_df['zone'].iloc[row_id]
        st.header(f'Location ID: {location_id} - {location_name}')

        # Plot predictions
        prediction = predictions_df['predicted_demand'].iloc[row_id]
        st.metric(label="Predicted demand", value=int(prediction))

        # Plot figure
        # Generate figure
        fig = plot_one_sample(
            example_id=row_id,
            features=features_df,
            targets=predictions_df['predicted_demand'],
            predictions=pd.Series(predictions_df['predicted_demand']),
            display_title=False,
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(6 / N_STEPS)  # Updating the progress bar
