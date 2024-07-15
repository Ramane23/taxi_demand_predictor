# Import the datetime and timedelta classes from the datetime module
from datetime import datetime, timedelta

# Import the numpy library and alias it as np
import numpy as np

# Import the pandas library and alias it as pd
import pandas as pd

# Import the streamlit library and alias it as st
import streamlit as st

# Import the mean_absolute_error function from sklearn.metrics module
from sklearn.metrics import mean_absolute_error

# Import the plotly.express library and alias it as px
import plotly.express as px

import sys
import os

# Adjust the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the load_predictions_and_actual_values_from_store function from the src.monitoring module
from src.monitoring import load_predictions_and_actual_values_from_store

# Set the configuration for the Streamlit app layout to wide
st.set_page_config(layout="wide")

# Title
# Get the current date and time in UTC, rounded to the nearest hour
current_date = pd.Timestamp(datetime.utcnow()).floor('H').tz_localize('UTC')

# Set the title of the Streamlit app
st.title(f'Monitoring dashboard 🔎')

# Initialize a progress bar in the sidebar with a header
progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)  # Set the initial progress to 0
N_STEPS = 3  # Define the number of steps for the progress bar

# Define a function to load predictions and actual values from the store with Streamlit caching
@st.cache_data
def _load_predictions_and_actuals_from_store(
    from_date: pd.Timestamp,  # Starting date for fetching data
    to_date: pd.Timestamp     # Ending date for fetching data
) -> pd.DataFrame:           # The function returns a pandas DataFrame
    """
    Wrapped version of src.monitoring.load_predictions_and_actual_values_from_store, so
    we can add Streamlit caching

    Args:
        from_date (pd.Timestamp): min datetime for which we want predictions and actual values
        to_date (pd.Timestamp): max datetime for which we want predictions and actual values

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
            - `rides`
    """
    # Call the load_predictions_and_actual_values_from_store function from src.monitoring module
    return load_predictions_and_actual_values_from_store(from_date, to_date)

# Use Streamlit's spinner to show a loading message while fetching data
with st.spinner(text="Fetching model predictions and actual values from the store"):
    
    # Call the function to load predictions and actual values from the store
    monitoring_df = _load_predictions_and_actuals_from_store(
        from_date = current_date - timedelta(days=14),  # Start date is 14 days before the current date
        to_date = current_date                          # End date is the current date
    )
    
    # Update the sidebar to indicate that data has been fetched
    st.sidebar.write('✅ Model predictions and actual values arrived')
    # Update the progress bar to the next step
    progress_bar.progress(1/N_STEPS)
    # Print the first few rows of the DataFrame to the console (for debugging purposes)
    print(monitoring_df.head())

# Use Streamlit's spinner to show a loading message while plotting MAE hour-by-hour
with st.spinner(text="Plotting aggregate MAE hour-by-hour"):
    
    # Set the header of the section
    st.header('Mean Absolute Error (MAE) hour-by-hour')

    # Calculate the Mean Absolute Error (MAE) per pickup_hour
    mae_per_hour = (
        monitoring_df
        .groupby('pickup_hour')  # Group by pickup_hour
        .apply(lambda g: mean_absolute_error(g['rides'], g['predicted_demand']))  # Apply MAE calculation
        .reset_index()  # Reset the index
        .rename(columns={0: 'mae'})  # Rename the MAE column
        .sort_values(by='pickup_hour')  # Sort by pickup_hour
    )

    # Create a bar plot using Plotly Express
    fig = px.bar(
        mae_per_hour,
        x='pickup_hour', y='mae',
        template='plotly_dark',
    )
    
    # Display the plot in the Streamlit app
    st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    # Update the progress bar to the next step
    progress_bar.progress(2/N_STEPS)

# Use Streamlit's spinner to show a loading message while plotting MAE hour-by-hour for top locations
with st.spinner(text="Plotting MAE hour-by-hour for top locations"):
    
    # Set the header of the section
    st.header('Mean Absolute Error (MAE) per location and hour')

    # Get the top 10 locations by demand
    top_locations_by_demand = (
        monitoring_df
        .groupby('pickup_location_id')['rides']  # Group by pickup_location_id and sum the rides
        .sum()
        .sort_values(ascending=False)  # Sort in descending order
        .reset_index()
        .head(10)['pickup_location_id']  # Get the top 10 locations
    )

    # Loop through each top location
    for location_id in top_locations_by_demand:
        
        # Calculate the Mean Absolute Error (MAE) per pickup_hour for the current location
        mae_per_hour = (
            monitoring_df[monitoring_df.pickup_location_id == location_id]  # Filter for the current location
            .groupby('pickup_hour')  # Group by pickup_hour
            .apply(lambda g: mean_absolute_error(g['rides'], g['predicted_demand']))  # Apply MAE calculation
            .reset_index()  # Reset the index
            .rename(columns={0: 'mae'})  # Rename the MAE column
            .sort_values(by='pickup_hour')  # Sort by pickup_hour
        )

        # Create a bar plot using Plotly Express for the current location
        fig = px.bar(
            mae_per_hour,
            x='pickup_hour', y='mae',
            template='plotly_dark',
        )
        
        # Set a subheader for the current location and display the plot
        st.subheader(f'{location_id=}')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    # Update the progress bar to the final step
    progress_bar.progress(3/N_STEPS)
