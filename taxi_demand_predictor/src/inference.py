from datetime import datetime, timedelta
import hopsworks
#from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np
import src.config as config

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
def get_feature_store ():
    
    project = get_hopsworks_project()
    return project.get_feature_store()

def load_batch_of_features_from_store(current_date: pd.Timestamp) -> pd.DataFrame:
    """Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 4 columns:
            - `pickup_hour`
            - `rides`
            - `pickup_location_id`
    """
    # Convert current_date to timezone-aware datetime
    current_date = current_date.tz_localize('UTC')

    feature_store = get_feature_store()
    n_features = config.N_FEATURES
    # Read time_series data from the feature_store
    fetch_data_from = current_date - timedelta(days=28)
    fetch_data_to = current_date - timedelta(hours=1)
    print(f"fetching data from {fetch_data_from} to {fetch_data_to}")
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1))
    )

    # Convert pickup_hour column to UTC if it is not already
    if ts_data['pickup_hour'].dt.tz is None:
        ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour']).dt.tz_localize('UTC')
    else:
        ts_data['pickup_hour'] = ts_data['pickup_hour'].dt.tz_convert('UTC')

    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    # Validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()
    assert len(ts_data) == n_features * len(location_ids), "Time-series data is not complete. Make sure your feature pipeline is up and running."
    # Sort by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    # Transform time-series data into a feature vector for each `pickup_location_id`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values
    # Convert numpy arrays to pandas DataFrame
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features
def load_model_from_registry():
    """
    Loads the model from the Hopsworks model registry.

    Returns:
        The loaded machine learning model.
    """
    # Import joblib for loading the model and Path for handling filesystem paths
    import joblib
    from pathlib import Path

    # Retrieve the Hopsworks project instance
    project = get_hopsworks_project()

    # Get the model registry from the project
    model_registry = project.get_model_registry()

    # Fetch the model metadata from the registry using the name and version specified in the config
    model = model_registry.get_model(
        name=config.MODEL_NAME,  # Model name from the config
        version=config.MODEL_VERSION,  # Model version from the config
    )  

    # Download the model files to a local directory
    model_dir = model.download()

    # Load the model from the downloaded 'model.pkl' file
    model = joblib.load(Path(model_dir) / 'model.pkl')
       
    # Return the loaded model
    return model

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for taxi demand based on input features using the provided model."""
    
    # Generate predictions using the provided model and the input features
    predictions = model.predict(features)
    
    # Create a new DataFrame to store the results
    results = pd.DataFrame()
    
    # Copy the 'pickup_location_id' column from the input features to the results DataFrame
    results['pickup_location_id'] = features['pickup_location_id'].values
    
    # Add the predictions to the results DataFrame, rounding the values to the nearest integer
    results['predicted_demand'] = predictions.round(0)
    
    # Return the DataFrame containing pickup locations and their predicted demand
    return results