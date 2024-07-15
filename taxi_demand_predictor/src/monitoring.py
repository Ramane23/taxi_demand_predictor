# Import datetime and timedelta classes from the datetime module
from datetime import datetime, timedelta

# Import ArgumentParser class from the argparse module
from argparse import ArgumentParser

# Import pandas library and alias it as pd
import pandas as pd
import sys
import os

# Adjust the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the config module from the src package
import src.config as config

# Import the get_logger function from the src.logger module
from src.logger import get_logger

# Import FEATURE_GROUP_PREDICTIONS_METADATA and FEATURE_GROUP_METADATA from src.config module
from src.config import FEATURE_GROUP_PREDICTIONS_METADATA, FEATURE_GROUP_METADATA

# Import get_or_create_feature_group and get_feature_store functions from src.feature_store_api module
from src.feature_store_api import get_or_create_feature_group, get_feature_store

# Initialize the logger
logger = get_logger()

# Define a function to load predictions and actual values from the feature store
# Import datetime and timedelta classes from the datetime module
from datetime import datetime, timedelta

# Import ArgumentParser class from the argparse module
from argparse import ArgumentParser

# Import pandas library and alias it as pd
import pandas as pd

# Import the config module from the src package
import src.config as config

# Import the get_logger function from the src.logger module
from src.logger import get_logger

# Import FEATURE_GROUP_PREDICTIONS_METADATA and FEATURE_GROUP_METADATA from src.config module
from src.config import FEATURE_GROUP_PREDICTIONS_METADATA, FEATURE_GROUP_METADATA

# Import get_or_create_feature_group and get_feature_store functions from src.feature_store_api module
from src.feature_store_api import get_or_create_feature_group, get_feature_store

# Initialize the logger
logger = get_logger()

# Define a function to load predictions and actual values from the feature store
def load_predictions_and_actual_values_from_store(
    from_date: datetime,  # Starting date for fetching data
    to_date: datetime,    # Ending date for fetching data
) -> pd.DataFrame:       # The function returns a pandas DataFrame
    """
    Fetches model predictions and actual values from `from_date` to `to_date` from the Feature Store and returns a dataframe.

    Args:
        from_date (datetime): min datetime for which we want predictions and actual values
        to_date (datetime): max datetime for which we want predictions and actual values

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
            - `rides`
    """
    # Get or create the feature group for predictions metadata
    predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
    
    # Get or create the feature group for actuals metadata
    actuals_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    # Convert from_date and to_date to timestamps in milliseconds
    from_ts = int(from_date.timestamp() * 1000)
    to_ts = int(to_date.timestamp() * 1000)
    
    # Create a query to join the predictions and actuals feature groups by `pickup_hour` and `pickup_location_id`
    query = predictions_fg.select_all() \
        .join(actuals_fg.select(['pickup_location_id', 'pickup_hour']),
              on=['pickup_hour', 'pickup_location_id'], prefix=None) \
        .filter(predictions_fg.pickup_ts >= from_ts) \
        .filter(predictions_fg.pickup_ts <= to_ts)

    # Get the feature store instance
    feature_store = get_feature_store()
    
    try:
        # Try to create the feature view if it does not already exist
        feature_store.create_feature_view(
            name=config.MONITORING_FV_NAME,      # Feature view name
            version=config.MONITORING_FV_VERSION, # Feature view version
            query=query                          # Query to create the feature view
        )
    except:
        # If feature view already exists, log the info and skip creation
        logger.info('Feature view already existed. Skip creation.')

    # Get the feature view for monitoring
    monitoring_fv = feature_store.get_feature_view(
        name=config.MONITORING_FV_NAME,         # Feature view name
        version=config.MONITORING_FV_VERSION    # Feature view version
    )
    
    # Fetch data from the feature view for the last 30 days
    monitoring_df = monitoring_fv.get_batch_data(
        start_time=from_date - timedelta(days=7), # Start time for data fetch
        end_time=to_date + timedelta(days=7),     # End time for data fetch
    )

    # Filter data to the specific time period we are interested in
    pickup_ts_from = int(from_date.timestamp() * 1000)
    pickup_ts_to = int(to_date.timestamp() * 1000)
    monitoring_df = monitoring_df[monitoring_df.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

    # Return the filtered DataFrame
    return monitoring_df

# If this script is run as the main module
if __name__ == '__main__':

    # Create an argument parser object
    parser = ArgumentParser()
    
    # Add an argument for `from_date` with a specific format
    parser.add_argument('--from_date',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    
    # Add an argument for `to_date` with a specific format
    parser.add_argument('--to_date',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function to load predictions and actual values from the store
    monitoring_df = load_predictions_and_actual_values_from_store(args.from_date, args.to_date)

    # Display the resulting DataFrame
    print(monitoring_df)
