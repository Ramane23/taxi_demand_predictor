import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = 'taxi_demand_forecasst'
try:
    # HOPSWORKS_PROJECT_NAME = os.environ['HOPSWORKS_PROJECT_NAME']
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception(
        'Create an .env file on the project root with the HOPSWORKS_API_KEY'
    )

# TODO: remove FEATURE_GROUP_NAME and FEATURE_GROUP_VERSION, and use FEATURE_GROUP_METADATA instead
FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "time_series_hourly_feature_view"
FEATURE_VIEW_VERSION = 1
# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28
MODEL_NAME = 'taxi_demand_predictor_next_hour'
MODEL_VERSION = 1