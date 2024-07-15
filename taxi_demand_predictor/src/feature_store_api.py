from typing import Optional, List
from dataclasses import dataclass

import hsfs
import hopsworks
import sys
import os

# Adjust the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.config as config
from src.logger import get_logger
from src.feature_store_config import FeatureGroupConfig, FeatureViewConfig  # Import from feature_store_config

# Get a logger instance to log messages
logger = get_logger()


def get_feature_store() -> hsfs.feature_store.FeatureStore:
    """Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    # Log in to Hopsworks and get the project
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    # Return the feature store for the project
    return project.get_feature_store()

def get_feature_group(
    name: str,
    version: Optional[int] = 1
) -> hsfs.feature_group.FeatureGroup:
    """Connects to the feature store and returns a pointer to the given feature group `name`

    Args:
        name (str): name of the feature group
        version (Optional[int], optional): _description_. Defaults to 1.

    Returns:
        hsfs.feature_group.FeatureGroup: pointer to the feature group
    """
    # Get the feature group from the feature store
    return get_feature_store().get_feature_group(
        name=name,
        version=version,
    )

def get_or_create_feature_group(
    feature_group_metadata: FeatureGroupConfig
) -> hsfs.feature_group.FeatureGroup:
    """Connects to the feature store and returns a pointer to the given feature group `name`

    Args:
        feature_group_metadata (FeatureGroupConfig): Metadata for the feature group

    Returns:
        hsfs.feature_group.FeatureGroup: pointer to the feature group
    """
    # Get or create the feature group in the feature store
    return get_feature_store().get_or_create_feature_group(
        name=config.FEATURE_GROUP_METADATA.name,
        version=config.FEATURE_GROUP_METADATA.version,
        description=config.FEATURE_GROUP_METADATA.description,
        primary_key=config.FEATURE_GROUP_METADATA.primary_key,
        event_time=config.FEATURE_GROUP_METADATA.event_time,
        online_enabled=config.FEATURE_GROUP_METADATA.online_enabled
    )

def get_or_create_feature_view(
    feature_view_metadata: FeatureViewConfig
) -> hsfs.feature_view.FeatureView:
    """Connects to the feature store and returns a pointer to the given feature view `name`

    Args:
        feature_view_metadata (FeatureViewConfig): Metadata for the feature view

    Returns:
        hsfs.feature_view.FeatureView: pointer to the feature view
    """
    # Get pointer to the feature store
    feature_store = get_feature_store()

    # Get pointer to the feature group
    feature_group = feature_store.get_feature_group(
        name=feature_view_metadata.feature_group.name,
        version=feature_view_metadata.feature_group.version
    )

    # Create feature view if it doesn't exist
    try:
        feature_store.create_feature_view(
            name=feature_view_metadata.name,
            version=feature_view_metadata.version,
            query=feature_group.select_all()
        )
    except:
        logger.info("Feature view already exists, skipping creation.")
    
    # Get feature view
    feature_store = get_feature_store()
    feature_view = feature_store.get_feature_view(
        name=feature_view_metadata.name,
        version=feature_view_metadata.version,
    )

    return feature_view
