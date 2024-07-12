from typing import Optional, List
from dataclasses import dataclass

@dataclass
class FeatureGroupConfig:
    """Data class to store configuration for a feature group"""
    name: str
    version: int
    description: str
    primary_key: List[str]
    event_time: str
    online_enabled: Optional[bool] = False

@dataclass
class FeatureViewConfig:
    """Data class to store configuration for a feature view"""
    name: str
    version: int
    feature_group: FeatureGroupConfig