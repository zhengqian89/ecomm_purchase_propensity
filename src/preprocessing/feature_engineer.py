import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    '''
    Feature engineering for the ecommerce purchase propensity model.
    '''

    def __init__(
        self,
        session_timeout_minutes: int = 30,
    ) -> None:
        '''
        Initialize feature engineer.

        Parameters:
            session_timeout_minutes: Minutes of inactivity to define a new session.
        '''
        self.session_timeout = timedelta(minutes=session_timeout_minutes)

    def _create_item_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Create item-level features.
        '''
        item_features = df.groupby('item_id').agg({
            # Popularity metrics
            'user_id': ['count', 'nunique'],
            'behavior_type': lambda x: x.value_counts().to_dict()
        }).reset_index()