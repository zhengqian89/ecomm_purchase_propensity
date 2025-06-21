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
        # Popularity metrics
        item_features = df.groupby('item_id').agg({
            'user_id': ['count', 'nunique'], # total views and unique viewers
            'behavior_type': lambda x: x.value_counts().to_dict() # behavior counts; A column of dictionary with key being behavior type and value being counts for each item
        }).reset_index()

        # Flatten behavior counts
        behavior_counts = pd.DataFrame(item_features[('behavior_type', '<lambda>')].to_list())
        behavior_counts = behavior_counts.fillna(0) # Not every item has all behavior types

        # Calculate conversion rates
        item_features['view_to_cart_rate'] = (
            behavior_counts['cart']
            .div(behavior_counts['pv'])
            .where(behavior_counts['pv'] > 0, 0)
        )
        item_features['cart_to_buy_rate'] = (
            behavior_counts['cart']
            .div(behavior_counts['buy'])
            .where(behavior_counts['buy'] > 0, 0)
        )
        item_features['view_to_buy_rate'] = {
            behavior_counts['pv']
            .div(behavior_counts['buy'])
            .where(behavior_counts['buy'] > 0, 0)
        }

        # Popularity score (weighted combination of different interactions)
        weights = {'pv': 1, 'fav': 2, 'cart': 3, 'buy': 4}
        item_features['popularity_score'] = sum(behavior_counts[btype] * weight for btype, weight in weights.item())

        return item_features