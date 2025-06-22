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

    def _create_session_ids(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Create session IDs based on time gaps between user actions.
        '''
        df = df.sort_values(['user_id', 'timestamp'])

        # Calculate time difference between consecutive actions
        time_diff = df.groupby('user_id')['timestamp'].diff()

        # New session starts when time difference > session_timeout
        new_session = (time_diff > self.session_timeout.total_seconds()).astype(int)

        # Cumulative sum of new_session gives session IDs
        df['session_id'] = (
            df['user_id'].astype(str) + '_' +
            new_session.groupby(df['user_id']).cumsum().astype(str)
        )

        return df

    def _create_item_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Create item-level features:
        - Total views (count)
        - Amount of unique viewers (nunique)
        - Behavior count: ['pv', 'fav', 'cart', 'buy']
        - Conversion rate: view_to_cart_rate, cart_to_buy_rate, view_to_buy_rate
        - Popularity score: Weighted combination of different interactions
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
            behavior_counts['buy']
            .div(behavior_counts['cart'])
            .where(behavior_counts['cart'] > 0, 0)
        )
        item_features['view_to_buy_rate'] = {
            behavior_counts['buy']
            .div(behavior_counts['pv'])
            .where(behavior_counts['pv'] > 0, 0)
        }

        # Popularity score (weighted combination of different interactions)
        weights = {'pv': 1, 'fav': 2, 'cart': 3, 'buy': 4}
        item_features['popularity_score'] = sum(behavior_counts[btype] * weight for btype, weight in weights.item())

        return item_features
    
    def _create_user_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Create user-level features:
        - Total actions
        - First action
        - Last action
        - Behavior counts
        - Unique items viewed
        - Unique categories viewed
        - Number of sessions
        '''
        # Activity level features
        user_features = (
            df
            .groupby('user_id')
            .agg({
                'timestamp': ['count', 'min', 'max'], # Total actions, First action, Last action
                'behavior_type': lambda x: x.value_counts().to_dict(), # Behavior counts
                'item_id': 'nunique', # Unique items viewed
                'category_id': 'nunique', # Unique categories viewed
                'session_id': 'nunique' # Number of sessions
            })
            .reset_index()
        )

        # Flatten behavior counts
        behavior_counts = pd.DataFrame(user_features[('behavior_type', '<lambda>')].to_list())
        behavior_counts = behavior_counts.fillna(0)

        # Calculate derived metrics
        user_features['total_actions'] = user_features[('timestamp', 'count')]
        user_features['account_age_days'] = (
            user_features[('timestamp', 'max')] - user_features[('timestamp', 'min')]
        ).dt.total_seconds() / (24 * 3600)

        # Conversion rates
        user_features['cart_to_buy_ratio'] = (
            behavior_counts['buy']
            .div(behavior_counts['cart'])
            .where(behavior_counts['cart'] > 0, 0)
        )
        user_features['fav_to_buy_ratio'] = (
            behavior_counts['buy']
            .div(behavior_counts['fav'])
            .where(behavior_counts['fav'] > 0, 0)
        )
        user_features['view_to_buy_ratio'] = {
            behavior_counts['buy']
            .div(behavior_counts['pv'])
            .where(behavior_counts['pv'] > 0, 0)
        }

        # Activity intensity
        user_features['actions_per_day'] = (
            user_features['total_actions']
            .div(user_features['account_age_days'])
            .where(user_features['account_age_days'] > 0, 0)
        )
        user_features['unique_items_per_session'] = user_features[('item_id', 'nunique')] / user_features[('session_id', 'nunique')]

        return user_features

    def _create_session_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Create session-level features:
        - Action counts per session per user
        - Session duration
        - Amount of unique categories per session
        - Behavior counts per session per user
        '''

        session_features = (
            df
            .groupby(['unique_id', 'session_id'])
            .agg({
                'timestamp': ['count', lambda x: (x.max() - x.min()).total_seconds()], # Action counts & Session duration
                'item_id': 'nunique', # Unique items in session
                'category_id': 'nunique', # Unique categories in session
                'behavior_type': lambda x: x.value_counts().to_dict()  # Behavior counts in session
            })
            .reset_index()
        )

        # Session duration and intensity metrics
        session_features['session_duration_minutes'] = session_features[('timestamp', '<lambda>')] / 60
        session_features['actions_per_minute'] = (
            session_features[('timestamp', 'count')]
            .div(session_features['session_duration_minutes'])
            .where(session_features['session_duration_minutes'] > 0, 0)
        )

        return session_features
    
    def _create_time_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Create time-based features:
        - Hour (present by default as 'hour')
        - Day of the week (present by default as 'day_of_week')
        - Weekend (dummy variable)
        '''
        # Ensure that 'timestamp' column is of datetime type
        if df['timestamp'].dtype != 'datetime64[ns]':
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Check if any of the features is present in the original DataFrame
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
        
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek

        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time since last action (any type)
        df['time_since_last_action'] = (
            df.groupby('user_id')['timestamp']
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )

        # Time since last purchase (if any)
        last_purchase = (
            df[df['behavior_type'] == 'buy']
            .groupby('user_id')['timestamp']
            .transform('max')
        )
        df['time_since_last_purchase'] = (
            (df['timestamp'] - last_purchase).dt.total_seconds()
        )

        return df