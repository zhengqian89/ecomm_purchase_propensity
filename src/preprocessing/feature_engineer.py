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

    def create_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Feature engineering pipeline.

        Parameters:
            df: DataFrame with preprocessed user behavior data, containing:
            ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp', 'datetime', 'hour', 'day_of_week', 'date']
        
        Returns:
            DataFrame with engineered features.
        '''
        logger.info('Starting feature engineering pipeline...')

        # Create session IDs
        df = self._create_session_ids(df)

        # Create user-level features
        user_features = self._create_user_features(df)

        # Create item-level features
        item_features = self._create_item_features(df)
        
        # Create category-level features
        category_features = self._create_category_features(df)

        # Create session-level features
        session_features = self._create_session_features(df)

        # Create time-based features
        time_features = self._create_time_features(df)

        # Combine all features
        features = (
            df
            .merge(user_features, on='user_id', how='left')
            .merge(item_features, on='item_id', how='left')
            .merge(category_features, on='category_id', how='left')
            .merge(session_features, on=['user_id', 'session_id'], how='left')
        )

        # Add time features
        features = pd.concat([features, time_features], axis=1)

        logger.info(f'Feature engineering completed. Generated {len(features.columns)} features.')

        return features

    def _create_session_ids(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Create session IDs based on time gaps between user actions.
        '''
        logger.info('Creating session id...')

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

        logger.info('Session ids created successfully')

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
        logger.info('Creating item features...')

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

        logger.info('Item features created successfully')

        return item_features

    def _create_category_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Create category-level features:
        - Category performance metrics
        - Category popularity
        - Conversion rates by category
        '''
        logger.info('Creating category features...')

        category_features = (
            df
            .groupby('category_id')
            .agg({
                'user_id': ['count', 'nunique'], # Total view and Unique viewers
                'item_id': 'nunique', # Unique items in category
                'behavior_type': lambda x: x.value_counts.to_dict() # Behavior counts
            })
            .reset_index()
        )
        
        # Similar calculations as item features
        behavior_counts = pd.DataFrame(category_features[('behavior_type', '<lambda>')].tolist())
        behavior_counts = behavior_counts.fillna(0)
        
        category_features['category_conversion_rate'] = behavior_counts['buy'] / (behavior_counts['pv'] + 1)
        category_features['category_popularity'] = category_features[('user_id', 'count')]
        
        return category_features
    
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
        logger.info('Creating user features...')

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

        logger.info('User features created successfully')

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
        logger.info('Creating session features...')

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

        logger.info('Session features created successfully')

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
        logger.info('Creating time features...')

        time_features = pd.DataFrame()
        # Ensure that 'timestamp' column is of datetime type
        if df['timestamp'].dtype != 'datetime64[ns]':
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Check if any of the features is present in the original DataFrame
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
        time_features['hour'] = df['hour']
        
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        time_features['day_of_week'] = df['day_of_week']

        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        time_features['is_weekend'] = df['is_weekend']
        
        # Time since last action (any type)
        time_features['time_since_last_action'] = (
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
        time_features['time_since_last_purchase'] = (
            (df['timestamp'] - last_purchase).dt.total_seconds()
        )

        logger.info('Time features created successfully')

        return time_features

    def create_target_variable(
        self,
        df: pd.DataFrame,
        prediction_window: timedelta = timedelta(hours=24)
    ) -> pd.DataFrame:
        '''
        Create target variable: Will user purchase within prediction_window?

        Parameters:
        - df: DataFrame with user behavior data
        - prediction_window: Time window to predict purchase within

        Returns:
            DataFrame with target variable
        '''
        logger.info('Creating target variable...')

        df = df.sort_values(['user_id', 'timestamp'])

        # Within-user 'will a next action occur within the window'
        future_purchase_mask = (
            df
            .groupby('user_id')['timestamp']
            .apply(lambda timestamp: timestamp.shift(-1) - timestamp <= prediction_window)
            .reset_index(level=0, drop=True)
        )

        # Within-user 'is the next action a buy?'
        next_is_buy = (
            df
            .groupby('user_id')['behavior_type']
            .shift(-1) == 'buy'
        )

        df['will_purchase'] = (next_is_buy & future_purchase_mask).astype(int)

        logger.info('Target variable created successfully')

        return df
    
feature_engineer = FeatureEngineer()