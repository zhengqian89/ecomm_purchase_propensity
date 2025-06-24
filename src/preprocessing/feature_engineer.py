import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils.logger import get_logger
from typing import Optional
import traceback

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
        try:
            self.session_timeout = timedelta(minutes=session_timeout_minutes)
            logger.info(f'FeatureEngineer initialized with session timeout: {session_timeout_minutes} minutes')
        except Exception as e:
            logger.error(f'Error initializing FeatureEngineer: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Flatten MultiIndex columns created by groupby aggregation.
        '''
        try:
            if isinstance(df.columns, pd.MultiIndex):
                # Create new column names by joining the levels
                df.columns = ['_'.join(col).strip('_') if col[1] != '' else col[0] 
                             for col in df.columns.values]
                logger.debug(f'Flattened MultiIndex columns: {list(df.columns)}')
            return df
        except Exception as e:
            logger.error(f'Error flattening columns: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def create_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Feature engineering pipeline.

        Parameters:
            df: DataFrame with preprocessed user behavior data, containing:
            ['user_id', 'item_id', 'category_id', 'behavior_type', 'datetime', 'hour', 'day_of_week', 'date']
        
        Returns:
            DataFrame with engineered features.
        '''
        try:
            logger.info('Starting feature engineering pipeline...')
            logger.info(f'Input DataFrame shape: {df.shape}')
            logger.info(f'Input DataFrame columns: {list(df.columns)}')

            # Validate input DataFrame
            required_columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'datetime']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f'Missing required columns: {missing_columns}')

            # Create session IDs
            df = self._create_session_ids(df)

            # Create user-level features
            user_features = self._create_user_features(df)
            logger.info(f'User features shape: {user_features.shape}')

            # Create item-level features
            item_features = self._create_item_features(df)
            logger.info(f'Item features shape: {item_features.shape}')
            
            # Create category-level features
            category_features = self._create_category_features(df)
            logger.info(f'Category features shape: {category_features.shape}')

            # Create session-level features
            session_features = self._create_session_features(df)
            logger.info(f'Session features shape: {session_features.shape}')

            # Create time-based features
            time_features = self._create_time_features(df)
            logger.info(f'Time features shape: {time_features.shape}')

            # Combine all features
            logger.info('Starting feature merging...')
            features = (
                df
                .merge(user_features, on='user_id', how='left')
                .merge(item_features, on='item_id', how='left')
                .merge(category_features, on='category_id', how='left')
                .merge(session_features, on=['user_id', 'session_id'], how='left')
            )

            # Add time features
            features = pd.concat([features, time_features], axis=1)

            logger.info(f'Feature engineering completed. Final shape: {features.shape}')
            logger.info(f'Generated {len(features.columns)} features.')

            return features

        except Exception as e:
            logger.error(f'Error in feature engineering pipeline: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def _create_session_ids(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Create session IDs based on time gaps between user actions.
        '''
        try:
            logger.info('Creating session ids...')
            
            # Validate datetime column
            if 'datetime' not in df.columns:
                raise ValueError("'datetime' column not found in DataFrame")
            
            # Ensure datetime is properly formatted
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                logger.info('Converting datetime column to datetime type')
                df['datetime'] = pd.to_datetime(df['datetime'])

            df = df.sort_values(['user_id', 'datetime'])

            # Calculate time difference between consecutive actions
            time_diff = df.groupby('user_id')['datetime'].diff()

            # New session starts when time difference > session_timeout
            # Fix: Compare timedelta with timedelta, not seconds
            new_session = (time_diff > self.session_timeout).astype(int)

            # Cumulative sum of new_session gives session IDs
            df['session_id'] = (
                df['user_id'].astype(str) + '_' +
                new_session.groupby(df['user_id']).cumsum().astype(str)
            )

            logger.info(f'Session ids created successfully. Found {df["session_id"].nunique()} unique sessions')
            return df

        except Exception as e:
            logger.error(f'Error creating session IDs: {str(e)}')
            logger.error(traceback.format_exc())
            raise
    
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
        try:
            logger.info('Creating item features...')

            # Validate required columns
            if 'item_id' not in df.columns or 'behavior_type' not in df.columns:
                raise ValueError("Required columns 'item_id' or 'behavior_type' not found")

            # Popularity metrics
            item_features = df.groupby('item_id').agg({
                'user_id': ['count', 'nunique'], # total views and unique viewers
                'behavior_type': lambda x: x.value_counts().to_dict() # behavior counts
            }).reset_index()

            # Flatten the MultiIndex columns
            item_features = self._flatten_columns(item_features)

            # Rename columns for clarity
            item_features = item_features.rename(columns={
                'user_id_count': 'total_views',
                'user_id_nunique': 'unique_viewers',
                'behavior_type_<lambda>': 'behavior_counts'
            })

            # Flatten behavior counts
            behavior_counts = pd.DataFrame(item_features['behavior_counts'].to_list())
            behavior_counts = behavior_counts.fillna(0) # Not every item has all behavior types

            # Add behavior counts as separate columns
            for behavior in ['pv', 'fav', 'cart', 'buy']:
                item_features[f'{behavior}_count'] = behavior_counts.get(behavior, 0)

            # Calculate conversion rates with error handling
            item_features['view_to_cart_rate'] = np.where(
                item_features['pv_count'] > 0,
                item_features['cart_count'] / item_features['pv_count'],
                0
            )
            item_features['cart_to_buy_rate'] = np.where(
                item_features['cart_count'] > 0,
                item_features['buy_count'] / item_features['cart_count'],
                0
            )
            item_features['view_to_buy_rate'] = np.where(
                item_features['pv_count'] > 0,
                item_features['buy_count'] / item_features['pv_count'],
                0
            )

            # Popularity score (weighted combination of different interactions)
            weights = {'pv': 1, 'fav': 2, 'cart': 3, 'buy': 4}
            item_features['popularity_score'] = sum(
                item_features[f'{btype}_count'] * weight for btype, weight in weights.items()
            )

            # Drop the behavior_counts column as we've expanded it
            item_features = item_features.drop('behavior_counts', axis=1)

            logger.info(f'Item features created successfully. Shape: {item_features.shape}')
            return item_features

        except Exception as e:
            logger.error(f'Error creating item features: {str(e)}')
            logger.error(traceback.format_exc())
            raise

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
        try:
            logger.info('Creating category features...')

            # Validate required columns
            if 'category_id' not in df.columns:
                raise ValueError("Required column 'category_id' not found")

            category_features = (
                df
                .groupby('category_id')
                .agg({
                    'user_id': ['count', 'nunique'], # Total view and Unique viewers
                    'item_id': 'nunique', # Unique items in category
                    'behavior_type': lambda x: x.value_counts().to_dict() # Behavior counts
                })
                .reset_index()
            )
            
            # Flatten the MultiIndex columns
            category_features = self._flatten_columns(category_features)
            
            # Rename columns for clarity
            category_features = category_features.rename(columns={
                'user_id_count': 'category_total_views',
                'user_id_nunique': 'category_unique_viewers',
                'item_id_nunique': 'category_unique_items',
                'behavior_type_<lambda>': 'behavior_counts'
            })
            
            # Similar calculations as item features
            behavior_counts = pd.DataFrame(category_features['behavior_counts'].tolist())
            behavior_counts = behavior_counts.fillna(0)
            
            # Add behavior counts as separate columns
            for behavior in ['pv', 'fav', 'cart', 'buy']:
                category_features[f'category_{behavior}_count'] = behavior_counts.get(behavior, 0)
            
            category_features['category_conversion_rate'] = (
                category_features['category_buy_count'] / 
                (category_features['category_pv_count'] + 1)
            )
            category_features['category_popularity'] = category_features['category_total_views']
            
            # Drop the behavior_counts column
            category_features = category_features.drop('behavior_counts', axis=1)
            
            logger.info(f'Category features created successfully. Shape: {category_features.shape}')
            return category_features

        except Exception as e:
            logger.error(f'Error creating category features: {str(e)}')
            logger.error(traceback.format_exc())
            raise
    
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
        try:
            logger.info('Creating user features...')

            # Validate required columns
            required_cols = ['user_id', 'datetime', 'behavior_type', 'item_id', 'category_id', 'session_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f'Missing required columns for user features: {missing_cols}')

            # Activity level features
            user_features = (
                df
                .groupby('user_id')
                .agg({
                    'datetime': ['count', 'min', 'max'], # Total actions, First action, Last action
                    'behavior_type': lambda x: x.value_counts().to_dict(), # Behavior counts
                    'item_id': 'nunique', # Unique items viewed
                    'category_id': 'nunique', # Unique categories viewed
                    'session_id': 'nunique' # Number of sessions
                })
                .reset_index()
            )

            # Flatten the MultiIndex columns
            user_features = self._flatten_columns(user_features)
            
            # Rename columns for clarity
            user_features = user_features.rename(columns={
                'datetime_count': 'total_actions',
                'datetime_min': 'first_action',
                'datetime_max': 'last_action',
                'behavior_type_<lambda>': 'behavior_counts',
                'item_id_nunique': 'unique_items_viewed',
                'category_id_nunique': 'unique_categories_viewed',
                'session_id_nunique': 'num_sessions'
            })

            # Flatten behavior counts
            behavior_counts = pd.DataFrame(user_features['behavior_counts'].to_list())
            behavior_counts = behavior_counts.fillna(0)

            # Add behavior counts as separate columns
            for behavior in ['pv', 'fav', 'cart', 'buy']:
                user_features[f'user_{behavior}_count'] = behavior_counts.get(behavior, 0)

            # Calculate derived metrics with error handling
            user_features['account_age_days'] = (
                user_features['last_action'] - user_features['first_action']
            ).dt.total_seconds() / (24 * 3600)

            # Conversion rates with safe division
            user_features['cart_to_buy_ratio'] = np.where(
                user_features['user_cart_count'] > 0,
                user_features['user_buy_count'] / user_features['user_cart_count'],
                0
            )
            user_features['fav_to_buy_ratio'] = np.where(
                user_features['user_fav_count'] > 0,
                user_features['user_buy_count'] / user_features['user_fav_count'],
                0
            )
            user_features['view_to_buy_ratio'] = np.where(
                user_features['user_pv_count'] > 0,
                user_features['user_buy_count'] / user_features['user_pv_count'],
                0
            )

            # Activity intensity with safe division
            user_features['actions_per_day'] = np.where(
                user_features['account_age_days'] > 0,
                user_features['total_actions'] / user_features['account_age_days'],
                0
            )
            user_features['unique_items_per_session'] = np.where(
                user_features['num_sessions'] > 0,
                user_features['unique_items_viewed'] / user_features['num_sessions'],
                0
            )

            # Drop the behavior_counts column
            user_features = user_features.drop('behavior_counts', axis=1)

            logger.info(f'User features created successfully. Shape: {user_features.shape}')
            return user_features

        except Exception as e:
            logger.error(f'Error creating user features: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def _create_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create session-level features for each user session.
        '''
        try:
            logger.info('Creating session features...')

            # Validate required columns
            required_cols = ['user_id', 'session_id', 'datetime', 'item_id', 'category_id', 'behavior_type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f'Missing required columns for session features: {missing_cols}')

            # Use 'datetime' instead of 'timestamp' to match your data
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                logger.info('Converting datetime column for session features')
                df['datetime'] = pd.to_datetime(df['datetime'])

            session_features = (
                df
                .groupby(['user_id', 'session_id'])
                .agg(
                    actions_count=('datetime', 'count'),
                    session_duration=('datetime', lambda x: (x.max() - x.min()).total_seconds()),
                    unique_items=('item_id', 'nunique'),
                    unique_categories=('category_id', 'nunique'),
                    behavior_counts=('behavior_type', lambda x: x.value_counts().to_dict())
                )
                .reset_index()
            )

            # Fill NaN session_duration (single-action sessions) with 0
            session_features['session_duration'] = session_features['session_duration'].fillna(0)
            session_features['session_duration_minutes'] = session_features['session_duration'] / 60
            
            # Calculate actions per minute with safe division
            session_features['actions_per_minute'] = np.where(
                session_features['session_duration_minutes'] > 0,
                session_features['actions_count'] / session_features['session_duration_minutes'],
                0
            )

            # Flatten behavior counts for sessions
            behavior_counts = pd.DataFrame(session_features['behavior_counts'].to_list())
            behavior_counts = behavior_counts.fillna(0)

            # Add behavior counts as separate columns
            for behavior in ['pv', 'fav', 'cart', 'buy']:
                session_features[f'session_{behavior}_count'] = behavior_counts.get(behavior, 0)

            # Drop the behavior_counts column
            session_features = session_features.drop('behavior_counts', axis=1)

            logger.info(f'Session features created successfully. Shape: {session_features.shape}')
            return session_features

        except Exception as e:
            logger.error(f'Error creating session features: {str(e)}')
            logger.error(traceback.format_exc())
            raise

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
        try:
            logger.info('Creating time features...')

            # Validate datetime column
            if 'datetime' not in df.columns:
                raise ValueError("'datetime' column not found for time features")

            time_features = pd.DataFrame()
            # Ensure that 'datetime' column is of datetime type
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                logger.info('Converting datetime column for time features')
                df['datetime'] = pd.to_datetime(df['datetime'])

            # Check if any of the features is present in the original DataFrame
            if 'hour' not in df.columns:
                df['hour'] = df['datetime'].dt.hour
            time_features['hour'] = df['hour']
            
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['datetime'].dt.dayofweek
            time_features['day_of_week'] = df['day_of_week']

            if 'is_weekend' not in df.columns:
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            time_features['is_weekend'] = df['is_weekend']
            
            # Time since last action (any type)
            time_features['time_since_last_action'] = (
                df.groupby('user_id')['datetime']
                .diff()
                .dt.total_seconds()
                .fillna(0)
            )

            # Time since last purchase (if any)
            purchase_data = df[df['behavior_type'] == 'buy']
            if not purchase_data.empty:
                last_purchase = (
                    purchase_data
                    .groupby('user_id')['datetime']
                    .transform('max')
                )
                time_features['time_since_last_purchase'] = (
                    (df['datetime'] - last_purchase).dt.total_seconds()
                ).fillna(0)
            else:
                logger.warning('No purchase data found, setting time_since_last_purchase to 0')
                time_features['time_since_last_purchase'] = 0

            logger.info(f'Time features created successfully. Shape: {time_features.shape}')
            return time_features

        except Exception as e:
            logger.error(f'Error creating time features: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def save_features(self, df: pd.DataFrame, path: str):
        '''
        Save features to a parquet file.
        '''
        try:
            df.to_parquet(path, index=False)
            logger.info(f'Saved features to {path}. Shape: {df.shape}')
        except Exception as e:
            logger.error(f'Error saving features to {path}: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def load_features(self, path: str) -> pd.DataFrame:
        '''
        Load features from a parquet file.
        '''
        try:
            logger.info(f'Loading features from {path}')
            df = pd.read_parquet(path)
            logger.info(f'Loaded features from {path}. Shape: {df.shape}')
            return df
        except Exception as e:
            logger.error(f'Error loading features from {path}: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def generate_and_save_session_ids(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        try:
            logger.info(f'Generating and saving session IDs to {path}')
            df = self._create_session_ids(df)
            self.save_features(df, path)
            return df
        except Exception as e:
            logger.error(f'Error in generate_and_save_session_ids: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def generate_and_save_user_features(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        try:
            logger.info(f'Generating and saving user features to {path}')
            user_features = self._create_user_features(df)
            self.save_features(user_features, path)
            return user_features
        except Exception as e:
            logger.error(f'Error in generate_and_save_user_features: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def generate_and_save_item_features(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        try:
            logger.info(f'Generating and saving item features to {path}')
            item_features = self._create_item_features(df)
            self.save_features(item_features, path)
            return item_features
        except Exception as e:
            logger.error(f'Error in generate_and_save_item_features: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def generate_and_save_category_features(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        try:
            logger.info(f'Generating and saving category features to {path}')
            category_features = self._create_category_features(df)
            self.save_features(category_features, path)
            return category_features
        except Exception as e:
            logger.error(f'Error in generate_and_save_category_features: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def generate_and_save_session_features(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        try:
            logger.info(f'Generating and saving session features to {path}')
            session_features = self._create_session_features(df)
            self.save_features(session_features, path)
            return session_features
        except Exception as e:
            logger.error(f'Error in generate_and_save_session_features: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def generate_and_save_time_features(self, df: pd.DataFrame, path: str) -> pd.DataFrame:
        try:
            logger.info(f'Generating and saving time features to {path}')
            time_features = self._create_time_features(df)
            self.save_features(time_features, path)
            return time_features
        except Exception as e:
            logger.error(f'Error in generate_and_save_time_features: {str(e)}')
            logger.error(traceback.format_exc())
            raise

    def merge_all_features(
        self,
        base_df_path: str,
        user_features_path: str,
        item_features_path: str,
        category_features_path: str,
        session_features_path: str,
        time_features_path: str,
        output_path: str
    ) -> pd.DataFrame:
        '''
        Merge all feature blocks into a single DataFrame and save.
        '''
        try:
            logger.info('Starting to merge all features...')
            
            df = self.load_features(base_df_path)
            user_features = self.load_features(user_features_path)
            item_features = self.load_features(item_features_path)
            category_features = self.load_features(category_features_path)
            session_features = self.load_features(session_features_path)
            time_features = self.load_features(time_features_path)

            logger.info('Performing feature merges...')
            features = (
                df
                .merge(user_features, on='user_id', how='left')
                .merge(item_features, on='item_id', how='left')
                .merge(category_features, on='category_id', how='left')
                .merge(session_features, on=['user_id', 'session_id'], how='left')
            )
            features = pd.concat([features, time_features], axis=1)
            
            self.save_features(features, output_path)
            logger.info(f'All features merged and saved to {output_path}. Final shape: {features.shape}')

            return features

        except Exception as e:
            logger.error(f'Error in merge_all_features: {str(e)}')
            logger.error(traceback.format_exc())
            raise