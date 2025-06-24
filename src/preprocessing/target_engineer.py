import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils.logger import get_logger
import traceback

logger = get_logger(__name__)

class TargetEngineer:
    '''
    Create target variables for purchase propensity prediction.
    '''
    def __init__(self):
        logger.info('TargetEngineer instantiated...')
    
    def create_purchase_targets(
        self,
        df: pd.DataFrame,
        prediction_window_hours: int = 24,
        method: str = 'next_session',
        prepare_modeling_dataset: bool = True,
        remove_future_data: bool = True,
    ) -> pd.DataFrame:
        '''
        Create target variables for purchase propensity prediction
        
        Parameters:
        - df: DataFrame with user behavior data
        - prediction_window_hours: Hours ahead to predict purchase
        - method: 'next_session', 'time_window', or 'session_end'
        - remove_future_data: Whether to remove columns that could cause data leakage

        Returns:
        - DataFrame with target variables added, ready for modeling
        '''
        try:
            logger.info(f'Creating purchase targets with method: {method}, window: {prediction_window_hours}h')

            # Ensure datetime is properly sorted
            df = df.sort_values(['user_id', 'datetime']).reset_index(drop=True)

            if method == 'next_session':
                df = self._create_next_session_purchase_target(df)
            elif method == 'time_window':
                df = self._create_time_window_purchase_target(df, prediction_window_hours)
            elif method == 'session_end':
                df = self._create_session_end_purchase_target(df)
            else:
                raise ValueError(f'Unknown method: {method}')

            # Log target distribution
            self._log_target_distribution(df)

            logger.info('Purchase targets created successfully')

        except Exception as e:
            logger.error(f'Error creating purchase targets: {str(e)}')
            logger.error(traceback.format_exc())
            raise
        
        if not prepare_modeling_dataset:
            return df
        else:
            target_map = {
                'next_session': 'will_buy_next_session',
                'time_window': 'will_buy_in_window',
                'session_end': 'will_buy_in_session'
            }
            target_col = target_map[method]
            
            logger.info(f'Primary target column: {target_col}')

            # Prepare dataset for modeling byu removing future data leakage
            try:
                logger.info(f'Preparing modeling dataset with target: {target_col}')

                # Remove rows where target is NaN (i.e., no future data available)
                df_model = df.dropna(subset=[target_col]).copy()

                if remove_future_data:
                    # Columns that contain future information
                    future_cols = ['last_action', 'total_actions', 'account_age_days']

                    # Remove future data columns if they exist
                    cols_to_remove = [col for col in future_cols if col in df_model.columns]
                    if cols_to_remove:
                        df_model = df_model.drop(columns=cols_to_remove)
                        logger.info(f'Removed future data columns: {cols_to_remove}')

                # Remove non-feature columns
                non_feature_cols = ['timestamp', 'datetime', 'date', 'session_id', 'behavior_type']
                cols_to_remove = [col for col in non_feature_cols if col in df_model.columns]
                df_model = df_model.drop(columns=cols_to_remove)

                logger.info(f'Modeling dataset prepared. Shape: {df_model.shape}')
                logger.info(f'Target distribution: {df_model[target_col].value_counts().to_dict()}')

                return df_model
            
            except Exception as e:
                logger.error(f'Error preparing modeling dataset: {str(e)}')
                logger.error(traceback.format_exc())
                raise

    def _create_time_window_purchase_target(
        self,
        df: pd.DataFrame,
        hours: int
    ) -> pd.DataFrame:
        '''
        Target: Will user make a purchase within the time window
        '''
        prediction_window = timedelta(hours=hours)

        # For each row, check if there is a purchase within the time window
        def has_future_purchase(group):
            # Retrieve all purchases' datetime
            purchases = group[group['behavior_type'] == 'buy']['datetime']

            # If no purchases, return a Series that is of same length as group with a value of 0
            if purchases.empty:
                return pd.Series(0, index=group.index)
            
            result = []
            for _, row in group.iterrows():
                current_time = row['datetime']
                future_purchases = purchases[purchases > current_time]
                has_purchase_in_window = any (
                    (purchase - current_time) <= prediction_window for purchase in future_purchases
                )
                result.append(int(has_purchase_in_window))
            
            return pd.Series(result, index=group.index)

        df['will_buy_in_window'] = df.groupby('user_id').apply(has_future_purchase).values

        return df

    def _create_session_end_purchase_target(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Target: Will user make a purchase in their NEXT session?
        '''
        # Find the last action in each session
        session_last_actions = df.groupby(['user_id', 'session_id'])['datetime'].transform('max')
        df['is_session_end'] = (df['datetime'] == session_last_actions)

        # Check if session contains any purchase
        session_purchases = (
            df
            .groupby(['user_id', 'session_id'])['behavior_type']
            .apply(lambda x: (x == 'buy').any())
            .reset_index()
        )
        session_purchases.columns = ['user_id', 'session_id', 'session_has_purchase']

        # Merge with df
        df = df.merge(session_purchases, on=['user_id', 'session_id'], how='left')
        df['will_buy_in_session'] = df['session_has_purchase'].astype(int)
        df.drop(['is_session_end', 'session_has_purchase'], axis=1, inplace=True)

        return df
    
    def _create_next_session_purchase_target(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Target: Will user make a purchase in their next session?
        '''
        # Get next session for each user
        df['next_session_id'] = df.groupby('user_id')['session_id'].shift(-1)

        # Find sessions with purchases
        session_with_purchases = df[df['behavior_type'] == 'buy']['session_id'].unique()

        # Create target: 1 if next session has purchase
        df['will_buy_next_session'] = df['next_session_id'].isin(session_with_purchases).astype(int)

        # Assign last observation which has no valid next session with null and drop accordingly
        df['will_buy_next_session'] = df['will_buy_next_session'].where(
            df['next_session_id'].notna(), np.nan
        )
        df.dropna(subset=['will_buy_next_session'], inplace=True)

        # Drop helper column
        df = df.drop('next_session_id', axis=1)

        # Cast back to int
        df['will_buy_next_session'] = df['will_buy_next_session'].astype(int)
        return df

    def _log_target_distribution(self, df: pd.DataFrame):
        '''
        Log the distribution of target variables.
        '''
        target_cols = [col for col in df.columns if col.startswith('will_')]
        
        for col in target_cols:
            distribution = df[col].value_counts(dropna=False)
            logger.info(f'Target {col} distribution: {distribution.to_dict()}')
            
            # Calculate positive class percentage
            if not df[col].isna().all():
                pos_rate = df[col].mean() * 100
                logger.info(f'Target {col} positive rate: {pos_rate:.2f}%')