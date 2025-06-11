'''
Data ingestion module for the eCommerce Purchase Propensity Engine.
'''
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path
import pandera.pandas as pa
from src.utils.config_loader import config_loader
from src.utils.logger import get_logger

# Set up and instantiate logger
logger = get_logger(__name__)

class ConfigLoader:
    '''Utility class to load configuration files.'''

    def __init__(self, config_dir: str = 'config'):
        '''
        Initialize data loader.
        '''
        self.config = config_loader.get_data_paths()
        self.model_config = config_loader.get_model_params()

    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        '''
        Load raw UserBehavior.csv data.

        Args:
            file_path: Optional path to the data file. Use config if not provided.

        Returns:
            DataFrame with raw user behavior data
        '''
        if file_path is None:
            file_path = self.config['raw_data_path']
        
        logger.info(f'Loading raw data from {file_path}')

        # Check if file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f'Data file not found: {file_path}')
        
        # Load data with appropriate column names
        column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

        try:
            df = pd.re(file_path, names=column_names, header=None)
            logger.info(f'Loaded {len(df):,} records from {file_path}')

            # Basic validation
            self._validate_data(df)
            self._validate_schema(df)

            return df
        except Exception as e:
            logger.error(f'Error loading data from {file_path}: {e}')
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        '''
        Validate the loaded data to ensure the data ingested is compatible with the pipeline
        and unexpected values aren't present.

        Args:
            df: DataFrame to validate
        '''
        logger.info('Validating loaded data...')

        # Check required columns
        required_columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')
        
        # Check for null values
        null_counts = df.isnull.sum()
        if null_counts.sum() > 0:
            logger.warning(f'Found null values in data:\n{null_counts[null_counts>0]}')
        
        # Check for non-positive timestamps
        invalid_timestamps = df[df['timestamp'] <= 0]
        if not invalid_timestamps.empty:
            before = len(df)
            df.drop(index=invalid_timestamps.index, inplace=True)
            after = len(df)
            dropped = before - after
            logger.warning(f'Dropped {dropped} rows with non-positive timestamps (<= 0). '
                           f'Before: {before:,}, After: {after:,}, Dropped: {dropped:,} ({dropped / before:.2%})')
        
        # Check behavior types
        '''
        - pv: Page view of an item's detail page, equivalent to an item click
        - buy: Purchase an item
        - cart: Add an item to the shopping cart
        - fav: Favor an item
        '''
        expected_behaviors = {'pv', 'buy', 'cart', 'fav'}
        actual_behaviors = set(df['behavior_type'].unique())
        unexpected_behaviors = actual_behaviors - expected_behaviors
        if unexpected_behaviors:
            logger.warning(f'Found unexpected behavior types: {unexpected_behaviors}')
        
        # Basic statistics
        logger.info(f'Data validation summary:')
        logger.info(f'  Users: {df['user_id'].nunique():,}')
        logger.info(f'  Items: {df['item_id'].nunique():,}')
        logger.info(f'  Categories: {df['category_id'].nunique():,}')
        logger.info(f'  Behavior distribution:\n{df['behavior_type'].value_counts()}')
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        '''
        Validate the data schema using Pandera.

        Args:
            df: DataFrame to validate
        '''
        logger.info('Running schema validation...')

        schema = pa.DataFrameSchema({
            'user_id': pa.Column(int, required=True, coerce=True, nullable=False),
            'item_id': pa.Column(int, required=True, coerce=True, nullable=False),
            'category_id': pa.Column(int, required=True, coerce=True, nullable=False),
            'behavior_type': pa.Column(str, required=True, coerce=True, nullable=False, checks=pa.Check.isin(['pv', 'buy', 'cart', 'fav'])),
            'timestamp': pa.Column(int, required=True, coerce=True, nullable=False, checks=pa.Check.greater_than(0))
        })

        try:
            schema.validate(df, lazy=True) # lazy=True to collect all validation errors for debugging; default is lazy=False
            logger.info('Schema validation passed.')
        except pa.errors.SchemaErrors as e:
            logger.error(f'Schema validation failed:\n{e.failure_cases}')
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Preprocess the raw data.

        Args:
            df: Raw DataFrame
        
        Returns:
            Preprocessed DataFrame
        '''
        logger.info('Preprocessing data...')

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['date'] = df['datetime'].dt.date
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info('Data preprocessing completed')
        return df

    def split_data_by_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''Split data into train/validation/test sets by time.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        '''
        logger.info('Splitting data by time...')
        
        # Get split ratios from config
        train_ratio = self.model_config['data_split']['train_ratio']
        val_ratio = self.model_config['data_split']['val_ratio']
        test_ratio = self.model_config['data_split']['test_ratio']
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp')
        
        # Calculate split indices
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f'Data split completed:')
        logger.info(f'  Train: {len(train_df):,} records ({len(train_df)/len(df)*100:.1f}%)')
        logger.info(f'  Validation: {len(val_df):,} records ({len(val_df)/len(df)*100:.1f}%)')
        logger.info(f'  Test: {len(test_df):,} records ({len(test_df)/len(df)*100:.1f}%)')
        
        # Log time ranges
        logger.info(f'Time ranges:')
        logger.info(f'  Train: {train_df['datetime'].min()} to {train_df['datetime'].max()}')
        logger.info(f'  Validation: {val_df['datetime'].min()} to {val_df['datetime'].max()}')
        logger.info(f'  Test: {test_df['datetime'].min()} to {test_df['datetime'].max()}')
        
        return train_df, val_df, test_df

    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame) -> None:
        '''Save processed data splits.
        
        Args:
            train_df: Training data
            val_df: Validation data  
            test_df: Test data
        '''
        processed_path = Path(self.config['processed_data_path'])
        processed_path.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet for efficiency
        train_df.to_parquet(processed_path / 'train.parquet', index=False)
        val_df.to_parquet(processed_path / 'val.parquet', index=False)
        test_df.to_parquet(processed_path / 'test.parquet', index=False)
        
        logger.info(f'Processed data saved to {processed_path}')
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''Load previously processed data splits.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        '''
        processed_path = Path(self.config['processed_data_path'])
        
        train_df = pd.read_parquet(processed_path / 'train.parquet')
        val_df = pd.read_parquet(processed_path / 'val.parquet')
        test_df = pd.read_parquet(processed_path / 'test.parquet')
        
        logger.info('Loaded processed data splits')
        return train_df, val_df, test_df

    def save_preprocessed_parquet(self, df: pd.DataFrame, folder_path: Optional[str] = None, filename: str = "preprocessed.parquet") -> None:
        '''Save preprocessed full dataset as parquet to processed data folder.'''
        if not folder_path:
            processed_path = Path(self.config['processed_data_path'])
        else:
            processed_path = Path(folder_path)
        
        processed_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path / filename, index=False)
        logger.info(f'Preprocessed data saved as CSV to {processed_path / filename}')

    def sample_preprocessed_data(self, df: pd.DataFrame, sample_frac: float = 0.1, save: bool = False, filename: str = "sampled_preprocessed.csv") -> pd.DataFrame:
        '''Sample a fraction of the preprocessed data and optionally save to CSV.'''
        if not (0 < sample_frac <= 1):
            raise ValueError('sample_frac must be between 0 and 1')
        original_size = len(df)
        sampled_df = df.groupby('user_id', group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=42))
        logger.info(f'Sampled {len(sampled_df):,} rows ({sample_frac*100:.1f}%) from original {original_size:,} rows (panel-aware)')

        if save:
            processed_path = Path(self.config['processed_data_path'])
            processed_path.mkdir(parents=True, exist_ok=True)
            sampled_df.to_csv(processed_path / filename, index=False)
            logger.info(f'Sampled preprocessed data saved as CSV to {processed_path / filename}')

        return sampled_df