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
