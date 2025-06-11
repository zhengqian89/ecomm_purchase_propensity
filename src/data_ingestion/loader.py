'''
Data ingestion module for the eCommerce Purchase Propensity Engine.
'''
from src.utils.config_loader import config_loader

class ConfigLoader:
    '''Utility class to load configuration files.'''

    def __init__(self, config_dir: str = 'config'):
        '''
        Initialize data loader.
        '''
        self.config = config_loader.get_data_paths()
        self.model_config = config_loader.get_model_params()

    def load_raw_data()