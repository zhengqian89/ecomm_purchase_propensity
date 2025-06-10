'''
Data ingestion module for the eCommerce Purchase Propensity Engine.
'''
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import config_loader

class ConfigLoader:
    '''Utility class to load configuration files.'''

    def __init__(self, config_dir: str = 'config'):
        '''
        Initialize the config loader.

        Args:
            config_dir: Directory containing configuration files
        '''
        # Always resolve path relative to this file
        self.config_dir = Path(__file__).resolve().parent.parent.parent / config_dir
        self._configs = {}