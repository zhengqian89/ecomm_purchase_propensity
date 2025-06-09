'''A utility class to load YAML configuration files.'''
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_dir: str = 'config'):
        '''
        Initialize the config loader

        Args:
        - config_dir: Directory containing configuration files
        '''
        # Resolve path relative to this file
        self.config_dir = Path(__file__).resolve().parent.parent.parent / config_dir
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        '''
        Load a configuration file

        Args:
        - config_name: Name of the config file (without .yaml extension)
        
        Return:
        - Dictionary containing configuration values
        '''
        # If configuration file has been read, retrieve from self._configs
        if config_name in self._configs:
            return self._configs[config_name]
        
        # Create absolute path for the configuration file being loaded
        config_path = self.config_dir / f'{config_name}.yaml'

        # Raise error if config_path does not exists
        if not config_path.exists():
            raise FileNotFoundError(f'Configuration file not found: {config_path}')
        
        # Read yaml file and safe_load it
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Save loaded yaml file to the dictionary
        self._configs[config_name] = config

        return config
    
    def get_data_paths(self) -> Dict[str, str]:
        '''
        Get data paths configuration
        '''
        return self.load_config('data_paths')
    
    def get_logging_config(self) -> Dict[str, Any]:
        '''
        Get logging configuration
        '''
        return self.load_config('logging_config')

# Global config loader instance
config_loader = ConfigLoader()