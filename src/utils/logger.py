'''A function to set up logging for the whole project using config files'''

import logging
import logging.config
from pathlib import Path
from .config_loader import config_loader

def setup_logging() -> logging.Logger:
    '''
    Set up logging configuration

    Return:
        Configured logger instance
    '''
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)

    # Load logging configuration
    try:
        logging_config = config_loader.get_logging_config()
        logging.config.dictConfig(logging_config)
    except Exception as e:
        # Fallback to basic logging if config fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.warning(f'Failed to load logging config: {e}. Using basic config.')
    
    return logging.getLogger('src')

def get_logger(name: str) -> logging.Logger:
    '''
    Get a logger instance.

    Args:
        name: Name for the logger
    
    Returns:
        Logger instance
    '''
    return logging.getLogger(f'{name}')


# Initialize logging
logger = setup_logging() # this gets run when the module is imported