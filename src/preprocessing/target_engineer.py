import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TargetEngineer:
    '''
    Create target variables for purchase propensity prediction.
    '''
    def __init__(self):
        logger.info('TargetEngineer instantiated...')
    