import os
from datetime import datetime
import logging

from matgraphdb.utils.config import LOG_DIR

class InfoFilter(logging.Filter):
    def filter(self, record):
        # Only allow INFO level messages
        return record.levelno == logging.INFO

PARENT_LOGGER_NAME = 'matgraphdb'

FORMAT_STRING = '%(name)s :: %(levelname)-8s :: %(message)s'
FORMATTER = logging.Formatter(fmt=FORMAT_STRING)

logging_levels=['critical','error','warning','info','debug']

def setup_logging(log_dir=LOG_DIR, log_level='info', console_out=True, format=FORMATTER):
    """
    Set up logging for the MatGraphDB package.

    Args:
        log_dir (str): The directory where the log file will be saved.

    Returns:
        logger (logging.Logger): The logger object for the MatGraphDB package.
    """
    if log_level == 'debug':
        log_level = logging.DEBUG
    elif log_level == 'info':
        log_level = logging.INFO
    elif log_level == 'warning':
        log_level = logging.WARNING
    elif log_level == 'error':
        log_level = logging.ERROR
    elif log_level == 'critical':
        log_level = logging.CRITICAL

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"package_{timestamp}.log"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handler = logging.FileHandler(os.path.join(LOG_DIR, f'{PARENT_LOGGER_NAME}.log'))
    handler.setFormatter(format)
    handler.setLevel(log_level)
    
    if console_out:
        handler = logging.StreamHandler() 
    else:
        handler = logging.FileHandler(os.path.join(LOG_DIR, f'{PARENT_LOGGER_NAME}.log'))
        handler.setFormatter(format)

    logger=logging.basicConfig(level=log_level, 
                        format=FORMAT_STRING,
                        handlers=[handler])

    return logger
