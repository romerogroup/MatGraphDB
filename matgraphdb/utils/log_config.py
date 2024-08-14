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

def setup_logging(log_dir=LOG_DIR, log_level='debug', format=FORMATTER):
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
    

    logger=logging.basicConfig(level=log_level, 
                        format=FORMAT_STRING,
                        handlers=[handler])

    return logger


def get_logger(name, console_out=False, propagate=False, log_level='info', format=FORMATTER):
    if log_level not in logging_levels:
        raise ValueError(f"Invalid log level: {log_level}. Valid log levels are: {logging_levels}")
    
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
    
    if console_out:
        handler = logging.StreamHandler() 
    else:
        handler = logging.FileHandler(os.path.join(LOG_DIR, f'{name}.log'))
        handler.setFormatter(format)

    handler.setLevel(log_level)

    logger = logging.getLogger(name)
    logger.propagate = propagate  # Prevent propagation to parent logger

    logger.addHandler(handler)

    return logger

def get_child_logger(name, console_out=False, log_level='info', format=FORMATTER):
    if log_level not in logging_levels:
        raise ValueError(f"Invalid log level: {log_level}. Valid log levels are: {logging_levels}")
    
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

    child_logger = logging.getLogger(f'{PARENT_LOGGER_NAME}.{name}')
    child_logger.propagate = False  # Prevent propagation to parent logger

    if console_out:
        child_handler = logging.StreamHandler() 
        child_handler.setLevel(log_level)
        child_logger.addHandler(child_handler)
    else:

        child_handler = logging.FileHandler(os.path.join(LOG_DIR, f'{name}.log'))
        child_handler.setLevel(log_level)
        child_handler.setFormatter(format)

        child_logger.addHandler(child_handler)

    return child_logger
