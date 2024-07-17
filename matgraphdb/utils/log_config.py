import os
from datetime import datetime
import logging

class InfoFilter(logging.Filter):
    def filter(self, record):
        # Only allow INFO level messages
        return record.levelno == logging.INFO

def setup_logging(log_dir,apply_filter=True):
    """
    Set up logging for the MatGraphDB package.

    Args:
        log_dir (str): The directory where the log file will be saved.

    Returns:
        logger (logging.Logger): The logger object for the MatGraphDB package.
    """

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"package_{timestamp}.log"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir,log_filename), 
                        level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line: %(lineno)d')

    logger = logging.getLogger('matgraphdb')  # define globally (used in train.py, val.py, detect.py, etc.)
    if apply_filter:
        logger.addFilter(InfoFilter())  # Apply the custom filter to only log INFO messages
    return logger