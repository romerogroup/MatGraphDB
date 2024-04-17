import os
import logging

def setup_logging(log_dir):
    """
    Set up logging for the MatGraphDB package.

    Args:
        log_dir (str): The directory where the log file will be saved.

    Returns:
        logger (logging.Logger): The logger object for the MatGraphDB package.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir,'package.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line: %(lineno)d')

    logger = logging.getLogger('matgraphdb')  # define globally (used in train.py, val.py, detect.py, etc.)
    return logger