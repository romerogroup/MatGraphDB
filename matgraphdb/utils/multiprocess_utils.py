import logging
from multiprocessing import Pool
from functools import partial

logger = logging.getLogger(__name__)

def multiprocess_task(func, list, n_cores=1, **kwargs):
    """
    Processes tasks in parallel using a pool of worker processes.

    Args:
        func (Callable): The function to be applied to each item in the list.
        list (list): A list of items to be processed by the function.
        **kwargs: Additional keyword arguments for the function.

    Returns:
        list: The results of applying the function to each item in the input list.
    """
    logger.info(f"Processing tasks in parallel using {n_cores} cores.")
    with Pool(n_cores) as p:
        results=p.map(partial(func,**kwargs), list)
    logger.info("Tasks processed successfully.")
    return results