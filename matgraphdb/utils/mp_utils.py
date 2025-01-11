import logging
import os
from functools import partial
from multiprocessing import Pool

from dask.distributed import Client

from matgraphdb.utils.config import config

logger = logging.getLogger(__name__)


def get_cpus_per_node():
    """
    Retrieves the number of CPUs per node from the SLURM environment.

    This function checks the SLURM environment variable 'SLURM_JOB_CPUS_PER_NODE'
    to determine the CPU configuration per node. It handles different formats
    provided by SLURM, such as a single number, comma-separated values, or formats
    like '32(x2)' indicating multiple nodes with the same number of CPUs.
    If no information is available, it defaults to 1 CPU.

    Returns
    -------
    list
        A list of integers representing the number of CPUs for each node.
    """
    logger.info("Fetching CPUs per node from SLURM.")
    cpu_per_node = os.getenv("SLURM_JOB_CPUS_PER_NODE")

    if cpu_per_node is None:
        logger.debug("SLURM_JOB_CPUS_PER_NODE is not defined. Assuming 1 CPU per node.")
        cpus_node_list = 1
    elif "(x" in cpu_per_node:
        logger.debug(
            "SLURM_JOB_CPUS_PER_NODE contains multiple nodes with variable CPUs per node."
        )
        cpu_per_node, num_nodes = cpu_per_node.strip(")").split("(x")
        cpus_node_list = [int(cpu_per_node) for _ in range(int(num_nodes))]
    else:
        logger.debug(
            "SLURM_JOB_CPUS_PER_NODE contains a multiples node with fixed CPUs per node."
        )
        cpus_node_list = [int(x) for x in cpu_per_node.split(",")]

    logger.info(f"SLURM_JOB_CPUS_PER_NODE: {cpu_per_node}")
    return cpus_node_list


def get_num_tasks():
    """
    Retrieves the number of tasks (SLURM_NTASKS) from the SLURM environment.

    This function checks the SLURM environment variable 'SLURM_NTASKS' to determine
    the number of tasks allocated for the SLURM job. It logs the number of tasks and
    returns it as an integer.

    Returns
    -------
    int or None
        The number of tasks allocated for the SLURM job, or None if not defined.
    """
    logger.info("Fetching number of tasks from SLURM.")
    num_tasks = os.getenv("SLURM_NTASKS")

    if num_tasks:
        logger.debug("SLURM_NTASKS is defined. Using it.")
        num_tasks = int(num_tasks)

    logger.info(f"SLURM_NTASKS: {num_tasks}")
    return num_tasks


def get_num_nodes():
    """
    Retrieves the number of nodes allocated for the SLURM job (SLURM_JOB_NUM_NODES).

    This function checks the SLURM environment variable 'SLURM_JOB_NUM_NODES' to
    get the number of nodes allocated to the current job and logs the value.

    Returns
    -------
    int
        The number of nodes allocated for the SLURM job.
    """
    num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))
    logger.info(f"SLURM_JOB_NUM_NODES: {num_nodes}")
    return num_nodes


def get_total_cores(cpus_per_node):
    """
    Calculates the total number of CPU cores based on the CPUs per node.

    This function takes a list of CPUs per node and returns the total number
    of cores by summing the values.

    Parameters
    ----------
    cpus_per_node : list of int
        A list of integers where each integer represents the number of CPUs per node.

    Returns
    -------
    int
        The total number of CPU cores across all nodes.
    """
    logger.info("Calculating total number of cores.")
    return sum(cpus_per_node)


def get_num_cores(n_cores: int = None):
    """
    Determines the total number of cores to be used for a SLURM job.

    This function checks if a specific number of cores is manually provided (`n_cores`).
    If not provided, it fetches the CPUs per node from SLURM and calculates the total
    number of cores. If the CPUs per node are represented as a list, it calculates
    the total cores across all nodes. If it is not a list, it assumes a single node
    setup and returns the number of CPUs.

    Parameters
    ----------
    n_cores : int, optional
        Manually specified number of cores. If provided, this value will be returned.
        Defaults to None.

    Returns
    -------
    int
        The total number of CPU cores to use.
    """
    logger.info("Determining number of cores to use.")
    cpus_per_node = get_cpus_per_node()

    total_cores = None
    if n_cores:
        logger.debug(f"Detected manually specified cores: {n_cores}")
        total_cores = n_cores
    elif isinstance(cpus_per_node, list):
        logger.debug(f"Detected multiple nodes: {cpus_per_node}")
        total_cores = get_total_cores(cpus_per_node)
    else:
        logger.debug(f"Detected other core format: {cpus_per_node}")
        total_cores = cpus_per_node

    logger.info(f"Total cores: {total_cores}")
    return total_cores


def multiprocess_task(func, list, n_cores=None, **kwargs):
    """
    Processes tasks in parallel using a pool of worker processes.

    This function applies a given function to a list of items in parallel, using
    multiprocessing with a specified number of cores. Each item in the list is processed
    by the function, and additional arguments can be passed through `kwargs`.
    By defualt, it will detect the number of cores available unless specified otherwise.

    Parameters
    ----------
    func : Callable
        The function to be applied to each item in the list.
    list : list
        A list of items to be processed by the function.
    n_cores : int, optional
        The number of cores to use for multiprocessing (default is None).
    **kwargs
        Additional keyword arguments to be passed to `func`.

    Returns
    -------
    list
        A list of results obtained by applying `func` to each item in the input list.
    """
    # n_cores = get_num_cores(n_cores)
    # logger.info(f"Processing tasks in parallel using {n_cores} cores.")
    logger.info(f"Passing the following arguments to the worker method", extra=kwargs)
    try:
        with Pool() as p:
            results = p.map(partial(func, **kwargs), list)
        return results
    except:
        logging.exception("Error processing tasks in parallel.")


def parallel_apply(func, data, processes=False):
    if len(data) > 2000 and config.use_multiprocessing:
        with Client(
            silence_logs=logging.ERROR,
            processes=processes,
        ) as client:
            serialized_futures = client.map(func, data)
            results = client.gather(serialized_futures)
    else:
        results = [func(item) for item in data]
    return results
