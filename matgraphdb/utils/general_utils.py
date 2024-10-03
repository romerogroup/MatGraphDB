import logging

import inspect
import os
from typing import Callable, List

logger = logging.getLogger(__name__)


def chunk_list(input_list: List, chunk_size: int):
    """
    Splits a list into smaller chunks of a specified size.

    Parameters
    ----------
    input_list : list
        The list to be chunked.
    chunk_size : int
        The size of each chunk.

    Returns
    -------
    list
        A list of smaller lists, each representing a chunk of the original list.

    Example
    -------
    >>> chunk_list([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    logger.info(f"Splitting list into chunks of size {chunk_size}")
    out=[input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    logger.info(f"Chunks created: {len(out)}")
    return out

def get_function_args(func: Callable):
    """
    Retrieves the positional and keyword arguments of a function.

    Parameters
    ----------
    func : Callable
        The function to inspect for its argument signature.

    Returns
    -------
    tuple
        A tuple containing two lists: the first list contains positional arguments, 
        and the second list contains keyword arguments.

    Example
    -------
    >>> def example_func(a, b, c=3, d=4):
    ...     pass
    >>> get_function_args(example_func)
    (['a', 'b'], ['c', 'd'])
    """
    logger.info(f"Inspecting function: {func.__name__}")
    signature = inspect.signature(func)
    params = signature.parameters
    
    args = []
    kwargs = []
    for name, param in params.items():
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if param.default == inspect.Parameter.empty:
                args.append(name)
            else:
                kwargs.append(name)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs.append(name)
    
    logger.info(f"{func.__name__}", extra={"args": args, "kwargs": kwargs})
    return args, kwargs


# if __name__ == '__main__':
#     # Example usage

#     skip_dirs = ['__pycache__']
#     print_directory_tree('Z:\Research Projects\crystal_generation_project\MatGraphDB\matgraphdb',skip_dirs=skip_dirs)
#     print_directory_tree('Z:\Research Projects\crystal_generation_project\MatGraphDB\matgraphdb',skip_dirs=skip_dirs)