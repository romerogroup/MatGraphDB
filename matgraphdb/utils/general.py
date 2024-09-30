import inspect
import os



def chunk_list(input_list, chunk_size):
    """
    Splits a list into smaller chunks of a specified size.

    Args:
        input_list (list): The list to be chunked.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of smaller chunks.

    Example:
        >>> chunk_list([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]



def print_directory_tree(directory, skip_dirs=None):
    """
    Prints the directory tree structure starting from the given directory.

    Args:
        directory (str): The root directory from which to start printing the tree.
        skip_dirs (list, optional): A list of directories to skip during printing. Defaults to None.

    Returns:
        None
    """

    if skip_dirs is None:
        skip_dirs = []

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs]  # Filter out skipped directories
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        for file in files:
            print(f'{indent}{file}')

def get_os():
    if os.name == 'nt':
        return "Windows"
    elif os.name == 'posix':
        return "Linux or macOS"
    else:
        return "Unknown OS"
    

def get_function_args(func):
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
    
    return args, kwargs


if __name__ == '__main__':
    # Example usage

    skip_dirs = ['__pycache__']
    print_directory_tree('Z:\Research Projects\crystal_generation_project\MatGraphDB\matgraphdb',skip_dirs=skip_dirs)
    print_directory_tree('Z:\Research Projects\crystal_generation_project\MatGraphDB\matgraphdb',skip_dirs=skip_dirs)