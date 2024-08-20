from typing import List, Union


def is_in_range(val:Union[float, int],min_val:Union[float, int],max_val:Union[float, int], negation:bool=True):
    """
    Screens a list of floats to keep only those that are within a given range.

    Args:
        floats (Union[float, int]): A list of floats to be screened.
        min_val (float): The minimum value to keep.
        max_val (float): The maximum value to keep.
        negation (bool, optional): If True, returns True if the value is within the range. 
                                If False, returns True if the value is outside the range.
                                Defaults to True.

    Returns:
        bool: A boolean indicating whether the value is within the given range.
    """
    if negation:
        return min_val <= val <= max_val
    else:
        return not (min_val <= val <= max_val)

def is_in_list(val, string_list: List, negation: bool = True) -> bool:
    """
    Checks if a value is (or is not, based on the inverse_check flag) in a given list.

    Args:
        val: The value to be checked.
        string_list (List): The list to check against.
        negation (bool, optional): If True, returns True if the value is in the list.
                                        If False, returns True if the value is not in the list.
                                        Defaults to True.

    Returns:
        bool: A boolean indicating whether the value is (or is not) in the list based on 'inverse_check'.
    """
    return (val in string_list) if negation else (val not in string_list)