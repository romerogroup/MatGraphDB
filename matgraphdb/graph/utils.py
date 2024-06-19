

from typing import List, Union


def format_list(prop_list):
    """
    Formats a list into a string for use in Cypher queries.

    Args:
        prop_list (list): A list containing the properties.

    Returns:
        str: A string representation of the properties.
    """
    return [f"{prop}" for prop in prop_list]

def format_string(prop_string):
    """
    Formats a string into a string for use in Cypher queries.

    Args:
        prop_string (str): A string containing the properties.

    Returns:
        str: A string representation of the properties.
    """
    return f"'{prop_string}'"

def format_dictionary(prop_dict):
    """
    Formats a dictionary into a string for use in Cypher queries.

    Args:
        prop_dict (dict): A dictionary containing the properties.

    Returns:
        str: A string representation of the properties.
    """
    formatted_properties="{"
    n_props=len(prop_dict)
    for i,(prop_name,prop_params) in enumerate(prop_dict.items()):
        if isinstance(prop_params,str):
            formatted_properties+=f"{prop_name}: {format_string(prop_params)}"
        elif isinstance(prop_params,int):
            formatted_properties+=f"{prop_name}: {prop_params}"
        elif isinstance(prop_params,float):
            formatted_properties+=f"{prop_name}: {prop_params}"
        elif isinstance(prop_params,List):
            formatted_properties+=f"{prop_name}: {format_list(prop_params)}"
        elif isinstance(prop_params,dict):
            formatted_properties+=f"{prop_name}: {format_dictionary(prop_params)}"

        if i!=n_props-1:
            formatted_properties+=", "

    formatted_properties+="}"
    return formatted_properties

def format_projection(projections:Union[str,List,dict]):
    formatted_projections=""
    if isinstance(projections,List):
        formatted_projections=format_list(projections)
    elif isinstance(projections,dict):
        formatted_projections=format_dictionary(projections)
    elif isinstance(projections,str):       
        formatted_projections=format_string(projections)
    return formatted_projections

