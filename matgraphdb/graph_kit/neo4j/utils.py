
from typing import List, Union

import os
import json

import openai
import tiktoken

from dotenv import load_dotenv
load_dotenv()

from matgraphdb.utils import OPENAI_API_KEY

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





def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Returns the number of tokens in a text string.
    
    Parameters:
        string (str): The input text string.
        encoding_name (str): The name of the encoding to use.
        
    Returns:
        int: The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding(text, client, model="text-embedding-3-small"):
    """
    Get the embedding for a given text using the specified model.

    Args:
        text (str): The input text to be embedded.
        client: The client object used for embedding.
        model (str, optional): The name of the model to use for embedding. Defaults to "text-embedding-3-small".

    Returns:
        list: The embedding vector for the input text.
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding
def get_embedding(text, client, model="text-embedding-3-small"):
   
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_similarity_query(prompt):
    """
    Retrieves the similarity query for a given prompt.

    Args:
        prompt (str): The prompt for which the similarity query is generated.

    Returns:
        tuple: A tuple containing the embedding and the execute statement.

    Example:
        >>> prompt = "What is the melting point of gold?"
        >>> embedding, execute_statement = get_similarity_query(prompt)
    """
    models=["text-embedding-3-small","text-embedding-3-large","ada v2"]
    cost_per_token=[0.00000002,0.00000013,0.00000010]
    model_index=0
    
    MODEL=models[model_index]

    client = openai.OpenAI()

    embedding=get_embedding(prompt,client, model=MODEL)

    execute_statement="""
    CALL db.index.vector.queryNodes('material-text-embedding-3-small-embeddings', $nresults, $embedding)
    YIELD node as sm, score
    RETURN sm, score
    """
    
    return embedding, execute_statement