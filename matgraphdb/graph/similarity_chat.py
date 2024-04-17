import os
import json

import openai
import tiktoken

from matgraphdb.utils import OPENAI_API_KEY

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

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    embedding=get_embedding(prompt,client, model=MODEL)

    execute_statement="""
    CALL db.index.vector.queryNodes('material-text-embedding-3-small-embeddings', $nresults, $embedding)
    YIELD node as sm, score
    RETURN sm, score
    """
    
    return embedding, execute_statement

# if __name__ == "__main__":
#     # prompt = "What are materials similar to the composition TiAu"
#     # prompt = "What are some materials with a large band gap?"
#     prompt = "band_gap greater than 1.0?"
#     # prompt = "What are some materials with a with hexagonal crystal system?"
#     similarity_query(prompt,n_results=20)