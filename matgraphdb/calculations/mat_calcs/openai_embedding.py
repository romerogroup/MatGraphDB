import os
import json

import openai
import tiktoken
import pandas as pd
import numpy as np

from matgraphdb.utils import OPENAI_API_KEY, DB_DIR, ENCODING_DIR
from matgraphdb.data.utils import process_database
from matgraphdb.graph.node_types import PROPERTIES


PROPERTY_NAMES = [prop[0] for prop in PROPERTIES]

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculates the number of tokens in a given string using the specified encoding.

    Args:
        string (str): The input string.
        encoding_name (str): The name of the encoding to use.

    Returns:
        int: The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_embedding(text, client, model="text-embedding-3-small"):
    """
    Get the embedding for a given text using OpenAI's text-embedding API.

    Parameters:
    text (str): The input text to be embedded.
    client: The OpenAI client object used to make API requests.
    model (str): The name of the model to use for embedding. Default is "text-embedding-3-small".

    Returns:
    list: The embedding vector for the input text.
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def extract_text_from_json(json_file):
    """
    Extracts specific text data from a JSON file and returns it as a compact JSON string.

    Args:
        json_file (str): The path to the JSON file.

    Returns:
        str: A compact JSON string containing the extracted text data.
    """
    import json

    # Extract text from json file
    with open(json_file, 'r') as f:
        data = json.load(f)

    emd_dict = {}
    for key in data.keys():
        if key in PROPERTY_NAMES:
            emd_dict[key] = data[key]
        elif key == 'structure':
            emd_dict['lattice'] = data[key]['lattice']

    compact_json_text = json.dumps(emd_dict, separators=(',', ':'))
    return compact_json_text

def main():
    """
    Main function for processing database and generating embeddings using OpenAI models.
    
    This function performs the following steps:
    1. Sets up the parameters for the models and cost per token.
    2. Initializes the OpenAI client using the API key.
    3. Processes the database and extracts raw JSON text.
    4. Calculates the total number of tokens and the cost.
    5. Retrieves the mp_ids from the database directory.
    6. Creates a dataframe of the results and adds the ada_embedding column.
    7. Creates a dataframe of the embeddings.
    8. Saves the embeddings to a CSV file.
    """

    ####################
    # Parameters
    ####################
    models=["text-embedding-3-small","text-embedding-3-large","ada v2"]
    cost_per_token=[0.00000002,0.00000013,0.00000010]
    model_index=0
    
    embedding_encoding = "cl100k_base"
    

    ####################
    # Code runs below
    ####################

    MODEL=models[model_index]
    filename=MODEL + ".csv"
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    print("Processing database...")
    # Extracting raw json text from the database
    results=process_database(extract_text_from_json)
    print("Finished processing database")

    # Calculate the total number of tokens and the cost
    token_count=0
    for result in results:
        token_count+=num_tokens_from_string(result,encoding_name=embedding_encoding)

    print("Total number of tokens: ",token_count)
    print("Total cost: ",token_count*cost_per_token[model_index], "$")

    # Get the mp_ids
    mp_ids=[ file.split('.')[0] for file in os.listdir(DB_DIR) if file.endswith('.json')]

    # put reselts into a dataframe under the column 'combined'
    df = pd.DataFrame(results, columns=['combined'], index=mp_ids)


    df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x,client, model=MODEL))

    # Create a dataframe of the embeddings. emb dim span columns size 1535
    df_embeddings = pd.DataFrame(np.array(df['ada_embedding'].tolist()), index=mp_ids)

    # Save the embeddings to a csv file
    df_embeddings.to_csv(os.path.join(ENCODING_DIR,filename), index=True)


if __name__=='__main__':
    main()


    # # Testing openai and tiktoken
    # embedding_encoding = "cl100k_base"
    # client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # prompt="What is Quantum Mechanics?"
    # num_tokens=num_tokens_from_string(string=prompt,encoding_name=embedding_encoding)
    # print(num_tokens)
    # MODEL="text-embedding-3-small"
    # response=get_embedding(prompt,client, model=MODEL)
    # print(response)

    # Testing extract_text_from_json
    # data=extract_text_from_json(os.path.join(DB_DIR,'mp-1000.json'))
    # embedding_encoding = "cl100k_base"
    # num_tokens=num_tokens_from_string(string=data,encoding_name=embedding_encoding)
    # print(data)
    # print("Number of tokens: ",num_tokens)


    

