import os
import json

import openai
import tiktoken
import pandas as pd
import numpy as np
from neo4j import GraphDatabase

from matgraphdb.utils import OPENAI_API_KEY, DB_DIR, ENCODING_DIR
from matgraphdb.database.utils import process_database
from matgraphdb.database.json.utils import PROPERTY_NAMES
from matgraphdb.utils import PASSWORD,USER,LOCATION,DB_NAME

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_embedding(text, client, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def similarity_query(prompt, n_results=5):

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

    embedding=get_embedding(prompt,client, model=MODEL)


    execute_statement="""
    CALL db.index.vector.queryNodes('material-text-embedding-3-small-embeddings', $nresults, $embedding)
    YIELD node as sm, score
    RETURN sm.composition_reduced,sm.formula_pretty,sm.symmetry, score
    """

    connection = GraphDatabase.driver(LOCATION, auth=(USER, PASSWORD))
    session = connection.session(database=DB_NAME)


    results = session.run(execute_statement, {"embedding": embedding,"nresults":n_results})

    for record in results:
        composition_reduced = record["sm.composition_reduced"]
        formula_pretty = record["sm.formula_pretty"]
        symmetry = record["sm.symmetry"]
        score = record["score"]
        
        print(f"Composition Reduced: {composition_reduced}")
        print(f"Formula Pretty: {formula_pretty}")
        print(f"Symmetry: {symmetry}")
        print(f"Score: {score}")
        print('_'*200)
    session.close()
    connection.close()

if __name__ == "__main__":
    prompt = "What are materials similar to the composition TiAu"
    similarity_query(prompt,n_results=20)