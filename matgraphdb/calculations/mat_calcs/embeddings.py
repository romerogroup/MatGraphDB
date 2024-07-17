import os
import json


import pandas as pd
import numpy as np
from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure import XRDPowderPattern,SineCoulombMatrix,CoulombMatrix
from matminer.featurizers.composition import ElementFraction, ElementProperty

from pymatgen.core import Structure




# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     """
#     Calculates the number of tokens in a given string using the specified encoding.

#     Args:
#         string (str): The input string.
#         encoding_name (str): The name of the encoding to use.

#     Returns:
#         int: The number of tokens in the string.
#     """
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens


# def get_embedding(text, client, model="text-embedding-3-small"):
#     """
#     Get the embedding for a given text using OpenAI's text-embedding API.

#     Parameters:
#     text (str): The input text to be embedded.
#     client: The OpenAI client object used to make API requests.
#     model (str): The name of the model to use for embedding. Default is "text-embedding-3-small".

#     Returns:
#     list: The embedding vector for the input text.
#     """
#     text = text.replace("\n", " ")
#     return client.embeddings.create(input=[text], model=model).data[0].embedding

# def extract_text_from_json(json_file):
#     """
#     Extracts specific text data from a JSON file and returns it as a compact JSON string.

#     Args:
#         json_file (str): The path to the JSON file.

#     Returns:
#         str: A compact JSON string containing the extracted text data.
#     """
#     import json
#     from matgraphdb.graph.node_types import PROPERTIES
#     PROPERTY_NAMES = [prop[0] for prop in PROPERTIES]
#     # Extract text from json file
#     with open(json_file, 'r') as f:
#         data = json.load(f)

#     emd_dict = {}
#     for key in data.keys():
#         if key in PROPERTY_NAMES:
#             emd_dict[key] = data[key]
#         elif key == 'structure':
#             emd_dict['lattice'] = data[key]['lattice']

#     compact_json_text = json.dumps(emd_dict, separators=(',', ':'))
#     return compact_json_text

# def generate_openai_embeddings(
#                             materials_text, 
#                             material_ids,
#                             model="text-embedding-3-small",
#                             embedding_encoding = "cl100k_base"
#                             ):
#     """
#     Main function for processing database and generating embeddings using OpenAI models.

#     This function performs the following steps:
#     1. Sets up the parameters for the models and cost per token.
#     2. Initializes the OpenAI client using the API key.
#     3. Processes the database and extracts raw JSON text.
#     4. Calculates the total number of tokens and the cost.
#     5. Retrieves the mp_ids from the database directory.
#     6. Creates a dataframe of the results and adds the ada_embedding column.
#     7. Creates a dataframe of the embeddings.
#     8. Saves the embeddings to a CSV file.

#     Args:
#         materials_text (list): A list of materials represented as text.
#         material_ids (list): A list of material IDs.
#         model (str): The name of the OpenAI model to use for embedding. Default is "text-embedding-3-small". 
#                         Possible values are "text-embedding-3-small", "text-embedding-3-large", and "ada v2".
#         embedding_encoding (str): The name of the encoding to use for embedding. Default is "cl100k_base".

#     Returns:
#         None
#     """

#     models_cost_per_token={
#         "text-embedding-3-small":0.00000002,
#         "text-embedding-3-large":0.00000013,
#         "ada v2":0.00000010
#     }

#     cost_per_token=models_cost_per_token[model]


#     client = openai.OpenAI(api_key=OPENAI_API_KEY)

#     # Calculate the total number of tokens and the cost
#     token_count=0
#     for material_text in materials_text:
#         token_count+=num_tokens_from_string(material_text,encoding_name=embedding_encoding)
#     LOGGER.info(f"Total number of tokens: {token_count}")
#     LOGGER.info(f"Cost per token: {cost_per_token}")

#     # put reselts into a dataframe under the column 'combined'
#     df = pd.DataFrame(materials_text, columns=['combined'], index=material_ids)
#     df['embedding'] = df.combined.apply(lambda x: get_embedding(x,client, model=model))

#     # Create a dataframe of the embeddings. emb dim span columns size 1535
#     tmp_dict={
#         "ada_embedding":df['embedding'].tolist(),
#     }
#     df_embeddings = pd.DataFrame(tmp_dict, index=material_ids)
#     # df_embeddings = pd.DataFrame(np.array(df['ada_embedding'].tolist()), index=material_ids)

#     return df_embeddings
 
    

def generate_composition_embeddings(compositions,material_ids):
    """Generate composition embeddings using Matminer and OpenAI.

    Args:
        compositions (list): A list of compositions.
        material_ids (list): A list of material IDs.

    Returns:
        None
    """
    composition_data = pd.DataFrame({'composition': compositions}, index=material_ids)
    composition_featurizer = MultipleFeaturizer([ElementFraction()])
    composition_features = composition_featurizer.featurize_dataframe(composition_data,"composition")
    composition_features=composition_features.drop(columns=['composition'])
    features=composition_features
    return features



def generate_matminer_embeddings(structures,material_ids,features=[]):
    """Generate structures embeddings using Matminer and OpenAI.

    Args:
        structures (list): A list of structures.
        material_ids (list): A list of material IDs.

    Returns:
        None
    """
    allowed_features=['sine_coulomb_matrix','element_fraction','element_property','xrd_pattern']
    for feature in features:
        if feature not in allowed_features:
            raise ValueError(f"Feature {feature} not allowed")
        
    compositions=[structure.composition for structure in structures]
    data = pd.DataFrame({'structure': structures, 'composition':compositions}, index=material_ids)

    composition_featurizers=[]
    if 'element_fraction' in features:
        composition_featurizers.append(ElementFraction())
    if 'element_property' in features:
        composition_featurizers.append(ElementProperty.from_preset(preset_name="magpie"))

    if len(composition_featurizers)>0:
        composition_featurizer = MultipleFeaturizer(composition_featurizers)
        data = composition_featurizer.featurize_dataframe(data,"composition",pbar=False)

    structure_featurizers=[]
    if 'xrd_pattern' in features:
        structure_featurizers.append(XRDPowderPattern())
    if 'sine_coulomb_matrix' in features:
        columb_matrix=SineCoulombMatrix()
        columb_matrix.fit(structures)
        structure_featurizers.append(columb_matrix)

    if len(structure_featurizers)>0:
        structure_featurizer=MultipleFeaturizer(structure_featurizers)
        data = structure_featurizer.featurize_dataframe(data,"structure",pbar=False)

    data=data.drop(columns=['structure','composition'])
    return data


if __name__=='__main__':
    from matgraphdb.data.manager import DBManager
    from pymatgen.core import Structure
    import itertools


    data=DBManager().load_json('mp-1000.json')
    features=['sine_coulomb_matrix','element_fraction','element_property','xrd_pattern']

    feature_sets=[
        ['sine_coulomb_matrix'],
        ['xrd_pattern'],
        ['element_fraction'],
        ['element_property'],
        ['sine_coulomb_matrix','element_property'],
        ['sine_coulomb_matrix','element_fraction'],
        ['element_property','element_fraction'],
        ['sine_coulomb_matrix','element_property','element_fraction'],
    ]
    structures = [Structure.from_dict(data['structure'])]
    print(structures)
    materials_ids=['mp-1000']
    # features=generate_matminer_embeddings(structures,materials_ids,features=['sine_coulomb_matrix','xrd_pattern'])
    # print(features)

    for feature_set in feature_sets:
        print(feature_set)
        features=generate_matminer_embeddings(structures,materials_ids,features=feature_set)
        print(features)



