import math
import os 

import torch

import numpy as np
import pandas as pd

class CategoricalEncoder:
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x

class ClassificationEncoder:
    """Converts a column of of unique itentities into a torch tensor. One hot encoding"""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        # Find unique values in the column
        unique_values = df.unique()
        # Create a dictionary mapping unique values to integers
        value_to_index = {value: i for i, value in enumerate(unique_values)}
        tensor=torch.zeros(len(df),len(unique_values))

        for irow,elements in enumerate(df):
            tensor[irow,value_to_index[elements]]=1
        return tensor
    
class BooleanEncoder:
    """Converts a column of boolean values into a torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        # Convert boolean values to integers (True to 1, False to 0)
        boolean_integers = df.astype(int)
        # Create a Torch tensor from the numpy array, ensure it has the correct dtype
        return torch.from_numpy(boolean_integers.values).view(-1, 1).type(self.dtype)
    
class IdentityEncoder:
    """Converts a column of numbers into torch tensor."""
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, df):
        tensor=torch.from_numpy(df.values).view(-1, 1).to(self.dtype)
        return tensor
    
class ListIdentityEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        values=[]
        for irow,row in enumerate(df):
            values.append(row)
        values=np.array(values)

        tensor=torch.from_numpy(values).to(self.dtype)
        return tensor

class IonizationEnergiesEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

        self.column_names=['mean_ionization_energy',
                           'standard_deviation_ionization_energy',
                           'min_ionization_energy',
                           'max_ionization_energy',
                           'median_ionization_energy']

    def __call__(self, df):
        values=[]
        for irow,row in enumerate(df):

            if len(row)==0:
                embedding=[0,0,0,0,0]
                continue

            mean=calculate_mean(row)
            std=calculate_standard_deviation(row)
            min_val=calculate_min(row)
            max_val=calculate_max(row)
            median=calculate_median(row)
            embedding=[mean,std,min_val,max_val,median]
            values.append(embedding)

        values=np.array(values)

        tensor=torch.from_numpy(values).to(self.dtype)
        return tensor
    
class OxidationStatesEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

        self.column_names=['mean_ionization_energy',
                            'standard_deviation_ionization_energy',
                            'min_ionization_energy',
                            'max_ionization_energy',
                            'median_ionization_energy']

    def __call__(self, df):
        values=[]
        for irow,row in enumerate(df):

            if len(row)==0:
                embedding=[0,0,0,0,0]
                values.append(embedding)
                continue

            mean=calculate_mean(row)
            std=calculate_standard_deviation(row)
            min_val=calculate_min(row)
            max_val=calculate_max(row)
            median=calculate_median(row)
            embedding=[mean,std,min_val,max_val,median]
            values.append(embedding)
            
        values=np.array(values)

        tensor=torch.from_numpy(values).to(self.dtype)
        return tensor

class ElementsEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        from matgraphdb.utils.chem_utils.periodic import atomic_symbols
        tensor=torch.zeros(len(df),118)
        element_to_z={element:i-1 for i,element in enumerate(atomic_symbols)}
        for irow,elements in enumerate(df):
            elemnt_indices=[element_to_z[e] for e in elements.split(';')]
            tensor[irow,elemnt_indices]+=1
        return tensor

class CompositionEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        from matgraphdb.utils.chem_utils.periodic import atomic_symbols
        import ast
        tensor=torch.zeros(len(df),118)
        element_to_z={element:i-1 for i,element in enumerate(atomic_symbols)}
        for irow,comp_string in enumerate(df):
            comp_mapping=ast.literal_eval(comp_string)
            for element,comp_val in comp_mapping.items():
                element_index=element_to_z[element]
                tensor[irow,element_index]+=comp_val
        # Normalize tensor by row
        tensor=tensor/tensor.sum(axis=1, keepdims=True)
        return tensor

class SpaceGroupOneHotEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        tensor=torch.zeros(len(df),230)
        for irow,space_group in enumerate(df):
            tensor[irow,space_group-1]+=1
        return tensor
    
class IntegerOneHotEncoder:
    """Converts a column of list of numbers into torch tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        
        possible_values=[]
        for irow,value in enumerate(df):
            possible_values.append(value)
            
        tensor=torch.zeros(len(df),len(possible_values))
        for irow,value in enumerate(df):
            index_value=value-1
            tensor[irow,index_value]+=1
        return tensor


def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def calculate_standard_deviation(numbers):
    mean = calculate_mean(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return math.sqrt(variance)

def calculate_min(numbers):
    return min(numbers)

def calculate_max(numbers):
    return max(numbers)

def calculate_median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    else:
        return sorted_numbers[mid]

# if __name__ == "__main__":
    # import pandas as pd
    # import os
    # import matplotlib.pyplot as plt
    # from matgraphdb.graph.material_graph import MaterialGraph
    # from matgraphdb.mlcore.transforms import min_max_normalize, standardize_tensor

    # material_graph=MaterialGraph()
    # graph_dir = material_graph.graph_dir
    # nodes_dir = material_graph.node_dir
    # relationship_dir = material_graph.relationship_dir
 

    # node_names=material_graph.list_nodes()
    # relationship_names=material_graph.list_relationships()

    # node_files=material_graph.get_node_filepaths()
    # relationship_files=material_graph.get_relationship_filepaths()
