import os
import json
import random
from glob import glob
from typing import List,Tuple

import torch
import numpy as np
from torch_geometric.data import Data, Dataset


featureList = ["atomic_number"]

def get_polyhedron_graph(poly_dict: str, device:str=None) -> Data:
    data = poly_dict
    # node Features
    n_faces = len(data['face_areas'])
    face_sides = np.array(data['face_sides'])
    face_areas = np.array(data['face_areas'])

    face_normals = np.array(data['face_normals'])

    # edge feature
    dihedral_matrix = np.array(data['dihedral_matrix'])
    edge_length_matrix = np.array(data['edge_length_matrix'])
    face_centers_distance_matrix = np.array(data['face_centers_distance_matrix'])

    # graph energy
    # energy = data['columb_energy']
    energy = data['three_body_energy']
    # energy = data['connected_energy']

    x = np.dstack((face_sides,face_areas))[0]
    
    edge_index = []
    edge_attr = []
    for i_face in range(n_faces):
        for j_face in range(i_face,n_faces):
            if dihedral_matrix[i_face,j_face] > 0:
                edge_index.append([i_face,j_face])
                edge_attr.append([dihedral_matrix[i_face,j_face],face_centers_distance_matrix[i_face,j_face]])
    
    edge_index = np.array(edge_index)
    edge_attr = np.array(edge_attr).T

    # Nodes. Node feature matrix with shape [num_nodes, num_node_features]
    x = torch.tensor(x, dtype=torch.float)
    # Edges. Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long,device=device).t().contiguous()
    # Edge attributes.  Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float,device=device).t().contiguous()
    # Target to train against (may have arbitrary shape)
    y = torch.tensor(energy, dtype=torch.float,device=device)

    pos = face_normals

    data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,pos = pos, y=y)
    data.to(device)
    return data

def get_polyhedra_graphs(file_name: str, device:str=None) -> Data:

    r"""
    this script opens a file of the GDB-9 database and processes it,
    returning the molecule structure in xyz format, a molecule identifier
    (tag), and a vector containing the entire list of molecular properties

    Args:

        file_name (str): filename containing the molecular information

    returns:

        molecule_id (int): integer identifying the molecule number
        in the database n_atoms (int): number of atoms in the molecule
        species (List[str]): the species of each atom (len = n_atoms)
        coordinates (np.array(float)[n_atoms,3]): atomic positions
        properties (np.array(float)[:]): molecular properties, see
        database docummentation charge (np.array(float)[n_atoms]):
        Mulliken charges of atoms

    """

    with open(file_name) as f:
        data = json.load(f)

    graph_a = get_polyhedron_graph(poly_dict=data['polyhedron_a'], device=device)
    graph_b = get_polyhedron_graph(poly_dict=data['polyhedron_b'], device=device) 
    similarity = torch.tensor(data['similarity'], device=device)

    return graph_a, graph_b, similarity

class PolyhedraDataset(Dataset):

    r"""Data set class to load molecular graph data"""

    def __init__(
        self,
        database_dir: str,
        n_max_entries: int = None,
        seed: int = 42,
        transform: object = None,
        pre_transform: object = None,
        pre_filter: object = None,
        device:str = None
    ) -> None:

        r"""

        Args:

           database_dir (str): the directory where the data files reside

           graphs: an object of class MolecularGraphs whose function is
                  to read each file in the data-base and return a
                  graph constructed according to the particular way
                  implemented in the class object (see MolecularGraphs
                  for a description of the class and derived classes)

           n_max_entries (int): optionally used to limit the number of clusters
                  to consider; default is all

           seed (int): initialises the random seed for choosing randomly
                  which data files to consider; the default ensures the
                  same sequence is used for the same number of files in
                  different runs

        """

        super().__init__(database_dir, transform, pre_transform, pre_filter)
        self.device=device
        self.database_dir = database_dir

        filenames = database_dir + "/*.json"

        files = glob(filenames)

        self.n_polyhedra = len(files)

        if n_max_entries and n_max_entries < self.n_polyhedra:

            self.n_polyhedra = n_max_entries
            random.seed(seed)
            self.filenames = random.sample(files, n_max_entries)

        else:

            self.n_polyhedra = len(files)
            self.filenames = files

        

    def len(self) -> int:
        r"""return the number of entries in the database"""

        return self.n_polyhedra

    def get(self, idx: int) -> Data:

        r"""
        This function loads from file the corresponding data for entry
        idx in the database and returns the corresponding graph read
        from the file
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.filenames[idx]

        polyhedra_graph_a,polyhedra_graph_b,similarity = get_polyhedra_graphs(file_name,device=self.device)

        return polyhedra_graph_a, polyhedra_graph_b,similarity

    def get_file_name(self, idx: int) -> str:

        r"""Returns the cluster data file name"""

        return self.filenames[idx]

def main():
    parent_dir = os.path.dirname(__file__)
    train_dir = f"{parent_dir}{os.sep}train"
    test_dir = f"{parent_dir}{os.sep}test"
    polyhedra_dir = f"{parent_dir}{os.sep}polyhedra"


    # filenames = train_dir + "/*.json"
    # files = glob(filenames)
    # get_polyhedra_graph(file_name=files[0])

    dataset = PolyhedraDataset(database_dir=train_dir)

    print(dataset.get_file_name(idx = 0))

    print(dataset.get(idx = 0))



if __name__=='__main__':
    main()