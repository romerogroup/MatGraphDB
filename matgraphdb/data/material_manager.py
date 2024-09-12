
import json
import os
from typing import Union, List, Tuple, Dict


from ase.db import connect
from ase import Atoms
import numpy as np
from ase.calculators.emt import EMT

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure

from matgraphdb.calculations.mat_calcs.chemenv_calc import calculate_chemenv_connections

def ase_to_pymatgen(ase_atoms: Atoms):
        """Convert an ASE Atoms object to a pymatgen Structure object."""
        return Structure.from_sites(ase_atoms.get_chemical_symbols(), ase_atoms.get_scaled_positions(), ase_atoms.get_cell())

def check_all_params_provided(**kwargs):
    param_names = list(kwargs.keys())
    param_values = list(kwargs.values())

    all_provided = all(value is not None for value in param_values)
    none_provided = all(value is None for value in param_values)
    
    if not (all_provided or none_provided):
        missing = [name for name, value in kwargs.items() if value is None]
        provided = [name for name, value in kwargs.items() if value is not None]
        raise ValueError(
            f"If any of {', '.join(param_names)} are provided, all must be provided. "
            f"Missing: {', '.join(missing)}. Provided: {', '.join(provided)}."
        )


class MaterialDatabaseManager:
    """This class is intended to be the Data Access Layer for the Material Database.
    It provides methods for adding, reading, updating, and deleting materials from the database.
    """

    def __init__(self, db_path: str):
        # Connect to the ASE SQLite database
        self.db_path=db_path
        self.db = connect(db_path)
        self.properties=set(self.db.metadata.get('properties',[]))


    def _process_structure(self, structure, coords, coords_are_cartesian, species, lattice):
        if isinstance(structure, Atoms):
            structure = structure
        # Handle structure: if not provided, construct it from lattice, species, and coords
        elif coords is not None:
            if coords_are_cartesian:
                structure = Atoms(symbols=species, positions=coords, cell=lattice, pbc=True)
            else:
                structure = Atoms(symbols=species, scaled_positions=coords, cell=lattice, pbc=True)
        else:
            structure=None
        return structure

    def _process_composition(self, composition):
        if isinstance(composition, Atoms):
            compoosition=composition
        elif isinstance(composition, str):
            composition_str = composition
            compsition=Atoms(composition_str)
        elif isinstance(composition, dict):
            composition_str = ', '.join(f"{k}:{v}" for k, v in composition.items())
            compsition=Atoms(composition_str)
        else:
            composition=None
        return composition
    
    def add_material(self,
                     composition: Union[str, dict, Atoms] = None,
                     structure: Atoms = None,
                     coords: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                     coords_are_cartesian: bool = False,
                     species: List[str] = None,
                     lattice: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                     properties: Dict = None,
                     db=None):
        """
        Add a material to the database.

        Args:
            composition (Union[str, dict], optional): The composition of the material.
            structure (Atoms, optional): The atomic structure in ASE format.
            coords (Union[List[Tuple[float,float,float]], np.ndarray], optional): Atomic coordinates.
            coords_are_cartesian (bool, optional): If the coordinates are cartesian.
            species (List[str], optional): Atomic species.
            lattice (Union[List[Tuple[float,float,float]], np.ndarray], optional): Lattice parameters.
            properties (dict, optional): Additional properties of the material.
        """
        
        # Handle composition: convert string or dict to a format for storage
        composition = self._process_composition(composition)

        check_all_params_provided(coords=coords, species=species, lattice=lattice)

        ase_structure = self._process_structure(structure, coords, coords_are_cartesian, species, lattice)

        if ase_structure is None and composition is None:
            raise ValueError("Either a structure or a composition must be provided")

        entry_data={}
        # Add any custom properties from the properties argument
        if properties:
            entry_data.update(properties)

        # Write to the ASE database
        if db is None:
            with connect(self.db_path) as db:
                db.write(structure, data=entry_data)
        else:
            db.write(structure, data=entry_data)

        return None
    
    def add_many(self, materials: List[Dict]):
        """
        Write many materials to the database in a single transaction.
        
        Args:
            materials (List[Dict]): A list of dictionaries where each dictionary represents a material
                                    with keys like 'composition', 'structure', 'coords', etc.
        """
        with connect(self.db_path) as db:
            for material in materials:
                self.add_material(db=db,**material)

    def read(self, selection=None , **kwargs):
        """Read a material from the database by ID."""
        return self.db.select(selection=selection, **kwargs)

    def read_properties(self, selection=None , **kwargs):
        """Read properties in the database by ID."""

        rows=self.db.select(selection=selection, **kwargs)
        properties=[row.data for row in rows]
        return properties

    def update_material(self, id: int,
                        structure: Atoms = None,
                        properties: Dict = None,
                        delete_keys: List[str] = None,
                        db=None,
                        **kwargs):
        
        entry_data={}
        # Add any custom properties from the properties argument
        if properties:
            entry_data.update(properties)

        # Write to the ASE database
        if db is None:
            with connect(self.db_path) as db:
                db.update(id, atoms=structure, data=entry_data, delete_keys=delete_keys)
        else:
            db.update(id, atoms=structure, data=entry_data, delete_keys=delete_keys)

    def update_many(self, update_list: List[Dict]):
        """
        Update many rows in the database in a single transaction.
        
        Args:
            updates (List[Tuple(id,Dict)]): A list of of tuples where the first element is the material ID 
            and the second is a dictionary of properties to update.
        """
        with connect(self.db_path) as db:
            for update in update_list:
                id=update[0]
                material_dict=update[1]
                self.update_material(id=id, properties=material_dict,  db=db)

    def delete_properties(self, delete_keys: List[str]):
        """
        Delete properties from a material in the database.
        
        Args:

            delete_keys (List[str]): A list of property keys to delete.
        """
        rows=self.list_materials()
        with connect(self.db_path) as db:
            for row in rows:
                self.update_material(id=row.id, delete_keys=delete_keys, db=db)

    def delete_material(self, material_id: int):
        """Delete a material from the database by ID."""
        with connect(self.db_path) as db:
            db.delete(material_id)

    def add_metadata(self, metadata: Dict):
        """Add metadata to the database."""
        with connect(self.db_path) as db:
            db_metadata=db.metadata
            db_metadata.update(metadata)
            db.metadata=db_metadata

    def delete_metadata(self, delete_keys: List[str]):
        """
        Delete metadata from the database.
        
        Args:

            delete_keys (List[str]): A list of metadata keys to delete.
        """
        with connect(self.db_path) as db:
            db_metadata=db.metadata
            for key in delete_keys:
                db_metadata.pop(key)
                db.metadata=db_metadata

    def update_data_properties(self):
        """Update the data properties of the database."""
        # with connect(self.db_path) as db:
        rows=self.read()
        for row in rows:
            data_properties=list(row.data.keys())
            self.properties.update(data_properties)

        with connect(self.db_path) as db:
            db_metadata=db.metadata
            db_metadata['properties']=list(self.properties)
            db.metadata=db_metadata

    def get_properties(self):
        """Get the properties of the database."""
        return self.properties


# class MaterialDatabaseManager:

#     def __init__(self, path: str):
#         # Connect to the ASE SQLite database
#         self.path=path
#         os.makedirs(self.path,exist_ok=True)
#         self.db_manager=MaterialDatabaseManager(db_path=os.path.join(self.path,'matgraphdb.db'))


#     def _calculate_space_group(structure,symprec=0.01):
#         sym_analyzer=SpacegroupAnalyzer(structure,symprec=symprec)

#         data={'symmetry':{}}
#         data["symmetry"]["crystal_system"]=sym_analyzer.get_crystal_system()
#         data["symmetry"]["number"]=sym_analyzer.get_space_group_number()
#         data["symmetry"]["point_group"]=sym_analyzer.get_point_group_symbol()
#         data["symmetry"]["symbol"]=sym_analyzer.get_hall()
#         data["symmetry"]["symprec"]=symprec

#         sym_dataset=sym_analyzer.get_symmetry_dataset()
#         data["wyckoffs"]=sym_dataset['wyckoffs']
#         data["symmetry"]["version"]="1.16.2"

#         return data

#     def _calculate_chemenv(structure):
#         data={'coordination_environments_multi_weight':None, 
#               'coordination_multi_connections':None, 
#               'coordination_multi_numbers':None}
#         try:
#             coordination_environments, nearest_neighbors, coordination_numbers = calculate_chemenv_connections(structure)
#             data["coordination_environments_multi_weight"]=coordination_environments
#             data["coordination_multi_connections"]=nearest_neighbors
#             data["coordination_multi_numbers"]=coordination_numbers
#         except:
#             pass

#         return data


if __name__ == "__main__":
    import time
    # Create a new database
    db_path = "data/raw/test.db"
    manager = MaterialDatabaseManager(db_path)

    # Add a material to the database
    composition = {"Fe": 1, "O": 2}

    # coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    # species=["Fe", "O"]
    # lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # f=manager.add_material(coords=coords, species=species, lattice=lattice, properties={"density": 1.0})
    
    # material_dict={
    #                "coords": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]), 
    #                "species": ["Fe", "O"], 
    #                "lattice": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
    #                "properties": {"density": 1.0}}

    # rows=manager.read()
    # print(type(rows))

    # rows=manager.read_properties()
    # rows=manager.add_metadata({'test2':1})


    # for row in rows:
    #     print(row)
        # print(row.id)
    # row = manager.read_material(material_id=1)
    # print(row.data)
    # for row in rows:
    #     print(row.id)
    # rows=manager.list_materials()

    # for row in rows:
    #     print(row.id)
    #     print(row.data)
















    # with open('sandbox/json_file_1000.json','r') as f:
    #     data=json.load(f)
    # material_dict['properties']['other']=data

    # materials= [material_dict for _ in range(10000)]
    # # print(f)\
    # begin_time = time.time()
    # manager.write_many(materials=materials)
    # print('Write time: ', time.time() - begin_time)

    # begin_time = time.time()
    # rows=manager.list_materials()
    # print(len(rows))
    # print('List time: ', time.time() - begin_time)

    # print(rows[0].data)
    # for row in rows:
    #     print(type(row.data))
    # print(manager.list_materials())