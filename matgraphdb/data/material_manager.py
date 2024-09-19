import json
import os
import logging
from typing import Union, List, Tuple, Dict


from ase.db import connect
from ase import Atoms
import numpy as np
from ase.calculators.emt import EMT

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure

from matgraphdb.calculations.mat_calcs.chemenv_calc import calculate_chemenv_connections

logger = logging.getLogger(__name__)

def ase_to_pymatgen(ase_atoms: Atoms):
    """Convert an ASE Atoms object to a pymatgen Structure object."""
    logger.debug("Converting ASE Atoms object to pymatgen Structure.")
    return Structure.from_sites(ase_atoms.get_chemical_symbols(), ase_atoms.get_scaled_positions(), ase_atoms.get_cell())

def check_all_params_provided(**kwargs):
    param_names = list(kwargs.keys())
    param_values = list(kwargs.values())

    all_provided = all(value is not None for value in param_values)
    none_provided = all(value is None for value in param_values)
    
    if not (all_provided or none_provided):
        missing = [name for name, value in kwargs.items() if value is None]
        provided = [name for name, value in kwargs.items() if value is not None]
        logger.error(
            f"If any of {', '.join(param_names)} are provided, all must be provided. "
            f"Missing: {', '.join(missing)}. Provided: {', '.join(provided)}."
        )
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

        logger.info(f"Initializing MaterialDatabaseManager with database at {db_path}")
        self.db = connect(db_path)
        self.data_keys=set(self.db.metadata.get('data_keys',[]))


    def _process_structure(self, structure, coords, coords_are_cartesian, species, lattice):
        logger.debug("Processing structure input.")
        if structure is not None:
            if not isinstance(structure, Atoms):
                logger.error("Structure must be an ASE Atoms object.")
                raise TypeError("Structure must be an ASE Atoms object")
            logger.debug("Using provided ASE Atoms structure.")
            return structure
        elif coords is not None and species is not None and lattice is not None:
            logger.debug("Building ASE Atoms structure from provided coordinates, species, and lattice.")
            if coords_are_cartesian:
                return Atoms(symbols=species, positions=coords, cell=lattice, pbc=True)
            else:
                return Atoms(symbols=species, scaled_positions=coords, cell=lattice, pbc=True)
        else:
            logger.debug("No valid structure information provided.")
            return None

    def _process_composition(self, composition):
        logger.debug("Processing composition input.")
        if isinstance(composition, Atoms):
            logger.debug("Composition provided as ASE Atoms object.")
            return composition
        elif isinstance(composition, str):
            composition_str = composition
            logger.debug(f"Composition provided as string: {composition_str}")
            return Atoms(composition_str)
        elif isinstance(composition, dict):
            composition_str = ', '.join(f"{k}:{v}" for k, v in composition.items())
            logger.debug(f"Composition provided as dict: {composition_str}")
            return Atoms(composition_str)
        else:
            logger.debug("No valid composition information provided.")
            return None
    
    def add_material(self,
                     coords: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                     coords_are_cartesian: bool = False,
                     species: List[str] = None,
                     lattice: Union[List[Tuple[float, float, float]], np.ndarray] = None,
                     structure: Atoms = None,
                     composition: Union[str, dict, Atoms] = None,
                     data: Dict = None,
                     db=None,
                     **kwargs):
        """
        Add a material to the database.

        Args:
            
            coords (Union[List[Tuple[float,float,float]], np.ndarray], optional): Atomic coordinates.
            coords_are_cartesian (bool, optional): If the coordinates are cartesian.
            species (List[str], optional): Atomic species.
            lattice (Union[List[Tuple[float,float,float]], np.ndarray], optional): Lattice parameters.
            structure (Atoms, optional): The atomic structure in ASE format.
            composition (Union[str, dict], optional): The composition of the material.
            data (dict, optional): Additional data of the material.
            **kwargs: Additional keyword arguments to pass to the ASE database.
        """

        logger.info("Adding a new material.")
        # Handle composition: convert string or dict to a format for storage
        ase_composition = self._process_composition(composition)

        check_all_params_provided(coords=coords, species=species, lattice=lattice)

        ase_structure = self._process_structure(structure, coords, coords_are_cartesian, species, lattice)

        if ase_structure is None and ase_composition is None:
            logger.error("Either a structure or a composition must be provided.")
            raise ValueError("Either a structure or a composition must be provided")
        
        if ase_composition is not None:
            ase_atoms = ase_composition
            has_structure=False
        else:
            ase_atoms = ase_structure
            has_structure=True

        entry_data={}
        # Add any custom data from the data argument
        if data:
            logger.debug(f"Adding custom data: {data}")
            entry_data.update(data)

        # Write to the ASE database
        if db is None:
            with connect(self.db_path) as db:
                db.write(ase_atoms, data=entry_data, has_structure=has_structure, **kwargs)
        else:
            db.write(ase_atoms, data=entry_data, has_structure=has_structure, **kwargs)

        logger.info("Material added successfully.")

        return None
    
    def add_many(self, materials: List[Dict]):
        """
        Write many materials to the database in a single transaction.
        
        Args:
            materials (List[Dict]): A list of dictionaries where each dictionary represents a material
                                    with keys like 'composition', 'structure', 'coords', etc.
        """
        logger.info(f"Adding {len(materials)} materials to the database.")
        with connect(self.db_path) as db:
            for material in materials:
                self.add_material(db=db, **material)
        logger.info("All materials added successfully.")

    def read(self, selection=None , **kwargs):
        """Read materials from the database by ID."""
        logger.debug(f"Reading materials with selection: {selection}, filters: {kwargs}")
        return self.db.select(selection=selection, **kwargs)

    def read_data(self, selection=None , **kwargs):
        """Read data in the database by ID."""
        logger.debug(f"Reading data with selection: {selection}, filters: {kwargs}")
        rows=self.db.select(selection=selection, **kwargs)
        data=[row.data for row in rows]
        return data

    def update_material(self, material_id: int,
                        structure: Atoms = None,
                        data: Dict = None,
                        delete_keys: List[str] = [],
                        db=None,
                        **kwargs):
        logger.info(f"Updating material with ID {material_id}.")

        if delete_keys is None:
            delete_keys=[]

        entry_data={}
        # Add any custom data from the data argument
        if data:
            logger.debug(f"Updating properties: {data}")
            entry_data.update(data)

        # Write to the ASE database
        if db is None:
            logger.debug("Opening database connection for updating.")
            with connect(self.db_path) as db:
                db.update(material_id, atoms=structure, data=entry_data, delete_keys=delete_keys, **kwargs)
        else:
            db.update(material_id, atoms=structure, data=entry_data, delete_keys=delete_keys, **kwargs)
        logger.info(f"Material with ID {material_id} updated successfully.")

    def update_many(self, update_list: List[Dict]):
        """
        Update many rows in the database in a single transaction.
        
        Args:
            updates (List[Tuple(id,Dict)]): A list of of tuples where the first element is the material ID 
            and the second is a dictionary of key value pairs for the update operation.
        """
        logger.info(f"Updating {len(update_list)} materials in the database.")
        with connect(self.db_path) as db:
            for update in update_list:
                id=update[0]
                material_dict=update[1]
                self.update_material(id=id, db=db, **material_dict)
        logger.info("All materials updated successfully.")

    def delete_data(self, delete_keys: List[str]):
        """
        Delete data from a material in the database.
        
        Args:

            delete_keys (List[str]): A list of data keys to delete.
        """
        logger.info(f"Deleting data {delete_keys} from all materials.")
        rows=self.read()
        with connect(self.db_path) as db:
            for row in rows:
                self.update_material(id=row.id, delete_keys=delete_keys, db=db)
        logger.info("Data deleted successfully.")

    def delete_material(self, material_ids: List[int]):
        """Delete a material from the database by ID."""
        logger.info(f"Deleting material with ID {material_ids}.")
        with connect(self.db_path) as db:
            db.delete(material_ids)
        logger.info(f"Material with ID {material_ids} deleted successfully.")

    def add_metadata(self, metadata: Dict):
        """Add metadata to the database."""
        logger.info("Adding metadata to the database.")
        with connect(self.db_path) as db:
            db_metadata=db.metadata
            db_metadata.update(metadata)
            db.metadata=db_metadata
        logger.debug(f"Metadata added: {metadata}")

    def delete_metadata(self, delete_keys: List[str]):
        """
        Delete metadata from the database.
        
        Args:

            delete_keys (List[str]): A list of metadata keys to delete.
        """
        logger.info(f"Deleting metadata keys {delete_keys}.")
        with connect(self.db_path) as db:
            db_metadata=db.metadata
            for key in delete_keys:
                db_metadata.pop(key)
                db.metadata=db_metadata
        logger.info("Metadata deleted successfully.")

    def update_data_properties(self):
        """Update the data properties of the database."""
        logger.info("Updating data properties of the database.")
        rows=self.read()
        for row in rows:
            data_properties=list(row.data.keys())
            self.data_keys.update(data_properties)

        with connect(self.db_path) as db:
            db_metadata=db.metadata
            db_metadata['data_keys']=list(self.data_keys)
            db.metadata=db_metadata
        logger.info("Data properties updated successfully.")

    def get_data(self):
        """Get the data of the database."""
        logger.debug("Retrieving data from the database.")
        return self.data_keys


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