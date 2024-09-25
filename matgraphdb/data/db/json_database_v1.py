from enum import Enum
import multiprocessing
import os
import json
from glob import glob
from multiprocessing import Pool
from functools import partial
import uuid

from matgraphdb.utils import DB_DIR,DB_CALC_DIR,N_CORES, GLOBAL_PROP_FILE, ENCODING_DIR, EXTERNAL_DATA_DIR, MP_DIR
from matgraphdb.utils import get_logger
from matgraphdb.utils.periodic_table import atomic_symbols


logger=get_logger(__name__, console_out=False, log_level='info')

class JsonDatabase:
    def __init__(self, db_path='MaterialsDatabase',  n_cores=N_CORES):
        """
        Initializes the Manager object.

        Args:
            root_dir (str): The path to the root directory of the database.
            db_dir (str): The path to the directory where the database is stored.
            calc_dir (str): The path to the directory where calculations are stored.
            n_cores (int): The number of CPU cores to be used for parallel processing.

        """
        self.db_path=db_path
        self.db_dir = os.path.join(self.db_path,'db')

        os.makedirs(self.db_dir,exist_ok=True)
  

        self.n_cores = n_cores
        self.metadata={}
        self._load_state()

        logger.info(f"db_dir: {self.db_dir}")
        logger.info(f"n_cores: {self.n_cores}")

    @property
    def n_files(self):
        return len(self.get_filepaths())

    @property
    def current_index_counter(self):
        return self.metadata['current_index_counter']
    @current_index_counter.setter
    def current_index_counter(self, value):
        self.metadata['current_index_counter']=value
        
    def create(self, data):
        """
        Creates a material in the database.

        Args:
            data (dict): The data of the material to be created.

        Returns:
            str: The filepath of the created material.
        """
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")
        filepath=self._save_data(data)
        self._save_state()
        return filepath
    
    def _create_many_task(self, data, index):
        """
        Creates a material in the database.

        Args:
            data (dict): The data of the material to be created.

        Returns:
            str: The filepath of the created material.
        """
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")
        filepath=self._save_data(data,m_id=index)
        return filepath

    def create_many(self, data_list):
        """
        Creates many materials in the database.

        Args:
            data_list (list): A list of dictionaries representing the data of the materials to be created.

        Returns:
            list: A list of filepaths of the created materials.
        """
        data_index_list=[(data,'m-' + str(self.current_index_counter + i) ) for i,data in enumerate(data_list)]
        results=self.process_task(self._create_many_task, data_index_list)

        # properties=[]
        # for result in results:
        #     for property_name in result:
        #         if result is None:
        #             properties.append(property_name)
        self.current_index_counter+=len(data_index_list)
        self._save_state()
        return results

    def read(self, m_id):
        """
        Reads a material from the database.

        Args:
            m_id (str): The ID of the material to be read.

        Returns:
            dict: The data of the material.
        """
        logger.info(f"Reading material | {m_id}")
        return self.load(m_id=m_id)
    
    def read_many(self, m_ids=None):
        """
        Reads many materials from the database.

        Args:
            m_ids (list): A list of IDs of the materials to be read.

        Returns:
            list: A list of dictionaries representing the data of the materials.
        """
        logger.info(f"Reading multiple materials")
        if m_ids is None:
            m_ids=self.get_m_ids()
        return self.process_task(self.read, m_ids)

    def update(self, data, m_id, update_state=True):
        """
        Updates a material in the database.

        Args:
            data (dict): The data of the material to be updated.
            m_id (str): The ID of the material to be updated.

        Returns:
            str: The filepath of the updated material.
        """
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")
        filepath=self._save_data(data, m_id=m_id)
        if update_state:
            logger.info(f"Updating db state")
            self._save_state()
        return filepath

    def update_many(self, data_list, m_ids):
        """
        Updates many materials in the database.

        Args:
            data_list (list): A list of dictionaries representing the data of the materials to be updated.
            m_ids (list): A list of IDs of the materials to be updated.

        Returns:
            list: A list of filepaths of the updated materials.
        """
        filepaths=[]
        if len(data_list)!=len(m_ids):
            raise ValueError("data_list and m_ids must be the same length")
        

        mp_list=[(data,m_id) for data,m_id in zip(data_list,m_ids)]

        logger.info(f"Attempting to update {len(mp_list)} materials")
        self.process_task(self.update, mp_list, update_state=False)

        logger.info(f"Successfully updated {len(mp_list)} materials")

        self._save_state()
        return filepaths
    
    def delete(self, m_id):
        """
        Deletes a material from the database.

        Args:
            m_id (str): The ID of the material to be deleted.

        Returns:
            None
        """
        logger.info(f"Deleting material | {m_id}")
        os.remove(self.get_filepath(m_id))

    def delete_many(self, m_ids):
        """
        Deletes many materials from the database.

        Args:
            m_ids (list): A list of IDs of the materials to be deleted.

        Returns:
            None
        """
        self.process_task(self.delete, m_ids)

    def get_properties(self, update=False):
        """
        Returns a list of properties available in the database.
        Returns:
            list: A list of property names.

        """
        if 'properties' in self.metadata and not update:
            return self.metadata['properties']
        

        mp_ids=self.get_m_ids()
        if len(mp_ids)==0:
            properties=[]
        else:
            m_id=mp_ids[0]
            data=self.load(m_id=m_id)
            properties=list(data.keys())

        self.metadata['properties']=properties
        return properties

    def get_m_ids(self):
        """
        Returns a list of m_ids in the database.

        Returns:
            list: A list of m_ids.
        """
        return [file.split(os.sep)[-1].split('.')[0] for file in self.get_filepaths()]
    
    def get_filepath(self, m_id):
        """
        Returns the filepath for the specified m_id.

        Args:
            m_id (str): The m_id of the material.

        Returns:
            str: The filepath for the specified m_id.
        """
        return os.path.join(self.db_dir, f"{m_id}.json")
    
    def get_filepaths(self, m_ids=None):
        """
        Returns a list of filepaths for the specified m_ids.

        Args:
            m_ids (list): A list of m_ids.

        Returns:
            list: A list of filepaths.
        """
        if m_ids:
            return [os.path.join(self.db_dir, f"{m_id}.json") for m_id in m_ids]
        else:
            return glob(self.db_dir + os.sep + '*.json')
    
    def get_metadata(self):
        """
        Returns the metadata of the database.

        Returns:
            dict: The metadata of the database.
        """
        metadata_path=os.path.join(self.root_dir,'metadata.json')
        logger.info(f"Attempting to get metadata from {metadata_path}")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata=json.load(f)
            logger.info(f"Successfully loaded metadata from {metadata_path}")
            return metadata
        else:
            logger.info(f"No metadata found.")
            return {}
    
    def process_task(self, func, list, **kwargs):
        logger.info(f"Process full database using {self.n_cores} cores")
        logger.info(f"Using {self.n_cores} cores")
        with Pool(self.n_cores) as p:
            # multi-arg functions
            if isinstance(list[0], tuple):
                logger.info(f"Using starmap")
                results=p.starmap(partial(func,**kwargs), list)
            else:
                logger.info(f"Using map")
                results=p.map(partial(func,**kwargs), list)
        return results
    
    def load(self, m_id=None, filepath=None):
        """
        Loads a material from the database.

        Args:
            m_id (str): The ID of the material to be loaded.

        Returns:
            dict: The data of the material.
        """
        logger.info(f"Loading material | {m_id}")
        try:
            if filepath:
                filepath=file
            if m_id:
                filepath=os.path.join(self.db_dir, f"{m_id}.json")
            with open(filepath, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"Material {m_id} not found in the database")
            return {}
        except Exception as e:
            raise e

    def load_all(self):
        files=self.get_filepaths()
        results=self.process_task(DatabaseMPTasks.load_data_task,files)
        return results
    
    def refresh_state(self):
        """
        Refreshes the state of the database.
        """
        pass

    def _save_data(self, data, m_id=None):
        """
        Saves data to a JSON file.

        Args:
            data (dict): The data to be saved.

        Returns:
            str: The path to the JSON file.
        """
        if m_id is None:
            m_id=self._get_new_index()
        data['id']=m_id
        filepath=os.path.join(self.db_dir,f"{m_id}.json")

        # Check if the file already exists and update it if it does
        existing_data={}
        logger.info(f"Attempting to saving material | {m_id} | {filepath}")
        if os.path.exists(filepath):
            with open(filepath) as f:
                existing_data = json.load(f)
        existing_data.update(data)

        logger.info(f"Attempting to saving material | {m_id} | {filepath}")
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=4)

        logger.info(f"Successfully saving material | {m_id}")

        return filepath
    
    def _increment_index(self):
        """
        Safely increments the current index counter using a lock.
        """
        self.current_index_counter += 1
        return self.current_index_counter
    
    def _get_new_index(self):
        """
        Returns a new index for the database.

        Returns:
            int: The new index.
        """
        m_id=f"m-{self.current_index_counter}"
        self._increment_index()
        return m_id
    
    def _get_largest_index(self):
        """
        Returns the largest index in the database.

        Returns:
            int: The largest index.
        """
        m_ids=self.get_m_ids()
        if m_ids==[]:
            return 0
        else:
            return max(m_ids)
    
    def _index_files(self, files):
        """
        Indexes a list of files.
        Args:
            files (list): A list of file paths.
        Returns:
            dict: A dictionary where the keys are the filenames and the values are the indexed filenames.

        """
        return {file.split(os.sep)[-1]:f"m-{i}" for i,file in enumerate(files)}

    def _save_state(self):
        """
        Saves the current state of the database.

        Returns:
            None
        """
        
        metadata_path=os.path.join(self.root_dir,'metadata.json')

        logger.info(f"Attempting to saving state to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        logger.info(f"Successfully saved state to {metadata_path}")
    
    def _get_id(self) -> int:
        return int(str(uuid.uuid4().int)[:18])

    def _cast_id(self, pk) -> int:
        return int(pk)

    def _load_state(self):
        """
        Loads the current state of the database.

        Returns:
            None
        """
        metadata_path=os.path.join(self.root_dir,'metadata.json')
        logger.info(f"Attempting to load state.")
        self.metadata=self.get_metadata()
        if self.metadata:
            for key in self.metadata.keys():
                setattr(self, key, self.metadata[key])
            logger.info(f"Successfully loaded state.")
            return self.metadata
        
        logger.info(f"No state found. Creating new state.")

        self.metadata={}
        self.metadata['root_dir']=self.root_dir
        self.metadata['db_dir']=self.db_dir
        self.metadata['calc_dir']=self.calc_dir
        self.metadata['current_index_counter']=0
        self.metadata['properties']=self.get_properties()

        with open(metadata_path,'w') as f:
            json.dump(self.metadata, f, indent=4)
        logger.info(f"Successfully created new state.")
        return self.metadata


class DatabaseMPTasks:
    def load_data_task(json_file):
        """
        Loads the data tasks.

        Returns:
            dict: A dictionary of data tasks.
        """
        try:
            with open(json_file) as f:
                data = json.load(f)
        except Exception as e:
            logger.debug(f"Error processing file: {e}")
            return None

        return data


if __name__=='__main__':
    db=Database(root_dir=os.path.join('data','raw','test'),n_cores=6)
    print(db.get_metadata())

    # data_list=[{'test1':i} for i in range(1000)]
    # db.create_many(data_list)



    data_list=[{'test2':i} for i in range(1000)]
    m_ids=[f'm-{i}' for i in range(1000)]
    db.update_many(data_list,m_ids)

    # print(db._get_id())

    # print(uuid.uuid4())
    # print(len(data_list))
    # db.create_many(data_list)


    # db.update_many(data_list,m_ids)

    # # print(db._index_counter)

    # db.create_material(composition='Li2O')
    # db.create_material(composition='Li2O')




    # def create_material(self,
    #                     composition:Union[str,dict,Composition]=None,
    #                     structure:Structure=None,
    #                     coords:Union[List[Tuple[float,float,float]],np.ndarray]=None,
    #                     coords_are_cartesian :bool=False,
    #                     species:List[str]=None,
    #                     lattice:Union[List[Tuple[float,float,float]],np.ndarray]=None,
    #                     properties:dict=None):

    #     """
    #     Create a material entry in the database.

    #     Args:
    #         composition (Union[str, dict, Composition], optional): The composition of the material. It can be provided as a string, a dictionary, or an instance of the Composition class. Defaults to None.
    #         structure (Structure, optional): The structure of the material. Defaults to None.
    #         coords (Union[List[Tuple[float,float,float]], np.ndarray], optional): The atomic coordinates of the material. Defaults to None.
    #         coords_are_cartesian (bool, optional): Whether the atomic coordinates are in Cartesian coordinates. Defaults to False.
    #         species (List[str], optional): The atomic species of the material. Required if coords is provided. Defaults to None.
    #         lattice (Union[List[Tuple[float,float,float]], np.ndarray], optional): The lattice parameters of the material. Required if coords is provided. Defaults to None.
    #         properties (dict, optional): Additional properties of the material. Defaults to None.

    #     Returns:
    #         str: The path to the JSON file containing the material data.
    #     """

    #     if isinstance(composition, Composition):
    #         composition = composition
    #     if isinstance(composition, str):
    #         composition = Composition(composition)
    #     elif isinstance(composition, dict):
    #         composition = Composition.from_dict(composition)
    #     else:
    #         composition=None

    #     if coords:
    #         if not species:
    #             raise ValueError("If coords is used, species must be provided")
    #         if not lattice:
    #             raise ValueError("If coords is used, lattice must be provided")

    #     if species:
    #         if not coords:
    #             raise ValueError("If species is provided, coords must be provided")
    #         if not lattice:
    #             raise ValueError("If species is provided, lattice must be provided")

    #     if lattice:
    #         if not species:
    #             raise ValueError("If lattice is provided, species must be provided")
    #         if not coords:
    #             raise ValueError("If lattice is provided, coords must be provided")

    #     if isinstance(structure, Structure):
    #         structure = structure
    #     elif coords:
    #         if coords_are_cartesian:
    #             structure = Structure(lattice, species, coords, coords_are_cartesian=True)
    #         else:
    #             structure = Structure(lattice, species, coords)
    #     else:
    #         structure=None

    #     if structure is None and composition is None:
    #         raise ValueError("Either a structure or a composition must be provided")

    #     if structure:
    #         composition=structure.composition

    #     default_properties=self.get_properties()
    #     if default_properties:
    #         data={property_name:None for property_name in default_properties}
    #     else:
    #         data={}

    #     data["elements"]=list(composition.as_dict().keys())
    #     data["nelements"]=len(composition.as_dict())
    #     data["composition"]=composition.as_dict()
    #     data["composition_reduced"]=dict(composition.to_reduced_dict)
    #     data["formula_pretty"]=composition.to_pretty_string()

    #     if structure:
    #         data["volume"]=structure.volume
    #         data["density"]=structure.density
    #         data["nsites"]=len(structure.sites)
    #         data["density_atomic"]=data["nsites"]/data["volume"]
    #         data["structure"]=structure.as_dict()

    #         symprec=0.01
    #         sym_analyzer=SpacegroupAnalyzer(structure,symprec=symprec)
    #         data["symmetry"]={}
    #         data["symmetry"]["crystal_system"]=sym_analyzer.get_crystal_system()
    #         data["symmetry"]["number"]=sym_analyzer.get_space_group_number()
    #         data["symmetry"]["point_group"]=sym_analyzer.get_point_group_symbol()
    #         data["symmetry"]["symbol"]=sym_analyzer.get_hall()
    #         data["symmetry"]["symprec"]=symprec

    #         sym_dataset=sym_analyzer.get_symmetry_dataset()

    #         data["wyckoffs"]=sym_dataset['wyckoffs']
    #         data["symmetry"]["version"]="1.16.2"
  

    #         # try:
    #         #     data["bonding_cutoff_connections"]=calculate_cutoff_bonds(structure)
    #         # except:
    #         #     pass
  
    #         # try:
    #         #     coordination_environments, nearest_neighbors, coordination_numbers = calculate_chemenv_connections(structure)
    #         #     data["coordination_environments_multi_weight"]=coordination_environments
    #         #     data["coordination_multi_connections"]=nearest_neighbors
    #         #     data["coordination_multi_numbers"]=coordination_numbers
    #         # except:
    #         #     pass

    #     filepath=self._save_data(data)
    #     self._save_state()
    #     return filepath