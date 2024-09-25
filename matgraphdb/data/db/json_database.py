from hmac import new
import os
import json
from glob import glob
import logging
from multiprocessing import Pool
from functools import partial
from typing import Callable, List, Union

from numpy import record

from matgraphdb import data


# Set up logger
logger = logging.getLogger(__name__)

class JsonDatabase:
    def __init__(self, db_path='MaterialsDatabase', n_cores=4):
        """
        Initializes the JsonDatabase object.

        Args:
            db_path (str): The path to the root directory of the database.
            n_cores (int): The number of CPU cores to be used for parallel processing.
        """
        self.db_path = db_path
        self.db_dir = os.path.join(self.db_path, 'db')

        os.makedirs(self.db_dir, exist_ok=True)

        self.n_cores = n_cores
        self.metadata = {}
        self._load_state()

        logger.info(f"db_dir: {self.db_dir}")
        logger.info(f"n_cores: {self.n_cores}")
        for x in self.metadata.items():
            logger.info(f"{x[0]}: {x[1]}")

    def _load_state(self):
        """Loads the state from the state.json file."""
        state_file = os.path.join(self.db_path, 'state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                self.metadata = json.load(f)
                self.metadata['fields'] = set(self.metadata['fields'])
        else:
            # Initialize metadata if state.json doesn't exist
            self.metadata = {'current_index': 0,
                             'fields':set()}
            self._save_state()

    def _save_state(self):
        """Saves the current state to the state.json file."""
        state_file = os.path.join(self.db_path, 'state.json')
        with open(state_file, 'w') as f:
            metadata=self.metadata.copy()
            metadata['fields']=list(self.metadata['fields'])
            json.dump(metadata, f)

    def process_task(self, func, items, **kwargs):
        logger.info(f"Processing tasks using {self.n_cores} cores")
        with Pool(self.n_cores) as p:
            if isinstance(items[0], tuple):
                logger.info("Using starmap")
                results = p.starmap(partial(func, **kwargs), items)
            else:
                logger.info("Using map")
                results = p.map(partial(func, **kwargs), items)
        return results

    def create(self, data):
        """
        Creates new records in the database.

        Args:
            data (dict or list of dicts): The data to be added to the database.

        Returns:
            List of IDs of the created records.
        """
        if isinstance(data, dict):
            # Single record
            results=[self._create_record(data)]
        elif isinstance(data, list):
            # Multiple records
            num_records = len(data)
            # Assign IDs
            start_index = self.metadata['current_index'] + 1
            ids = list(range(start_index, start_index + num_records))

            # Prepare list of (id, record_data) tuples
            tasks = list(zip(ids, data))

            # Process in parallel
            results = self.process_task(self._create_record_with_id, tasks)

            
        else:
            raise ValueError("Data must be a dict or a list of dicts.")
        

        ids, records_fields = zip(*results)
        # Update state metadata
        for record_fields in records_fields:
            self.metadata['fields'].update(record_fields)
        self.metadata['current_index'] += num_records
        self._save_state()

        logger.info(f"Created records {ids}")

        return ids

    def _create_record(self, record_data):
        """Creates a single record with a new ID."""
        # Get new ID
        self.metadata['current_index'] += 1
        new_id = self.metadata['current_index']
        
        # Save record
        record_file = os.path.join(self.db_dir, f'{new_id}.json')
        with open(record_file, 'w') as f:
            record_fields=list(record_data.keys())
            json.dump(record_data, f)

        self.metadata['fields'].update(record_data)
        self._save_state()
        logger.info(f"Created record {new_id}")
        return new_id,record_fields

    def _create_record_with_id(self, id, record_data):
        """Creates a single record with a specified ID."""
        record_file = os.path.join(self.db_dir, f'{id}.json')
        with open(record_file, 'w') as f:
            record_fields=list(record_data.keys())
            json.dump(record_data, f)
        logger.info(f"Created record {id}")
        return id, record_fields

    def read(self, ids=None, filter_func=None, fields=[], negate=False):
        """
        Reads records from the database.

        Args:
            ids (int or list of ints): The IDs of the records to read.
              If None, reads all records in the database.
            filter_func (function): A function to filter the record data.
              If provided, the function is applied to each record data before returning.
            fields (list of str): A list of fields to include in the returned data.
              If empty, all fields are included.
            negate (bool): If True, field will be excluded from the returned data.

        Returns:
            List of record data.
        """
        if ids is None:
            # Read all record IDs from the db_dir
            logger.info("No IDs provided. Reading all records in the database.")
            try:
                files = os.listdir(self.db_dir)
                # Extract IDs from filenames matching the pattern 'm-<id>.json'
                ids = []
                for fname in files:
                    id=fname.split('-')[-1].split('.')[0]
                    ids.append(int(id))
                if not ids:
                    logger.info("No records found in the database.")
                    return []
            except Exception as e:
                logger.error(f"Error while listing records: {e}")
        if isinstance(ids, int):
            # Single record
            return [self._read_record(ids)]
        elif isinstance(ids, list):
            # Multiple records
            return self.process_task(self._read_record, ids, filter_func=filter_func, fields=fields, negate=negate)
        else:
            raise ValueError("IDs must be an int or a list of ints.")

    def _read_record(self, id, filter_func=None, fields=[], negate=False):
        """Reads a single record."""
        record_file = os.path.join(self.db_dir, f'{id}.json')
        if os.path.exists(record_file):
            with open(record_file, 'r') as f:
                data = json.load(f)
                if filter_func is not None:
                    data = filter_func(data)

                if fields:
                    raw_data = {}
                    for key, value in data.items():
                        if negate:
                            if key not in fields:
                                raw_data[key] = value
                        else:
                            if key in fields:
                                raw_data[key] = value
                    data = raw_data
            logger.info(f"Read record {id}")
            return data
        else:
            logger.warning(f"Record {id} does not exist.")
            return None

    def update(self, ids, data):
        """
        Updates records in the database.

        Args:
            ids (int or list of ints): The IDs of the records to update.
            data (dict or list of dicts): The new data for the records.

        Returns:
            None
        """
        if isinstance(ids, int) and isinstance(data, dict):
            # Single record
            results=self._update_record(ids, data)
        elif isinstance(ids, list) and isinstance(data, list) and len(ids) == len(data):
            # Multiple records
            tasks = list(zip(ids, data))
            results=self.process_task(self._update_record, tasks)
        else:
            raise ValueError("IDs and data must be matching lists.")
        
        ids, records_fields = zip(*results)
        for record_fields in records_fields:
            self.metadata['fields'].update(record_fields)
        self._save_state()
        return ids

    def _update_record(self, id, new_data):
        """Updates a single record."""
        record_file = os.path.join(self.db_dir, f'{id}.json')
        if os.path.exists(record_file):
            with open(record_file, 'r') as f:
                record_data = json.load(f)
            
            record_data.update(new_data)

            new_record_fields=list(new_data.keys())

            with open(record_file, 'w') as f:
                json.dump(record_data, f)

            logger.info(f"Updated record {id}")
        else:
            logger.warning(f"Record {id} does not exist.")

        return id, new_record_fields

    def delete(self, ids):
        """
        Deletes records from the database.

        Args:
            ids (int or list of ints): The IDs of the records to delete.

        Returns:
            None
        """
        if isinstance(ids, int):
            # Single record
            self._delete_record(ids)
        elif isinstance(ids, list):
            # Multiple records
            self.process_task(self._delete_record, ids)
        else:
            raise ValueError("IDs must be an int or a list of ints.")

    def _delete_record(self, id):
        """Deletes a single record."""
        record_file = os.path.join(self.db_dir, f'{id}.json')
        if os.path.exists(record_file):
            os.remove(record_file)
            logger.info(f"Deleted record {id}")
        else:
            logger.warning(f"Record {id} does not exist.")


    def apply(self,func:Callable, ids:Union[List,int]=None):
        """
        Updates records in the database.

        Args:
            ids (int or list of ints): The IDs of the records to update.
            func (Callable): A function to apply to each record data. 
            This function will operate on each dictionary. It should return the record dictionary

        Returns:
            None
        """
        if ids is None:
            # Read all record IDs from the db_dir
            logger.info("No IDs provided. Reading all records in the database.")
            try:
                files = os.listdir(self.db_dir)
                # Extract IDs from filenames matching the pattern 'm-<id>.json'
                ids = []
                for fname in files:
                    id=fname.split('-')[-1].split('.')[0]
                    ids.append(int(id))
                if not ids:
                    logger.info("No records found in the database.")
                    return []
            except Exception as e:
                logger.error(f"Error while listing records: {e}")

        if isinstance(ids, int) and isinstance(data, dict):
            # Single record
            results=self._apply_record(ids, func=func)
        elif isinstance(ids, list) and isinstance(data, list) and len(ids) == len(data):
            # Multiple records
            tasks = list(zip(ids, data))
            results=self.process_task(self._apply_record, tasks, func=func)
        else:
            raise ValueError("IDs and data must be matching lists.")
        
        ids, records_fields = zip(*results)
        for record_fields in records_fields:
            self.metadata['fields'].update(record_fields)
        self._save_state()
        return ids

    def _apply_record(self, id, func=None):
        """Applies a function to a single record."""
        record_file = os.path.join(self.db_dir, f'{id}.json')
        if os.path.exists(record_file):
            with open(record_file, 'r') as f:
                record_data = json.load(f)
                record_data=func(record_data)

            new_record_fields=list(record_data.keys())

            with open(record_file, 'w') as f:
                json.dump(record_data, f)

            logger.info(f"Applied function to record {id}")
        else:
            logger.warning(f"Record {id} does not exist.")

        return id, new_record_fields
    
    def update_fields_metadata(self):
        # Read all record IDs from the db_dir
        try:
            files = os.listdir(self.db_dir)
            # Extract IDs from filenames matching the pattern 'm-<id>.json'
            ids = []
            for fname in files:
                id=fname.split('-')[-1].split('.')[0]
                ids.append(int(id))
            if not ids:
                logger.info("No records found in the database.")
                return []
        except Exception as e:
            logger.error(f"Error while listing records: {e}")

        tasks = list(zip(ids, data))
        results=self.process_task(self._apply_record, tasks, func=lambda x: x)

        ids, records_fields = zip(*results)
        new_fields=set()
        for record_fields in records_fields:
            new_fields.update(record_fields)
        self.metadata['fields']=new_fields
        self._save_state()
        return self.metadata['fields']
            




if __name__ == "__main__":
    # data
    db = JsonDatabase(db_path='data/raw/MatGraphDB_test')

    # print(db.metadata)
    entry_data = {'field1': 'value1', 'field2': 'value2'}

    with open('test.json','w') as f:
        json.dump(entry_data, f)

    entry_data = {'field3': 'value3'}

 

    with open('test.json','w') as f:
        json.dump(entry_data, f)

    # entry_data = {'field1': 'value1', 'field2': 'value2'}

    # entry_data_list=[entry_data for _ in range(10)]
    # db.create(entry_data_list)
    # print(db.read(fields=['field1'], negate=True))

    # db.create(test_entry_data)
    # db.read()
    # db.create(test_entry_data)
    # db.read()
    # db.create({'field':set()})
    # db.read()
    # dict={'field':set()}

    # with open('state.json','w') as f:
    #     json.dump(dict,f)