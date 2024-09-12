from functools import partial
import os
import json
import threading
from multiprocessing import Pool

class Database:
    def __init__(self, directory, n_cores=1):
        self.directory = directory
        self.data_dir = os.path.join(self.directory, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.schema_file = os.path.join(self.directory, 'schema.json')
        self.state_file = os.path.join(self.directory, 'state.json')
        self.n_cores=n_cores
        self._initialize_schema_and_state()

    def _initialize_schema_and_state(self):
        """Initialize schema and state from files or set defaults."""
        if not os.path.exists(self.schema_file):
            self.schema = set(['_id'])
            self._save_schema()
        else:
            with open(self.schema_file, 'r') as file:
                self.schema = set(json.load(file))

        if not os.path.exists(self.state_file):
            self.state = {'index': 0}  # Start index at 0
            self._save_state()
        else:
            with open(self.state_file, 'r') as file:
                self.state = json.load(file)

    def _save_schema(self):
        """Save the current schema to schema.json."""
        with open(self.schema_file, 'w') as file:
            json.dump(list(self.schema), file, indent=4)

    def _save_state(self):
        """Save the current state (such as index) to state.json."""
        with open(self.state_file, 'w') as file:
            json.dump(self.state, file, indent=4)

    def _get_file_path_by_id(self, m_id):
        """Get the file path corresponding to the m_id."""
        file_name = f'{m_id}.json'
        return os.path.join(self.data_dir, file_name)

    def _update_schema(self, new_data):
        """Update schema with new properties detected in new_data."""
        new_keys = set(new_data.keys()) - self.schema
        new_keys = new_keys
        if new_keys:
            self.schema.update(new_keys)
            self._save_schema()
        return new_keys
    
    def _remove_mids(self,original_list, remove_list):
        """Remove elements of remove_list from original_list."""
        return [mid for mid in original_list if mid not in remove_list]

    def _apply_new_schema_to_file(self, m_id):
        """Apply new schema properties to all existing JSON files."""
        file_path = os.path.join(self.data_dir, f'{m_id}.json')
        with open(file_path, 'r') as file:
            data = json.load(file)
        # Add new properties to each file with value None


        for key in self.schema:
            if key not in data:
                data[key] = None
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _apply_new_schema(self, m_ids_updated):
        all_mids = self.get_all_mids()
        remaining_mids = self._remove_mids(all_mids, m_ids_updated)
        with Pool(self.n_cores) as pool:
            pool.map(self._apply_new_schema_to_file, all_mids)

    def _validate_keys(self,data_list):
        """Ensure all dictionaries in data_list have the same keys."""
        if not data_list:
            return True  # Nothing to validate if the list is empty
        reference_keys = set(data_list[0].keys())
        for data in data_list:
            if set(data.keys()) != reference_keys:
                raise ValueError("All dictionaries must have the same keys.")
        return True

    def get_all_mids(self):
        """Get all m_ids in the data directory."""
        return [file_name.split('.')[0] for file_name in os.listdir(self.data_dir) if file_name.endswith('.json')]

    def create(self, data, m_id=None, is_multiprocess=False):
        """Create a new record with schema validation."""
        if not m_id:  # If m_id is not provided, generate one based on the current index
            m_id = f'm-{self.state["index"]}'
            self.state['index'] += 1
            
        file_path = self._get_file_path_by_id(m_id)

        data['_id']=m_id

        # Ensure all data has the same keys
        for key in self.schema:
            if key not in data:
                data[key] = None
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print(f'Created file: {m_id}')

        if not is_multiprocess:
            new_keys=self._update_schema(data)
            m_ids_updated=[m_id]
            if new_keys:
                self._apply_new_schema(m_ids_updated)

            self._save_state()
        return m_id

    def read(self, m_id):
        """Read a record from the data directory using m_id."""
        file_path = self._get_file_path_by_id(m_id)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        else:
            print(f'File with ID {m_id} does not exist.')

    def update(self, m_id, new_data, is_multiprocess=False):
        """Update a record using m_id and apply schema changes if needed."""
        file_path = self._get_file_path_by_id(m_id)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            # Merge the new data into the existing data
            data.update(new_data)

            # Ensure all data has the same keys
            for key in self.schema:
                if key not in data:
                    data[key] = None
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            print(f'Updated file with ID: {m_id}')

            if not is_multiprocess:
                new_keys=self._update_schema(data)
                m_ids_updated=[m_id]
                if new_keys:
                    self._apply_new_schema(m_ids_updated)

                self._save_state()
        else:
            print(f'File with ID {m_id} does not exist.')

    def delete(self, m_id):
        """Delete a record from the data directory using m_id."""
        file_path = self._get_file_path_by_id(m_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f'Deleted file with ID: {m_id}')
        else:
            print(f'File with ID {m_id} does not exist.')

    def create_many(self, data_list):
        """Create multiple records in parallel."""

        self._validate_keys(data_list)

        # Generate all m_ids sequentially beforehand
        data_index_list = [(data, 'm-' + str(self.state['index'] + i)) for i, data in enumerate(data_list)]

        # Update the current index after generating m_ids
        self.state['index'] += len(data_list)
        self._save_state()

        # Create records in parallel
        with Pool(self.n_cores) as pool:
            m_ids=pool.starmap(partial(self.create, is_multiprocess=True), data_index_list)

        new_keys=self._update_schema(data_list[0])
        m_ids_updated=m_ids
        if new_keys:
            self._apply_new_schema(m_ids_updated)

        self._save_state()

    def read_many(self, m_id_list):
        """Read multiple records in parallel."""
        with Pool(self.n_cores) as pool:
            return pool.map(self.read, m_id_list)

    def update_many(self, update_list):
        """Update multiple records in parallel. Expects a list of (m_id, new_data)."""
        self._validate_keys(data_list)
        new_keys=self._update_schema(data_list[0])
        with Pool(self.n_cores) as pool:
            pool.starmap(partial(self.update, is_multiprocess=True), update_list)

    def delete_many(self, m_id_list):
        """Delete multiple records in parallel."""
        with Pool(self.n_cores) as pool:
            pool.map(self.delete, m_id_list)



if __name__ == '__main__':
    db = Database(os.path.join('data','raw','test'),n_cores=6)
    print(db.schema)
    # Create multiple records in parallel
    data_list = [
        {'name': 'Alice', 'age': 25},
        {'name': 'Bob', 'age': 30},
        {'name': 'Charlie', 'age': 35}
    ]


    # db.create(data_list[0])
    # db.create({'name': 'Bob', 'age': 30, 'email': 'bob@example.com'})


    # db.update('m-0',{'name': 'Bob', 'age': 30, 'email22': 'bob@example.com'})
    db.create_many(data_list)

    # # Read multiple records in parallel
    # m_id_list = ['m-0', 'm-1', 'm-2']
    # print(db.read_many(m_id_list))

    # # Update multiple records in parallel
    # update_list = [('m-0', {'email': 'alice@example.com'}), ('m-1', {'email': 'bob@example.com'})]
    # db.update_many(update_list)

    # # Delete multiple records in parallel
    # db.delete_many(['m-0', 'm-1'])