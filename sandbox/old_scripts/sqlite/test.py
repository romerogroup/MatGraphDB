from glob import glob
import json
import logging
import os
import sqlite3

from matgraphdb.utils import timeit

logger = logging.getLogger('matgraphdb')

logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)  

logger.addHandler(ch)

# TODO: Need to improve the interface for writing and reading data
# TODO: Need to allow reads to be more comples.

CREATE_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS main (
        id INTEGER PRIMARY KEY, 
        data TEXT
    );
"""


schema_init_statments= [CREATE_TABLE_QUERY]

table_names= ['main']


class SQLiteDatabase:
    def __init__(self, db_file):
        super().__init__()
        self.db_file = db_file
        self.connection = None
        self.cursor = None

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._disconnect()

    def _connect(self):
        self.connection = sqlite3.connect(self.db_file)
        # Enable foreign key constraints
        self.connection.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.connection.cursor()
        self._create_schema()
        

    def _disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def _create_schema(self):
        """
        Create the necessary schema (tables) in the database.
        """
        self.cursor.execute( 'SELECT COUNT(*) FROM sqlite_master WHERE name="main"')
        if self.cursor.fetchone()[0] == 0:
            logger.info("Creating database")
            for statement in schema_init_statments:
                self.cursor.executescript(statement)
            # Commit the changes
            self.connection.commit()
        else:
            logger.info("Database already created")

    @timeit
    def _write(self, table_name, data):
        """
        Write data to the specified table in the database. Supports bulk insert.
        Parameters:
            table_name: Name of the table to insert data into.
            data: A dictionary or a list of dictionaries representing data.
        """
        if table_name not in table_names:
            raise ValueError(f"Invalid table name '{table_name}'. Valid table names are: {table_names}")

        # Build insert query and data fields based on table schema
        if table_name == 'main':
            insert_query = """
            INSERT OR IGNORE INTO main (data)
            VALUES (?)
            """

            data_fields = ['data']

        else:
            raise ValueError(f"Unknown table '{table_name}'")


        if isinstance(data, list):
            # Bulk insert
            records = []
            for item in data:
                record = tuple(item.get(field) for field in data_fields)
                records.append(record)
            self.cursor.executemany(insert_query, records)
        else:
            # Single insert
            record = tuple(data.get(field) for field in data_fields)
            self.cursor.execute(insert_query, record)

        # Commit the changes
        self.connection.commit()

    @timeit
    def _read(self, table_name):
        """
        Read data from the specified table in the database based on a query.
        Parameters:
            table_name: Name of the table to read data from.
            query: A dictionary with column-value pairs to filter the data.
        """
        if table_name not in table_names:
            raise ValueError(f"Invalid table name '{table_name}'. Valid table names are: {table_names}")

        base_query = f"SELECT * FROM {table_name}"
        conditions = []
        parameters = []

        # for key, value in query.items():
        #     conditions.append(f"{key} = ?")
        #     parameters.append(value)

        # if conditions:
        #     base_query += " WHERE " + " AND ".join(conditions)

        # self.cursor.execute(base_query, parameters)
        self.cursor.execute(base_query)
        rows = self.cursor.fetchall()

        # Get column names
        # columns = [description[0] for description in self.cursor.description]
        # result = [dict(zip(columns, row)) for row in rows]
        return rows


if __name__ == '__main__':
    database_dir='data/raw/SQLiteDB'
    os.makedirs(database_dir,exist_ok=True)
    db = SQLiteDatabase(db_file='data/raw/SQLiteDB/db.db')
    db._connect()


    ex_json_file_1='data/production/materials_project/json_database/mp-27352.json'
    ex_json_file_2='data/production/materials_project/json_database/mp-1000.json'

    files=glob('data/production/materials_project/json_database/*.json')


    with open(ex_json_file_1, 'r') as f:
        data_1 = json.load(f)

    with open(ex_json_file_2, 'r') as f:
        data_2 = json.load(f)

    json_string_1 = json.dumps(data_1)
    json_string_2 = json.dumps(data_2)
    data_list=[]
    for i in range(100000):
        if i%2==0:
            data_list.append({'data':json_string_1})
        else:
            data_list.append({'data':json_string_2})
    # db._write(table_name='main', data=data_list)


    rows=db._read(table_name='main')
    # print(rows)
    