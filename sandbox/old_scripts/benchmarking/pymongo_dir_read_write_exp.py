import os
import json
import time
import random
from pymongo import MongoClient

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')  # Adjust if using a remote MongoDB
db_name = 'MatgraphDB'
collection_name = 'test'
db = client[db_name]
collection = db[collection_name]

# Directory to hold the JSON files
directory = "sandbox/mongodb_files_experiment"
os.makedirs(directory, exist_ok=True)

# Function to create a single JSON file with specified number of entries
def create_json_file(file_path, num_entries):
    data = []
    for _ in range(num_entries):
        entry = {
            "id": random.randint(1, 100000),
            "name": f"name_{random.randint(1, 100000)}",
            "value": random.random(),
            "is_active": random.choice([True, False]),
            "tags": [f"tag_{i}" for i in range(random.randint(1, 10))]
        }
        data.append(entry)
    
    with open(file_path, "w") as f:
        json.dump(data, f)

# Function to insert data into MongoDB
def insert_data_to_mongo(collection, json_data):
    # Insert the data into MongoDB
    start_time = time.time()
    collection.insert_many(json_data)  # Use insert_many for batch insert
    write_time = time.time() - start_time
    return write_time

# Function to read the JSON files from the directory
def read_json_files(directory):
    start_time = time.time()
    
    # List all the JSON files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
    
    json_data = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            json_data.extend(data)  # Add each file's data to the list

    read_time = time.time() - start_time
    return json_data, read_time

# Function to read all data from MongoDB
def read_data_from_mongo(collection):
    start_time = time.time()
    
    # Retrieve all documents from the collection
    data = list(collection.find({}))  # Convert cursor to a list
    
    read_time = time.time() - start_time
    return data, read_time


# Experiment setup
num_files = 10000
num_entries_per_file = 1000

# # Generate the JSON files if needed (or you can skip this if you already have the files)
# print(f"Generating {num_files} JSON files with {num_entries_per_file} entries each...")
# for i in range(num_files):
#     create_json_file(os.path.join(directory, f"file_{i}.json"), num_entries_per_file)

# # Read the JSON files and insert them into MongoDB
# print("Reading the JSON files...")
# json_data, read_time = read_json_files(directory)

# print("Inserting data into MongoDB...")
# write_time = insert_data_to_mongo(collection, json_data)

# # Output the results
# print(f"Read Time: {read_time:.2f} seconds")
# print(f"Write Time (MongoDB Insert): {write_time:.2f} seconds")



data,read_time = read_data_from_mongo(collection)
print("read time: ", read_time)
# Optionally, clean up the database after the experiment (e.g., for future tests)
# collection.delete_many({})  # Uncomment this to clear the collection after the experiment
