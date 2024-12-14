import os
import json
import time
import random

# Directory to hold the JSON files
directory = "sandbox/json_files_experiment"

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Function to create a JSON file with specified number of entries
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

# Generate 10,000 JSON files with 1,000 entries each
def generate_json_files(num_files, num_entries_per_file, directory):
    for i in range(num_files):
        file_path = os.path.join(directory, f"file_{i}.json")
        create_json_file(file_path, num_entries_per_file)
        if i % 1000 == 0:  # Progress tracking every 1000 files
            print(f"{i} files created")

# Function to read all JSON files in the directory and measure the time
def read_json_files(directory):
    start_time = time.time()
    
    # List all the JSON files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
    
    data_list = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            data_list.append(data)
    
    total_time = time.time() - start_time
    return total_time

# Experiment setup
num_files = 10000
num_entries_per_file = 1000

# Generate JSON files
print(f"Generating {num_files} JSON files with {num_entries_per_file} entries each...")
begin_time = time.time()
generate_json_files(num_files, num_entries_per_file, directory)
print(f"Time taken to generate JSON files: {time.time() - begin_time:.2f} seconds")

# # Read the JSON files and measure the time
# print("Reading the JSON files and measuring the time...")
# read_time = read_json_files(directory)

# # Report the time taken
# print(f"Total time taken to read {num_files} JSON files: {read_time:.2f} seconds")

# Optionally, you can clean up the files after the experiment
# import shutil
# shutil.rmtree(directory)  # Uncomment this to delete the directory after the experiment
