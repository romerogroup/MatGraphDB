import os
import json
import time
import random
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving the plot
import matplotlib.pyplot as plt
import pandas as pd

from multiprocessing import Pool


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

# Function to generate multiple JSON files using multiple cores
def generate_json_files(num_files, num_entries_per_file, directory, num_cores):
    start_time = time.time()
    
    # Create list of tasks for the executor
    tasks = []
    for i in range(num_files):
        file_path = os.path.join(directory, f"file_{i}.json")
        tasks.append((file_path, num_entries_per_file))
    
    # Write files using ProcessPoolExecutor
    with Pool(num_cores) as p:
        p.starmap( create_json_file, tasks)
    
    write_time = time.time() - start_time
    return write_time

# Function to read a single JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to read all JSON files in parallel and measure the time
def read_json_files(directory, num_cores):
    start_time = time.time()
    
    # List all the JSON files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
    
    # Use ProcessPoolExecutor to read files in parallel
    with Pool(num_cores) as p:
        p = p.map(read_json_file, files)
    
    total_time = time.time() - start_time
    return total_time

if __name__ == "__main__":

    # Directory to hold the JSON files
    directory = "sandbox/json_files_experiment"
    
    # Experiment setup
    num_files = 10000
    num_entries_per_file = 1000
    cores_to_test = [1, 2, 4, 8]  # Number of CPU cores to test
    results = []

    # Run the experiment for each core setting
    for cores in cores_to_test:
        core_directory=os.path.join(directory,f'cores_{cores}')
        os.makedirs(core_directory, exist_ok=True)

        # Write JSON files and measure write time
        print(f"Writing JSON files using {cores} cores...")
        write_time = generate_json_files(num_files, num_entries_per_file, core_directory, cores)
        
        # Read JSON files and measure read time
        print(f"Reading JSON files...")
        read_time = read_json_files(core_directory, cores)
        
        results.append({
            "Cores": cores,
            "Write Time (s)": write_time,
            "Read Time (s)": read_time
        })


    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot the results
    plt.plot(results_df["Cores"], results_df["Write Time (s)"], label='Write Time', marker='o')
    plt.plot(results_df["Cores"], results_df["Read Time (s)"], label='Read Time', marker='o')

    plt.scatter(results_df["Cores"], results_df["Write Time (s)"], label='Write Time')
    plt.scatter(results_df["Cores"], results_df["Read Time (s)"], label='Read Time')

    plt.xlabel('Number of Cores')
    plt.ylabel('Time (Seconds)')
    plt.title('JSON Read and Write Performance as a Function of Cores')
    plt.legend()
    plt.savefig('sandbox/json_write_read_parallel_performance.png')

    # Optionally, clean up the directory after the experiment
    # import shutil
    # shutil.rmtree(directory)  # Uncomment this to delete the directory after the experiment
