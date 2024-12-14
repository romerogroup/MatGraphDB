import json
import os
import pandas as pd
import time
import random

# Function to create large JSON files
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

# Function to measure time taken to read a JSON file
def measure_read_time(file_path):
    start_time = time.time()
    
    # Method 1: Using json.load
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    method1_time = time.time() - start_time
    
    # Method 2: Using pandas.read_json
    start_time = time.time()
    pd_data = pd.read_json(file_path)
    method2_time = time.time() - start_time
    
    return method1_time, method2_time

# Experiment with different file sizes
# file_sizes = [1000, 10000, 100000, 1000000]  # Number of entries to generate
file_sizes = [10, 100, 200, 500, 1000, 10000]  # Number of entries to generate
results = []

for size in file_sizes:
    file_path = f"sandbox/json_file_{size}.json"
    
    # Create JSON file of the specified size
    create_json_file(file_path, size)
    
    # Measure reading times
    method1_time, method2_time = measure_read_time(file_path)
    
    results.append({
        "File Size (Entries)": size,
        "json.load Time (s)": method1_time,
        "pandas.read_json Time (s)": method2_time
    })
    
    # # Remove the generated file
    # os.remove(file_path)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

#
# Plot the results
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
import matplotlib.pyplot as plt

plt.plot(results_df["File Size (Entries)"], results_df["json.load Time (s)"], label='json.load')
plt.plot(results_df["File Size (Entries)"], results_df["pandas.read_json Time (s)"], label='pandas.read_json')

plt.scatter(results_df["File Size (Entries)"], results_df["json.load Time (s)"], label='json.load')
plt.scatter(results_df["File Size (Entries)"], results_df["pandas.read_json Time (s)"], label='pandas.read_json')
plt.xlabel('File Size (Entries)')
plt.ylabel('Time (Seconds)')
plt.title('Performance of JSON Reading Methods')
plt.legend()
# plt.xscale('log')
# plt.yscale('log')
plt.savefig('sandbox/json_read_performance_not-log.png')
