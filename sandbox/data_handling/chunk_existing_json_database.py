import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List
import math
from matgraphdb import config

def chunk_json_files(
    source_dir: str,
    output_dir: str,
    chunk_size: int = 1000,
    filename_prefix: str = "chunk"
) -> None:
    """
    Chunks multiple JSON files into larger JSON files with specified size.
    
    Args:
        source_dir (str): Directory containing source JSON files
        output_dir (str): Directory where chunked files will be written
        chunk_size (int): Number of records per chunk file
        filename_prefix (str): Prefix for output chunk filenames
    """
    print("Source directory:", source_dir)
    print("Output directory:", output_dir)
    print("Chunk size:", chunk_size)
    print("Filename prefix:", filename_prefix)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize storage for records
    current_chunk: Dict[str, List] = {"entries": []}
    chunk_counter = 0
    file_counter = 0
    
    
    # Get total number of files for progress tracking
    files = glob(os.path.join(source_dir, '*.json'))
    total_files = len(files)
    
    # Process each JSON file in the source directory
    for file_path in files:
        filename = os.path.basename(file_path)
        
        file_counter += 1
        if file_counter % 1000 == 0:
            print(f"Processed {file_counter}/{total_files} files")
            
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                current_chunk["entries"].append(data)
                
                # Write chunk if it reaches the specified size
                if len(current_chunk["entries"]) >= chunk_size:
                    output_path = os.path.join(output_dir,f"matgraphdb_chunk_{chunk_counter}.json")
                    with open(output_path, 'w') as out_file:
                        json.dump(current_chunk, out_file)
                    
                    # Reset chunk
                    current_chunk = {"entries": []}
                    chunk_counter += 1
                    
        except json.JSONDecodeError as e:
            print(f"Error reading {filename}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error processing {filename}: {str(e)}")
    
    # Write remaining records if any
    if current_chunk["entries"]:
        output_path = os.path.join(output_dir,f"matgraphdb_chunk_{chunk_counter}.json")
        with open(output_path, 'w') as out_file:
            json.dump(current_chunk, out_file)

if __name__ == "__main__":
    # Example usage
    source_directory = os.path.join(config.data_dir,'production','materials_project','json_database')
    output_directory = os.path.join(config.data_dir,'production','materials_project','chunked_json')
    
    chunk_json_files(
        source_dir=source_directory,
        output_dir=output_directory,
        chunk_size=10000,
        filename_prefix="chunk"
    )
