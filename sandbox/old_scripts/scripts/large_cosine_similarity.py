from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
import numpy as np

def calculate_similarity_chunk(chunk):
    # Calculate cosine similarity for a chunk of data
    return cosine_similarity(chunk)

def chunkify(lst, n):
    """Divide a list of items into n chunks"""
    return [lst[i::n] for i in range(n)]

def calculate_similarity_multiprocessing(data):
    # Number of processes
    num_processes = cpu_count() - 8 
    print("Number of processes: ", num_processes)

    # Split data into chunks for each process
    chunks = chunkify(data, num_processes)
    
    # Create a multiprocessing Pool
    pool = Pool(processes=num_processes)
    
    # Map the calculate_similarity_chunk function to each data chunk
    result_chunks = pool.map(calculate_similarity_chunk, chunks)
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    
    # Combine the results from the chunks
    # Since result_chunks is a list of 2D arrays, you might need additional steps to 
    # combine these into a full similarity matrix, depending on how you've chunked the data.
    similarity_matrix = np.vstack(result_chunks)
    
    return similarity_matrix



if __name__ == '__main__':
    # Example dataset
    data = np.random.rand(100, 50)  # 100 items with 50 features each

    # Calculate similarity
    similarity_matrix = calculate_similarity_multiprocessing(data)