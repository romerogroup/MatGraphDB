import numpy as np

def chunk_list(input_list, chunk_size):
    """Divide a list into chunks of a specified size."""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def cosine_similarity(a,b):
    dot_product = np.dot(a,b)
    norm_A = np.linalg.norm(a)
    norm_B = np.linalg.norm(b)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_A * norm_B)
    return similarity