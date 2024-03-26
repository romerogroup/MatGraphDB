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


PROPERTIES=[
    ("material_id","string"),
    ("nsites","int"),
    ("elements","string[]"),
    ("nelements","int"),
    ("composition","string"),
    ("composition_reduced","string"),
    ("formula_pretty","string"),
    ("volume","float"),
    ("density","float"),
    ("density_atomic","float"),
    ("symmetry","string"),
    ("energy_per_atom","float"),
    ("formation_energy_per_atom","float"),
    ("energy_above_hull","float"),
    ("is_stable","boolean"),
    ("band_gap","float"),
    ("cbm","float"),
    ("vbm","float"),
    ("efermi","string"),
    ("is_gap_direct","boolean"),
    ("is_metal","boolean"),
    ("is_magnetic","boolean"),
    ("ordering","string"),
    ("total_magnetization","float"),
    ("total_magnetization_normalized_vol","float"),
    ("num_magnetic_sites","int"),
    ("num_unique_magnetic_sites","int"),
    ("k_voigt","float"),
    ("k_reuss","float"),
    ("k_vrh","float"),
    ("g_voigt","float"),
    ("g_reuss","float"),
    ("g_vrh","float"),
    ("universal_anisotropy","float"),
    ("homogeneous_poisson","float"),
    ("e_total","float"),
    ("e_ionic","float"),
    ("e_electronic","float"),
    ("wyckoffs","string[]"),
]
