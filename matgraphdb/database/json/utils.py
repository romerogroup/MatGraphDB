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


PROPERTY_NAMES=[
    "material_id",
    "nsites",
    "elements",
    "nelements",
    "composition",
    "composition_reduced",
    "formula_pretty",
    "volume",
    "density",
    "density_atomic",
    "symmetry",
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
    "is_gap_direct",
    "is_metal",
    "is_magnetic",
    "ordering",
    "total_magnetization",
    "total_magnetization_normalized_vol",
    "num_magnetic_sites",
    "num_unique_magnetic_sites",
    "k_voigt",
    "k_reuss",
    "k_vrh",
    "g_voigt",
    "g_reuss",
    "g_vrh",
    "universal_anisotropy",
    "homogeneous_poisson",
    "e_total",
    "e_ionic",
    "e_electronic",
    "wyckoffs"
]
