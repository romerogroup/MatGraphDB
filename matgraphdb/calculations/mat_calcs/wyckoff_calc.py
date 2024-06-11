import pymatgen.core as pmat
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matgraphdb.utils import LOGGER


def calculate_wyckoff_positions(struct):
    """
    Calculates the Wyckoff positions of a given structure.

    Args:
        struct (Structure): The structure to be processed.

    Returns:
        list: A list of Wyckoff positions.
    """
    wyckoffs=None
    try:
        spg_a = SpacegroupAnalyzer(struct)
        sym_dataset=spg_a.get_symmetry_dataset()
        wyckoffs=sym_dataset['wyckoffs']
    except Exception as e:
        LOGGER.error(f"Error processing structure: {e}")

    return wyckoffs

if __name__=='__main__':
    # Testing calculate_wyckoff_positions
    struct = pmat.Structure.from_dict({
        "lattice": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]],
        "species": ["H", "H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]],
    })
    calculate_wyckoff_positions(struct)
