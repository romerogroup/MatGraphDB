import glob
import json
import logging
import os

import numpy as np

from matgraphdb.utils.config import PKG_DIR, config
from matgraphdb.utils.file_utils import load_json, load_parquet, save_parquet

logger = logging.getLogger(__name__)


def extract_coordination_encoding(data: dict):
    """
    Extract coordination encoding and coordination number from the given data dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing at least the 'mp_symbol' key. 'mp_symbol' should be in the format 'symbol:coordination_number'.

    Returns
    -------
    dict or None
        A dictionary with keys 'coordination_encoding' and 'coordination_number' if 'mp_symbol' exists in the input data.
        Returns None if 'mp_symbol' is not found.

    Example
    -------
    >>> data = {'mp_symbol': 'Cu:12'}
    >>> extract_coordination_encoding(data)
    {'coordination_encoding': array([0., 0., ..., 1., 0.]), 'coordination_number': 12}
    """
    mp_symbol = data.get("mp_symbol", None)
    if mp_symbol is None:
        logger.info(f"No 'mp_symbol' key found in data: {data}")
        return None

    coord_encoding = np.zeros(shape=14)
    coord_num = int(mp_symbol.split(":")[-1])
    if coord_num <= 13:
        coord_encoding[coord_num - 1] = 1
    elif coord_num == 20:
        coord_encoding[-1] = 1
    return dict(coordination_encoding=coord_encoding, coordination_number=coord_num)


def convert_coord_geo_json_to_parquet(json_files, parquet_file, **kwargs):
    """
    Convert a list of JSON files containing coordination geometry information to a Parquet file.

    Parameters
    ----------
    json_files : list
        List of paths to JSON files containing coordination geometry information.
    parquet_file : str
        The path where the Parquet file will be saved.
    kwargs
        Additional keyword arguments to pass to the `save_parquet` function.

    Returns
    -------
    pyarrow.Table
        The resulting Parquet table after conversion.

    Example
    -------
    >>> json_files = ['file1.json', 'file2.json']
    >>> convert_coord_geo_json_to_parquet(json_files, 'output.parquet')
    <pyarrow.Table>
    """
    data_list = []
    for file in json_files:
        data = load_json(file)

        data.update(extract_coordination_encoding(data))
        data_list.append(data)

    table = save_parquet(data_list, parquet_file, **kwargs)
    return table


def get_coordination_geometry(
    parquet_file=None, columns=None, output_format="pandas", **kwargs
):
    """
    Load coordination geometry information from a Parquet file.

    This function reads a Parquet file containing coordination geometry information and returns a dictionary
    with atomic symbols as keys and lists of coordinates as values.

    Parameters
    ----------
    parquet_file : str
        The path to the Parquet file containing coordination geometry information.
    columns : list, optional
        A list of column names to include in the returned data. By default, all columns are included.
    output_format : str, optional
        The format of the returned data (default is 'pandas'). Options are 'pandas' or 'pyarrow'.
    kwargs
        Additional keyword arguments to pass to the `load_parquet` function.

    Returns
    -------
    dict
        A dictionary with atomic symbols as keys and lists of coordinates as values.
    """
    resource_dir = os.path.join(config.pkg_dir, "utils", "chem_utils", "resources")
    if parquet_file is None:
        parquet_file = os.path.join(resource_dir, "coordination_geometries.parquet")
    output_formats = ["pandas", "pyarrow"]
    if output_format not in output_formats:
        raise ValueError(f"type must be one of {output_formats}")
    table = load_parquet(parquet_file, columns=columns, **kwargs)
    if output_format == "pandas":
        return table.to_pandas()
    elif output_format == "pyarrow":
        return table
    return table
