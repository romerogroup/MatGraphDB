import json
import os
from itertools import combinations, combinations_with_replacement

import numpy as np

from matgraphdb.utils.config import PKG_DIR, config
from matgraphdb.utils.file_utils import load_parquet, save_parquet

S_COLUMNS = np.arange(1, 3)
P_COLUMNS = np.arange(27, 33)
D_COLUMNS = np.arange(17, 27)
F_COLUMNS = np.arange(3, 17)


def get_element_properties(
    parquet_file=None, type="imputed", columns=None, output_format="pandas", **kwargs
):
    """
    Load element properties from a parquet file based on type and return it in the specified format.

    Parameters
    ----------
    parquet_file : str, optional
        The path to the parquet file containing the element properties. If not provided, the default file is used.
    type : str, optional
        The type of data to load (default is 'imputed'). Options are 'imputed', 'raw', or 'interim'.
    columns : list, optional
        The column names to load into memory
    output_format : str, optional
        The format of the returned data (default is 'pandas'). Options are 'pandas' or 'pyarrow'.
    kwargs
        The additional keyword arguments for load_parquet

    Returns
    -------
    pandas.DataFrame or pyarrow.Table
        Data containing element properties.

    Example
    -------
    >>> get_element_properties(type='raw', output_format='pandas')
    pandas.DataFrame with element properties
    """
    types = ["imputed", "raw", "interim"]
    output_formats = ["pandas", "pyarrow"]
    if output_format not in output_formats:
        raise ValueError(f"type must be one of {output_formats}")
    if type not in types:
        raise ValueError(f"type must be one of {types}")

    resources_dir = os.path.join(config.pkg_dir, "utils", "chem_utils", "resources")
    if parquet_file is None:
        parquet_file = os.path.join(
            resources_dir, f"{type}_periodic_table_values.parquet"
        )

    table = load_parquet(parquet_file, columns=columns, **kwargs)
    if output_format == "pandas":
        return table.to_pandas()
    elif output_format == "pyarrow":
        return table


def get_group_period_edge_index(df):
    """
    Generate a list of edge indexes based on atomic numbers, extended groups, and periods from the provided dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns 'atomic_number', 'extended_group', 'period', and 'symbol'.

    Returns
    -------
    list
        A list of tuples representing edge indexes between atomic elements.

    Example
    -------
    >>> df = pd.DataFrame({'atomic_number': [1, 2], 'extended_group': [1, 18], 'period': [1, 1], 'symbol': ['H', 'He']})
    >>> get_group_period_edge_index(df)
    [(0, 1)]
    """
    if "atomic_number" not in df.columns:
        raise ValueError("Dataframe must contain 'atomic_number' column")
    if "extended_group" not in df.columns:
        raise ValueError("Dataframe must contain 'extended_group' column")
    if "period" not in df.columns:
        raise ValueError("Dataframe must contain 'period' column")
    if "symbol" not in df.columns:
        raise ValueError("Dataframe must contain 'symbol' column")

    edge_index = []
    for irow, row in df.iterrows():
        symbol = row["symbol"]
        atomic_number = row["atomic_number"]
        extended_group = row["extended_group"]
        period = row["period"]

        if extended_group in S_COLUMNS:
            # Hydrogen
            if period == 1:
                period_neighbors = (None, 1)
                atomic_number_neighbors = (None, None)

            # Lithium
            elif extended_group == 1 and period == 2:
                period_neighbors = (-1, 1)
                atomic_number_neighbors = (None, 1)
            # Francium
            elif extended_group == 1 and period == 7:
                period_neighbors = (-1, None)
                atomic_number_neighbors = (None, 1)
            # Beryllium
            elif extended_group == 2 and period == 2:
                period_neighbors = (None, 1)
                atomic_number_neighbors = (-1, 1)
            # Radium
            elif extended_group == 2 and period == 7:
                period_neighbors = (-1, None)
                atomic_number_neighbors = (-1, 1)
            elif extended_group == 1:
                period_neighbors = (-1, 1)
                atomic_number_neighbors = (None, 1)
            else:
                period_neighbors = (-1, 1)
                atomic_number_neighbors = (-1, 1)

        if extended_group in P_COLUMNS:
            # Helium
            if period == 1:
                period_neighbors = (None, 1)
                atomic_number_neighbors = (None, None)
            # Boron
            elif extended_group == 27 and period == 2:
                period_neighbors = (None, 1)
                atomic_number_neighbors = (-1, 1)
            # Nihonium
            elif extended_group == 27 and period == 7:
                period_neighbors = (-1, None)
                atomic_number_neighbors = (-1, 1)
            # Neon
            elif extended_group == 32 and period == 2:
                period_neighbors = (None, 1)
                atomic_number_neighbors = (-1, None)
            # Oganesson
            elif extended_group == 32 and period == 7:
                period_neighbors = (-1, None)
                atomic_number_neighbors = (-1, None)
            elif extended_group == 32:
                period_neighbors = (-1, 1)
                atomic_number_neighbors = (-1, None)
            else:
                period_neighbors = (-1, 1)
                atomic_number_neighbors = (-1, 1)

        if extended_group in D_COLUMNS:
            # Scandium
            if extended_group == 17 and period == 4:
                period_neighbors = (None, 1)
                atomic_number_neighbors = (-1, 1)
            # Lawrencium
            elif extended_group == 17 and period == 7:
                period_neighbors = (-1, None)
                atomic_number_neighbors = (-1, 1)
            # Zinc
            elif extended_group == 26 and period == 4:
                period_neighbors = (None, 1)
                atomic_number_neighbors = (-1, 1)
            # Copernicium
            elif extended_group == 26 and period == 7:
                period_neighbors = (-1, None)
                atomic_number_neighbors = (-1, 1)
            else:
                period_neighbors = (-1, 1)
                atomic_number_neighbors = (-1, 1)

        if extended_group in F_COLUMNS:
            # Lanthanum
            if extended_group == 3 and period == 6:
                period_neighbors = (None, 1)
                atomic_number_neighbors = (-1, 1)
            # Actinium
            elif extended_group == 3 and period == 7:
                period_neighbors = (-1, None)
                atomic_number_neighbors = (-1, 1)
            # Zinc
            elif extended_group == 16 and period == 6:
                period_neighbors = (None, 1)
                atomic_number_neighbors = (-1, 1)
            # Copernicium
            elif extended_group == 16 and period == 7:
                period_neighbors = (-1, None)
                atomic_number_neighbors = (-1, 1)
            else:
                period_neighbors = (-1, 1)
                atomic_number_neighbors = (-1, 1)

        for neighbor_period in period_neighbors:
            current_period = period

            if neighbor_period is not None:
                current_period += neighbor_period

                matching_indexes = df[
                    (df["period"] == current_period)
                    & (df["extended_group"] == extended_group)
                ].index.values
                if len(matching_indexes) != 0:

                    edge_index.append((irow, matching_indexes[0]))

        for neighbor_atomic_number in atomic_number_neighbors:
            current_atomic_number = atomic_number

            if neighbor_atomic_number is not None:
                current_atomic_number += neighbor_atomic_number
                matching_indexes = df[
                    df["atomic_number"] == current_atomic_number
                ].index.values
                if len(matching_indexes) != 0:
                    edge_index.append((irow, matching_indexes[0]))
    return edge_index


def covalent_cutoff_map(tol: float = 0.1):
    """
    Generate a dictionary of covalent cutoff values for all element pairs, accounting for a tolerance value.

    Parameters
    ----------
    tol : float, optional
        Tolerance applied to the covalent radii sum (default is 0.1).

    Returns
    -------
    dict
        Dictionary where keys are element pairs, and values are covalent cutoffs.

    Example
    -------
    >>> covalent_cutoff_map(tol=0.05)
    {('H', 'H'): 0.66, ('H', 'He'): 0.98, ...}
    """
    cutoff_dict = {}
    element_combs = list(combinations_with_replacement(atomic_symbols[1:], r=2))

    covalent_map = {
        element: covalent_radii
        for element, covalent_radii in zip(atomic_symbols[1:], covalent_radii[1:])
    }
    for element_comb in element_combs:
        element_1 = element_comb[0]
        element_2 = element_comb[1]
        covalent_radii_1 = covalent_map[element_1]
        covalent_radii_2 = covalent_map[element_2]
        cutoff = (covalent_radii_1 + covalent_radii_2) * (1 + tol)
        cutoff_dict.update({element_comb: cutoff})

    return cutoff_dict


# if __name__ == '__main__':
#     df=get_element_properties(type='interim',columns=['symbol'], output_format='pandas')
#     print(df.head())

#     # df.drop(columns=['Unnamed: 0'], inplace=True)
#     # save_parquet(df, os.path.join(config.data_dir,'interim_periodic_table_values.parquet'), from_pandas_args={'preserve_index':False})
