import numpy as np

def face_sides_bin_encoder(node_values):
    """Ceates bins for the number of sides on a face

    Parameters
    ----------
    node_values : _type_
        The number of nodes(faces)

    Returns
    -------
    np.ndarray
        The encoded vector
    """
    n_nodes = len(node_values)
    encoded_vec = np.zeros(shape = (n_nodes, 8))

    for i_node,node_value in enumerate(node_values):
        if node_value <= 8:
            encoded_vec[i_node,node_value-3]  = 1
        else:
            encoded_vec[i_node,-1]  = 1
    return encoded_vec


def gaussian_continuous_bin_encoder(values, min_val:float=0, max_val:float=40, sigma:float= 2):
    """Creates bins graph continuous features by gaussian method

    Parameters
    ----------
    values : : float
        The continuous value to bin
    min_val : float, optional
        The minimum value, by default 0
    max_val : float, optional
        The max value, by default 40
    sigma : float, optional
        The standard dev for the binning, by default 2

    Returns
    -------
    np.ndarray
        The binned feature
    """
    filter = np.arange(min_val, max_val + sigma, step=sigma)
    values = np.array(values)
    encoded_vec = np.exp(-(values - filter)**2 / sigma**2)
    return encoded_vec