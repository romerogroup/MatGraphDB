import logging

import numpy as np
from scipy.spatial.distance import cdist

logger=logging.getLogger(__name__)

def rot_z(theta):
    """
    Returns a 3x3 rotation matrix for rotating around the z-axis.

    Parameters
    ----------
    theta : float
        The angle of rotation in degrees.

    Returns
    -------
    numpy.ndarray
        The 3x3 rotation matrix for a z-axis rotation.

    Examples
    --------
    >>> rot_z(90)
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])
    """
    logger.info(f"Generating rotation matrix for theta={theta} degrees.")
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    result=np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    logger.info(f"Rotation matrix: {result}")
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def face_sides_bin_encoder(node_values):
    """
    Creates a binary encoded vector representing the number of sides on a face.

    Parameters
    ----------
    node_values : list or array-like
        The number of nodes (faces).

    Returns
    -------
    numpy.ndarray
        A binary encoded vector representing the sides of each face.

    Examples
    --------
    >>> face_sides_bin_encoder([3, 4, 5, 6, 8])
    array([[1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1.]])
    """
    logger.info(f"Encoding node values: {node_values}")
    n_nodes = len(node_values)
    encoded_vec = np.zeros(shape=(n_nodes, 8))

    for i_node, node_value in enumerate(node_values):
        if node_value <= 8:
            encoded_vec[i_node, node_value - 3] = 1
        else:
            encoded_vec[i_node, -1] = 1
    logger.info(f"Face sides encoded vector: {encoded_vec}")
    return


def gaussian_continuous_bin_encoder(values, n_bins=50, min_val=0, max_val=40, sigma=2):
    """
    Encodes continuous values into bins using a Gaussian method.

    Parameters
    ----------
    values : list or array-like
        The continuous values to bin.
    n_bins : int, optional
        Number of bins, by default 50.
    min_val : float, optional
        Minimum value, by default 0.
    max_val : float, optional
        Maximum value, by default 40.
    sigma : float, optional
        Standard deviation for Gaussian binning, by default 2.

    Returns
    -------
    numpy.ndarray
        The encoded binned values.

    Examples
    --------
    >>> gaussian_continuous_bin_encoder([10, 20, 30])
    array([...])  # Gaussian-encoded output.
    """
    logger.info(f"Gaussian bin encoding for values={values}, n_bins={n_bins}, min_val={min_val}, max_val={max_val}, sigma={sigma}")
    filter = np.linspace(min_val, max_val, n_bins)
    values = np.array(values)
    encoded_vec = np.exp(-(values - filter)**2 / sigma**2)
    logger.info(f"Gaussian encoded vector: {encoded_vec}")
    return encoded_vec


def cosine_similarity(a, b):
    """
    Computes the cosine similarity between two vectors.

    Parameters
    ----------
    a : array-like
        The first vector.
    b : array-like
        The second vector.

    Returns
    -------
    float
        The cosine similarity between the two vectors.

    Examples
    --------
    >>> cosine_similarity([1, 2, 3], [4, 5, 6])
    0.9746318461970762
    """
    logger.info(f"Calculating cosine similarity between vectors a={a} and b={b}")
    dot_product = np.dot(a, b)
    norm_A = np.linalg.norm(a)
    norm_B = np.linalg.norm(b)
    result=dot_product / (norm_A * norm_B)
    logger.info(f"Cosine similarity: {result}")
    return result


def distance_similarity(x, y):
    """
    Calculates the distance similarity between two vectors.

    Parameters
    ----------
    x : numpy.ndarray
        The first vector.
    y : numpy.ndarray
        The second vector.

    Returns
    -------
    float
        The distance similarity between x and y.

    Examples
    --------
    >>> distance_similarity(np.array([1, 2, 3]), np.array([4, 5, 6]))
    0.126
    """
    logger.info(f"Calculating distance similarity between vectors x={x} and y={y}")
    result=np.linalg.norm(x / np.linalg.norm(x) - y / np.linalg.norm(y))
    logger.info(f"Distance similarity: {result}")
    return result



def similarity_score(point_set_1, point_set_2, loss=None, max_iter=100, alpha=1, threshold_plan=0):
    """
    Calculates the similarity score between two point sets using Gromov-Wasserstein distance.

    Parameters
    ----------
    point_set_1 : numpy.ndarray
        The first set of points.
    point_set_2 : numpy.ndarray
        The second set of points.
    loss : function, optional
        Loss function for Gromov-Wasserstein distance. Defaults to absolute difference.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    alpha : float, optional
        Regularization parameter, by default 1.
    threshold_plan : float, optional
        Stopping criterion threshold, by default 0.

    Returns
    -------
    float
        Estimated Gromov-Wasserstein distance.
    float
        Standard deviation of Gromov-Wasserstein distance.

    Examples
    --------
    >>> similarity_score(np.random.rand(5, 2), np.random.rand(5, 2))
    (0.123, 0.005)
    """
    logger.info(f"Calculating similarity score between point_set_1 and point_set_2 with alpha={alpha} and max_iter={max_iter}")
    import ot
    if loss is None:
        def loss(x, y):
            return np.abs(x - y)

    C1 = cdist(point_set_1, point_set_1)
    C2 = cdist(point_set_2, point_set_2)

    C1 /= C1.max()
    C2 /= C2.max()

    n_p = point_set_1.shape[0]
    n_q = point_set_2.shape[0]
    p = ot.unif(n_p)
    q = ot.unif(n_q)
    pgw, plog = ot.gromov.pointwise_gromov_wasserstein(C1, C2, p, q, loss, 
                                                      max_iter=max_iter,
                                                      alpha=alpha,
                                                      threshold_plan=threshold_plan,
                                                      log=True)
    
    logger.info(f"gromov_wasserstein similarity score.", extra=plog)
    return plog['gw_dist_estimated'], plog['gw_dist_std']


def softmax(arr):
    """
    Compute the softmax of an array.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array.

    Returns
    -------
    numpy.ndarray
        The softmax of the input array.

    Examples
    --------
    >>> softmax(np.array([1.0, 2.0, 3.0]))
    array([0.09003057, 0.24472847, 0.66524096])
    """
    logger.info(f"Calculating softmax for array: {arr}")
    exp_arr = np.exp(arr - np.max(arr))  # subtract max for numerical stability
    result=exp_arr / exp_arr.sum()
    logger.info(f"Softmax: {result}")
    return result
