import numpy as np
from scipy.spatial.distance import cdist
import ot

def rot_z(theta):
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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


def gaussian_continuous_bin_encoder(values,n_bins:int=50, min_val:float=0, max_val:float=40, sigma:float= 2):
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

    filter = np.linspace(min_val, max_val,n_bins)
    values = np.array(values)
    encoded_vec = np.exp(-(values - filter)**2 / sigma**2)
    return encoded_vec


def cosine_similarity(x,y):
    return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))

def distance_similarity(x,y):
    return np.linalg.norm(x/np.linalg.norm(x) - y/np.linalg.norm(y))


def similarity_score(point_set_1, point_set_2, loss=None,max_iter=100,alpha=1,threshold_plan =0):
    if loss is None:
        def loss(x, y):
            return np.abs(x - y)

    # https://pythonot.github.io/gen_modules/ot.gromov.html
    C1 = cdist(point_set_1, point_set_1)
    C2 = cdist(point_set_2, point_set_2)

    C1 /= C1.max()
    C2 /= C2.max()

    n_p=point_set_1.shape[0]
    n_q=point_set_2.shape[0]
    p = ot.unif(n_p)
    q = ot.unif(n_q)
    pgw, plog = ot.gromov.pointwise_gromov_wasserstein(C1, C2, p, q, loss, 
                                                    max_iter=max_iter,
                                                    alpha=alpha,
                                                    threshold_plan=threshold_plan,
                                                        log=True)
    
    return plog['gw_dist_estimated'],plog['gw_dist_std']


def softmax(arr):
    # print(exp_arr.sum())
    exp_arr = np.exp(arr- np.max(arr))  # subtract max for numerical stability
    
    return exp_arr / exp_arr.sum()