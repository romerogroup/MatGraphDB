import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FeaturePropagation


def feature_propagation(data=None, x=None, edge_index=None, **kwargs):
    """
    Perform feature propagation on a graph to impute missing values.

    This function applies the FeaturePropagation transformation to a given 
    graph's node features, filling in missing values based on the graph 
    structure and available features. The function accepts either a PyTorch 
    Geometric `Data` object, or separate `x` (node features) and `edge_index` 
    (graph edges) arrays.

    Parameters:
    -----------
    data : torch_geometric.data.Data, optional
        A PyTorch Geometric Data object containing the node features `x` and 
        the adjacency information `edge_index`. If provided, `x` and 
        `edge_index` are extracted from this object.
    
    x : torch.Tensor, optional
        A tensor containing the node features. Must be provided if `data` 
        is not given.
    
    edge_index : torch.Tensor, optional
        A tensor defining the edge connections in the graph, represented 
        as a list of source and target node indices. Must be provided if 
        `data` is not given.
    
    **kwargs : dict
        Additional keyword arguments passed to the `FeaturePropagation` 
        class, which controls the behavior of the feature propagation 
        algorithm (e.g., the propagation method, number of iterations, etc.).

    Returns:
    --------
    numpy.ndarray
        The imputed node features as a NumPy array, where missing values 
        have been filled through feature propagation.
    
    Raises:
    -------
    ValueError
        If neither `data` nor both `x` and `edge_index` are provided.

    Example:
    --------
    >>> imputed_features = feature_propagation(x=my_node_features, 
                                               edge_index=my_edge_index)
    """
    if data is None and x is None and edge_index is None:
        raise ValueError("Either data or x and edge_index must be provided")
    if data is None:
        data = Data(x=x, edge_index=edge_index)
    else:
        x=data.x
    transform = FeaturePropagation(missing_mask=torch.isnan(x), **kwargs)
    homo_graph_transformed = transform(data)
    return homo_graph_transformed.x.numpy()
