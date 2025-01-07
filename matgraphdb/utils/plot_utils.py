import logging
import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from matgraphdb.utils.file_utils import load_json

logger = logging.getLogger(__name__)

def plot_graph(session):
    """
    Plot a graph based on the relationships between nodes in the session.

    Parameters
    ----------
    session : object
        The session object used to query the database.

    Returns
    -------
    None

    Example
    -------
    >>> plot_graph(session)
    """

    logger.info("Fetching data from session.")
    # Fetch data
    data = session.run("""
    MATCH (a)-[r:KNOWS]->(b)
    RETURN a.name, b.name, r.weight
    """).data()

    logger.info("Creating NetworkX graph.")
    # Create a NetworkX graph
    G = nx.Graph()
    for row in data:
        G.add_edge(row['a.name'], row['b.name'], weight=row['r.weight'])

    logger.info("Drawing graph.")
    # Draw graph
    edge_colors = [d['weight'] for _, _, d in G.edges(data=True)]
    nx.draw(G, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
    plt.show()


def plot_node_and_connections(session, center_name, center_class, edge_name, edge_type, 
                              use_weights_for_thickness=True, filename=None, 
                              figsize=(12, 12), node_size=800, node_spacing=3, font_size=12):
    """
    Plot a graph with nodes and connections based on the provided parameters.

    Parameters
    ----------
    session : object
        The Neo4j session object.
    center_name : str
        The name of the center node.
    center_class : str
        The class of the center node.
    edge_name : str
        The name of the edge connecting the center node to surrounding nodes.
    edge_type : str
        The type of the edge connecting the center node to surrounding nodes.
    use_weights_for_thickness : bool, optional
        Whether to use edge weights for edge thickness (default is True).
    filename : str, optional
        The filename to save the plot as an image. If not provided, the plot will be displayed (default is None).
    figsize : tuple, optional
        The size of the figure in inches (default is (12, 12)).
    node_size : int, optional
        The size of the nodes (default is 800).
    node_spacing : float, optional
        The spacing between nodes (default is 3).
    font_size : int, optional
        The font size of the node labels (default is 12).

    Returns
    -------
    None

    Example
    -------
    >>> plot_node_and_connections(session, 'NodeA', 'ClassA', 'KNOWS', 'Friend', True)
    """

    logger.info("Building Cypher query.")
    execute_statement = 'MATCH (center:' + f'{center_class} ' + "{name: " + f"'{center_name}'" + "})"
    execute_statement += f"-[r:{edge_name} {{type:'{edge_type}'}}]-(surrounding)\n"
    execute_statement += "WHERE NOT center = surrounding\n"
    execute_statement += "RETURN center.name, surrounding.name, r.weight"

    logger.info("Fetching data from session.")
    data = session.run(execute_statement).data()

    logger.info("Creating NetworkX graph.")
    G = nx.Graph()
    for row in data:
        G.add_edge(row['center.name'], row['surrounding.name'], weight=row['r.weight'])

    logger.info("Drawing graph.")
    edge_colors = [d['weight'] for _, _, d in G.edges(data=True)]
    
    if use_weights_for_thickness:
        edge_widths = [d['weight'] for _, _, d in G.edges(data=True)]
    else:
        edge_widths = None

    min_width = min(edge_widths)
    max_width = max(edge_widths)

    normalized_edge_widths = [1 + 9 * ((w - min_width) / (max_width - min_width)) for w in edge_widths]

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=node_spacing)

    nx.draw(G, pos=pos, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Blues, 
            width=normalized_edge_widths, node_size=node_size, font_size=font_size, style='dashed')

    if filename:
        logger.info(f"Saving plot to file {filename}.")
        plt.savefig(filename)
    else:
        logger.info("Displaying plot.")
        plt.show()


def plot_node_and_connections_test(session, center_name, center_class, edge_name, edge_type, 
                                   use_weights_for_thickness=True, filename=None):
    """
    Plot a graph of nodes and connections based on the specified parameters.

    Parameters
    ----------
    session : object
        The Neo4j session object.
    center_name : str
        The name of the center node.
    center_class : str
        The class of the center node.
    edge_name : str
        The name of the edge.
    edge_type : str
        The type of the edge.
    use_weights_for_thickness : bool, optional
        Whether to use edge weights for edge thickness (default is True).
    filename : str, optional
        The filename to save the plot as. If not provided, the plot will be displayed (default is None).

    Returns
    -------
    None

    Example
    -------
    >>> plot_node_and_connections_test(session, 'NodeA', 'ClassA', 'KNOWS', 'Friend', True)
    """

    logger.info("Building Cypher query.")
    execute_statement = f'MATCH (center:{center_class} {{name: "{center_name}"}})'
    execute_statement += f'-[r:{edge_name} {{type:"{edge_type}"}}]-(surrounding)\n'
    execute_statement += 'WHERE NOT center = surrounding\n'
    execute_statement += 'RETURN center.name, surrounding.name, r.weight'

    logger.info("Fetching data from session.")
    data = session.run(execute_statement).data()

    logger.info("Creating NetworkX graph.")
    G = nx.Graph()
    for row in data:
        G.add_edge(row['center.name'], row['surrounding.name'], weight=row['r.weight'])

    logger.info("Drawing graph.")
    edge_colors = [d['weight'] for _, _, d in G.edges(data=True)]
    
    if use_weights_for_thickness:
        edge_widths = [d['weight'] for _, _, d in G.edges(data=True)]
    else:
        edge_widths = None

    min_width = min(edge_widths)
    max_width = max(edge_widths)
    normalized_edge_widths = [1 + 9 * ((w - min_width) / (max_width - min_width)) for w in edge_widths]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5)

    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Blues, 
            width=normalized_edge_widths, node_size=800, font_size=12)

    if filename:
        logger.info(f"Saving plot to file {filename}.")
        plt.savefig(filename)
    else:
        logger.info("Displaying plot.")
        plt.show()




def plot_graph_from_neo4j_session(session):
    """
    Plot a graph based on the relationships between nodes in the session.

    Parameters:
    - session: The session object used to query the database.

    Returns:
    None
    """

    # Fetch data
    data = session.run("""
    MATCH (a)-[r:KNOWS]->(b)
    RETURN a.name, b.name, r.weight
    """).data()

    # Create a NetworkX graph
    G = nx.Graph()
    for row in data:
        G.add_edge(row['a.name'], row['b.name'], weight=row['r.weight'])

    # Draw graph
    edge_colors = [d['weight'] for _, _, d in G.edges(data=True)]
    nx.draw(G, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
    plt.show()

def plot_node_and_connections_from_neo4j_session(session, center_name, center_class, edge_name, edge_type, use_weights_for_thickness=True,
                              filename=None, figsize=(12, 12), node_size=800, node_spacing=3, font_size=12):
    """
    Plot a graph with nodes and connections based on the provided parameters from a neoj4j session.

    Args:
        session: The Neo4j session object.
        center_name: The name of the center node.
        center_class: The class of the center node.
        edge_name: The name of the edge connecting the center node to the surrounding nodes.
        edge_type: The type of the edge connecting the center node to the surrounding nodes.
        use_weights_for_thickness: A boolean indicating whether to use edge weights for edge thickness. Default is True.
        filename: The filename to save the plot as an image. If not provided, the plot will be displayed. Default is None.
        figsize: The size of the figure (width, height) in inches. Default is (12, 12).
        node_size: The size of the nodes. Default is 800.
        node_spacing: The spacing between nodes. Default is 3.
        font_size: The font size of the node labels. Default is 12.
    """

    execute_statement = 'MATCH (center:' + f'{center_class} ' + "{name: " + f"'{center_name}'" + "})"
    execute_statement += f"-[r:{edge_name} {{type:'{edge_type}'}}]-(surrounding)\n"
    execute_statement += "WHERE NOT center = surrounding\n"  # Exclude self-connections
    execute_statement += "RETURN center.name, surrounding.name, r.weight"

    data = session.run(execute_statement).data()
    # Create a NetworkX graph
    G = nx.Graph()
    for row in data:
        G.add_edge(row['center.name'], row['surrounding.name'], weight=row['r.weight'])

    # Draw graph
    edge_colors = [d['weight'] for _, _, d in G.edges(data=True)]

    if use_weights_for_thickness:
        edge_widths = [d['weight'] for _, _, d in G.edges(data=True)]
    else:
        edge_widths = None

    min_width = min(edge_widths)
    max_width = max(edge_widths)

    normalized_edge_widths = [1 + 9 * ((w - min_width) / (max_width - min_width)) for w in edge_widths]

    plt.figure(figsize=figsize)

    # Use spring layout
    pos = nx.spring_layout(G, k=node_spacing)  # k adjusts the optimal distance between nodes. Play around with it.

    edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels,verticalalignment='top')

    nx.draw(G, pos=pos, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=normalized_edge_widths,
            node_size=node_size, font_size=font_size, style='dashed')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()




# Function to plot bond order matrices
def plot_bond_order_matrix(matrix, title, filename, atomic_symbols, save_dir, use_log_scale=False):
    """
    Plot and save a bond order matrix.
    
    Args:
        matrix (np.ndarray): The bond order matrix to be plotted.
        title (str): Title of the plot.
        filename (str): Name of the file to save the plot.
        atomic_symbols (list): List of atomic symbols for labeling the axes.
        save_dir (str): Directory to save the figure.
        use_log_scale (bool): Whether to use logarithmic scaling for the plot.
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    
    if use_log_scale:
        im = ax.imshow(matrix, cmap='hot', interpolation='nearest', norm=mcolors.LogNorm())
    else:
        im = ax.imshow(matrix, cmap='hot', interpolation='nearest')
    
    fig.colorbar(im)
    ax.set_title(title)
    ax.set_xlabel('Atomic Symbols')
    ax.set_ylabel('Atomic Symbols')
    ax.set_xticks(np.arange(len(atomic_symbols)))
    ax.set_yticks(np.arange(len(atomic_symbols)))
    ax.set_xticklabels(atomic_symbols, rotation=90)
    ax.set_yticklabels(atomic_symbols)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    
    plt.savefig(os.path.join(save_dir, filename))  # Save the figure
    plt.close()

# Main function to handle all bond order plotting
def plot_all_bond_order_matrices(json_file, save_dir):
    """
    Load bond order data and plot various matrices (avg, std, occurrences, log occurrences).
    
    Args:
        json_file (str): Path to the JSON file containing bond order data.
        save_dir (str): Directory to save the plots.
    """
    from matgraphdb.utils.chem_utils.periodic import atomic_symbols
    atomic_symbols=atomic_symbols[1:]

    os.makedirs(save_dir,exist_ok=True)
    
    data = load_json(json_file)
    
    bond_orders_avg = np.array(data['bond_orders_avg'])
    bond_orders_std = np.array(data['bond_orders_std'])
    n_bond_orders = np.array(data['n_bond_orders'])
    
    # Plot Bond Orders Average
    plot_bond_order_matrix(bond_orders_avg, 'Bond Orders Average', 'bond_orders_avg.png', atomic_symbols, save_dir)
    
    # Plot Bond Orders Standard Deviation
    plot_bond_order_matrix(bond_orders_std, 'Bond Orders Standard Deviation', 'bond_orders_std.png', atomic_symbols, save_dir)
    
    # Plot Bond Orders Occurrences
    plot_bond_order_matrix(n_bond_orders, 'Bond Orders Occurrences', 'bond_orders_occurrences.png', atomic_symbols, save_dir)
    
    # Handle log scale for occurrences
    min_nonzero = np.min(n_bond_orders[n_bond_orders > 0])
    n_bond_orders_log = np.where(n_bond_orders > 0, n_bond_orders, min_nonzero)
    
    # Plot Bond Orders Occurrences with Logarithmic Scale
    plot_bond_order_matrix(n_bond_orders_log, 'Bond Orders Occurrences (Log Scale)', 'bond_orders_occurrences_log.png', atomic_symbols, save_dir, use_log_scale=True)


def plot_points(plotter, points, color='green', size=15):
    """
    Plot a set of points in a 3D plotter.

    Parameters:
    - plotter: pyvista.Plotter
        The 3D plotter object to add the points to.
    - points: pyvista.PolyData
        The points to be plotted.
    - color: str, optional
        The color of the points. Default is 'green'.
    - size: int, optional
        The size of the points. Default is 15.

    Returns:
    None
    """
    import pyvista
    plotter.add_mesh(points, color=color, point_size=size, render_points_as_spheres=True)


def plot_adjacency(plotter, adj_matrix, points):
    """
    Plot the adjacency of points using a given plotter.

    Parameters:
    - plotter: The plotter object used for visualization.
    - adj_matrix: The adjacency matrix representing the connections between points.
    - points: The list of points to be plotted.

    Returns:
    None
    """

    import pyvista
    lines = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if adj_matrix[i, j]:
                plotter.add_lines(np.array([points[i], points[j]]))


def plot_similarity_matrix(similarity_matrix, labels, add_values=True, filename=None):
    """
    Plots a similarity matrix.

    Parameters:
    similarity_matrix (numpy.ndarray): The similarity matrix to be plotted.
    labels (list): The labels for the rows and columns of the matrix.
    add_values (bool, optional): Whether to add the values of the matrix as text. Defaults to True.
    filename (str, optional): The filename to save the plot as an image. If not provided, the plot will be displayed.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(similarity_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)

    fig.suptitle('Similarity matrix')

    if add_values:
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                ax.text(j, i, format(similarity_matrix[i, j], ".2f"), 
                        ha="center", va="center", 
                        color="w" if np.abs(similarity_matrix[i, j]) > 0.5 else "black")
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_training_curves(epochs, train_loss, val_loss=None, test_loss=None, loss_label='MSE', filename=None, log_scale=False):
    """
    Plots the training curves for a given set of epochs and loss values.

    Parameters:
    - epochs (list): A list of integers representing the epochs.
    - train_loss (list): A list of floats representing the training loss values.
    - val_loss (list, optional): A list of floats representing the validation loss values. Default is None.
    - test_loss (list, optional): A list of floats representing the test loss values. Default is None.
    - loss_label (str, optional): The label for the loss values. Default is 'MSE'.
    - filename (str, optional): The filename to save the plot. Default is None.
    - log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis. Default is False.

    Returns:
    - None

    """
    
    fig, ax = plt.subplots(1)

    ax.plot(epochs, train_loss, label='train')
    if val_loss:
        ax.plot(epochs, val_loss, label='val')
    if test_loss:
        ax.plot(epochs, test_loss, label='test')

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('epochs')
    ax.set_ylabel(loss_label) 
    ax.legend()

    fig.suptitle("Training Curves")

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_dataframe_distributions(df, save_dir=None, col_names=None):
    """Plots the distributions of the columns in a dataframe."""

    if col_names is None:
        numeric_columns = df.select_dtypes(include=['number']).columns
    else:
        numeric_columns = col_names

    num_cols = len(numeric_columns)
    
    if num_cols == 0:
        print("No numeric columns to plot.")
        return

    os.makedirs(save_dir,exist_ok=True)
    
    for i, col in enumerate(numeric_columns):
        property_name=col.split(':')[0]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Create a subplot for each type of plot

        # Histogram
        sns.histplot(df[col], kde=False, ax=axes[0])
        axes[0].set_title(f'Histogram of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')

        # Density Plot
        sns.kdeplot(df[col], ax=axes[1], fill=True)
        axes[1].set_title(f'Density Plot of {col}')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Density')

        # Box Plot
        sns.boxplot(x=df[col], ax=axes[2])
        axes[2].set_title(f'Box Plot of {col}')
        axes[2].set_xlabel(col)
        axes[2].set_ylabel('Value')

        # Violin Plot
        sns.violinplot(x=df[col], ax=axes[3])
        axes[3].set_title(f'Violin Plot of {col}')
        axes[3].set_xlabel(col)
        axes[3].set_ylabel('Value')

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir,f'{property_name}.png'))
        else:
            plt.show()