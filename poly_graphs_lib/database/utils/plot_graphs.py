import os
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import matplotlib as mpl

from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME
from poly_graphs_lib.utils import PROJECT_DIR

def plot_graph(session):
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

def plot_node_and_connections(session,center_name,center_class,edge_name,edge_type,use_weights_for_thickness=True,
                              filename=None,
                              figsize=(12,12),
                              node_size=800,
                              node_spacing=3,
                              font_size=12
                              ):


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


    nx.draw(G,pos=pos, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Blues,width=normalized_edge_widths,
            node_size=node_size, font_size=font_size,style='dashed')
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_node_and_connections_test(session,center_name,center_class,edge_name,edge_type,use_weights_for_thickness=True,filename=None):
    execute_statement = f'MATCH (center:{center_class} {{name: "{center_name}"}})'
    execute_statement += f'-[r:{edge_name} {{type:"{edge_type}"}}]-(surrounding)\n'
    execute_statement += 'WHERE NOT center = surrounding\n'
    execute_statement += 'RETURN center.name, surrounding.name, r.weight'

    data = session.run(execute_statement).data()

    G = nx.Graph()
    for row in data:
        G.add_edge(row['center.name'], row['surrounding.name'], weight=row['r.weight'])

    edge_colors = [d['weight'] for _, _, d in G.edges(data=True)]

    if use_weights_for_thickness:
        edge_widths = [d['weight'] for _, _, d in G.edges(data=True)]
    else:
        edge_widths = None

    min_width = min(edge_widths)
    max_width = max(edge_widths)
    normalized_edge_widths = [1 + 9 * ((w - min_width) / (max_width - min_width)) for w in edge_widths]

    # Set figure size
    plt.figure(figsize=(12, 12))

    # Use spring layout
    pos = nx.spring_layout(G, k=0.5)  # k adjusts the optimal distance between nodes. Play around with it.

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=normalized_edge_widths, node_size=800, font_size=12)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def main():

    # This statement Connects to the database server
    connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
    # To read and write to the data base you must open a session
    session = connection.session(database=DB_NAME)

    reports_dir=os.path.join(PROJECT_DIR,'reports',DB_NAME)



    center_class='chemenv'
    save_dir=os.path.join(reports_dir,center_class)
    os.makedirs(save_dir,exist_ok=True)
    plot_node_and_connections(session,center_name='O_6',
                              center_class=center_class,
                              edge_name='CONNECTS', 
                              edge_type=f'chemenv-{center_class}',
                              node_spacing=2,
                              filename=os.path.join(save_dir,'C'))
    # plot_node_and_connections(session,center_name='O',
    #                           center_class=center_class,
    #                           edge_name='CONNECTS', 
    #                           edge_type=f'chemenv-{center_class}',
    #                           node_spacing=5,
    #                           filename=os.path.join(save_dir,'O'))
    # plot_node_and_connections(session,center_name='Ti',
    #                           center_class=center_class,
    #                           edge_name='CONNECTS', 
    #                           edge_type=f'chemenv-{center_class}',
    #                           node_spacing=2,
    #                           filename=os.path.join(save_dir,'Ti'))


    # center_class='magnetic_states'
    # save_dir=os.path.join(reports_dir,center_class)
    # os.makedirs(save_dir,exist_ok=True)
    # plot_node_and_connections(session,center_name='FM',
    #                           center_class=center_class,
    #                           edge_name='CONNECTS', 
    #                           edge_type=f'chemenv-{center_class}',
    #                           node_spacing=5,
    #                           filename=os.path.join(save_dir,'FM'))
    # plot_node_and_connections(session,center_name='AFM',
    #                           center_class=center_class,
    #                           edge_name='CONNECTS', 
    #                           edge_type=f'chemenv-{center_class}',
    #                           node_spacing=2,
    #                           filename=os.path.join(save_dir,'AFM'))
    # plot_node_and_connections(session,center_name='FiM',
    #                           center_class=center_class,
    #                           edge_name='CONNECTS', 
    #                           edge_type=f'chemenv-{center_class}',
    #                           node_spacing=3,
    #                           filename=os.path.join(save_dir,'FiM'))
    # plot_node_and_connections(session,center_name='NM',
    #                           center_class=center_class,
    #                           edge_name='CONNECTS', 
    #                           edge_type=f'chemenv-{center_class}',
    #                           node_spacing=10,
    #                           filename=os.path.join(save_dir,'NM'))


    # center_class='chemenv'
    # save_dir=os.path.join(reports_dir,center_class)
    # os.makedirs(save_dir,exist_ok=True)
    # plot_node_and_connections(session,center_name='T_4',
    #                           center_class=center_class,
    #                           edge_name='CONNECTS', 
    #                           edge_type=f'chemenv-{center_class}',
    #                           node_spacing=10,
    #                           filename=os.path.join(save_dir,'T_4'))
    # plot_node_and_connections(session,center_name='C_12',
    #                           center_class=center_class,
    #                           edge_name='CONNECTS', 
    #                           edge_type=f'chemenv-{center_class}',
    #                           node_spacing=10,
    #                           filename=os.path.join(save_dir,'C_12'))
    # plot_node_and_connections(session,center_name='O_6',
    #                           center_class=center_class,
    #                           edge_name='CONNECTS', 
    #                           edge_type=f'chemenv-{center_class}',
    #                           node_spacing=10,
    #                           filename=os.path.join(save_dir,'O_6'))


    session.close()
    connection.close()

if __name__ == "__main__":
    main()