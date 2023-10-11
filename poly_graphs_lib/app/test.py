
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output
import dash_cytoscape as cyto
from py2neo import Graph
from neo4j import GraphDatabase, Driver

from poly_graphs_lib.database import PASSWORD,DBMS_NAME,LOCATION,DB_NAME
from poly_graphs_lib.utils import ROOT


app_dir=os.path.join(ROOT,'app')
data_dir=os.path.join(app_dir,'mnt','data')
os.makedirs(data_dir,exist_ok=True)

def create_colorbar(min_val, max_val, cmap='coolwarm'):
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    
    norm = plt.Normalize(min_val, max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.15, 0.3, 0.7, 0.4])
    plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=np.linspace(min_val, max_val, 5))
    
    plt.savefig(os.path.join(data_dir,'colorbar.png'))

# Create the colorbar
min_property = 0  # Replace with your minimum property value
max_property = 100  # Replace with your maximum property value
create_colorbar(min_property, max_property)

# This statement Connects to the database server
connection = GraphDatabase.driver(LOCATION, auth=(DBMS_NAME, PASSWORD))
# To read and write to the data base you must open a session
neo4j_graph = connection.session(database=DB_NAME)

# Fetch data from Neo4j (Replace Cypher query as needed)
cypher_query = """
MATCH (n:Structure)
RETURN n.name AS source, n.name AS name, n.band_gap AS property
"""

# data = neo4j_graph.run(cypher_query).to_data_frame()
result = neo4j_graph.run(cypher_query)
data = pd.DataFrame([record.values() for record in result], columns=result.keys())

edge_df = pd.DataFrame(data)

# Create Dash app
app = dash.Dash(__name__)


import matplotlib.pyplot as plt

def map_property_to_color(value, min_value, max_value):
    cmap = plt.get_cmap('coolwarm')
    norm_value = (value - min_value) / (max_value - min_value)
    rgba = cmap(norm_value)
    return f'rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})'

min_property = data['property'].min()
max_property = data['property'].max()

stylesheet = [
    {
        'selector': 'node',
        'style': {
            'background-color': f'mapData(property, {min_property}, {max_property}, blue, red)',
            'label': 'data(label)'
        }
    }
]
# Initialize Cytoscape elements
cyto_elements = [
        {'data': {'id': src, 'label': src,'property':prop}} for src,prop in zip(data['source'].unique(), data['property'])
    ] 
# + [
#     {'data': {'id': dest, 'label': dest}} for dest in edge_df['destination'].unique()
# ] + [
#     {'data': {'source': src, 'target': dest}} for src, dest in zip(edge_df['source'], edge_df['destination'])
# ]

# App layout
app.layout = html.Div([
    html.H1("Neo4j Graph Visualization of Structure Nodes"),
    dcc.RangeSlider(
        id='property-slider',
        min=data['property'].min(),
        max=data['property'].max(),
        value=[data['property'].min(), data['property'].max()],
        # marks={str(prop): str(prop) for prop in data['property'].unique()},
        step=None
    ),
    cyto.Cytoscape(
        id='cytoscape',
        elements=cyto_elements,
        stylesheet=stylesheet,
        layout={'name': 'circle'},
        style={'width': '100%', 'height': '400px'}
    ),
    html.Img(id='colorbar', src=os.path.join(data_dir,'colorbar.png')),
    html.Div(id='node-data', style={'margin-top': '20px'})
])


# Callback to update graph elements based on property slider
@app.callback(
    Output('cytoscape', 'elements'),
    Input('property-slider', 'value')
)
def update_elements(selected_range):
    lower_bound, upper_bound = selected_range
    filtered_data = data[(data['property'] >= lower_bound) & (data['property'] <= upper_bound)]
    updated_elements = [
        {'data': {'id': src, 'label': src,'property':prop}} for src,prop in zip(filtered_data['source'].unique(), filtered_data['property'])
    ] 
    # + [
    #     {'data': {'id': dest, 'label': dest}} for dest in filtered_data['destination'].unique()
    # ] + [
    #     {'data': {'source': src, 'target': dest}} for src, dest in zip(filtered_data['source'], filtered_data['destination'])
    # ]

    return updated_elements

# Callback to display node data
@app.callback(
    Output('node-data', 'children'),
    Input('cytoscape', 'tapNodeData')
)
def display_tap_node_data(data):
    if data is None:
        return "Click on a node to see its data."
    return f"Node {data['id']} has property: {data.get('property', 'N/A')}"




if __name__ == '__main__':
    app.run_server(debug=True)