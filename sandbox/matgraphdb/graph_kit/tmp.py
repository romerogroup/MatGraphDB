# import pyarrow as pa

# schema = pa.schema([
#     ('name', pa.string()),
#     ('age', pa.int32()),
#     ('city', pa.string())
# ])

# print(schema)

# metadata = {
#     'created_by': 'Your Name',
#     'creation_date': '2024-09-05',
#     'description': 'Schema for user data'
# }

# new_schema = schema.with_metadata(metadata)
# print('-'*100)
# print(new_schema)


import os
from matgraphdb.graph_kit.nodes import NodeManager

if __name__ == "__main__":
    node_dir = os.path.join('data','raw','nodes')

    manager=NodeManager(node_dir=node_dir)
    manager.load_node('ELEMENT')

    




    # node=ElementNodes(node_dir=node_dir)
    # print(node.get_property_names())