class Node:
    def __init__(self, id, node_type, attributes=None):
        self.id = id
        self.node_type = node_type
        self.attributes = attributes or {}
        self.relationships = {}

class Relationship:
    def __init__(self, rel_type, target_node, attributes=None):
        self.rel_type = rel_type
        self.target_node = target_node
        self.attributes = attributes or {}


class Heterograph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, id, node_type, attributes=None):
        if id not in self.nodes:
            self.nodes[id] = Node(id, node_type, attributes)
        return self.nodes[id]

    def add_relationship(self, source_id, target_id, rel_type, attributes=None):
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the graph")
        
        source_node = self.nodes[source_id]
        target_node = self.nodes[target_id]
        
        if rel_type not in source_node.relationships:
            source_node.relationships[rel_type] = []
        
        source_node.relationships[rel_type].append(Relationship(rel_type, target_node, attributes))

    def get_neighbors(self, node_id, rel_type=None):
        if node_id not in self.nodes:
            raise ValueError("Node does not exist in the graph")
        
        node = self.nodes[node_id]
        neighbors = []
        
        if rel_type:
            return [rel.target_node for rel in node.relationships.get(rel_type, [])]
        else:
            return [rel.target_node for rels in node.relationships.values() for rel in rels]

    def get_node_by_type(self, node_type):
        return [node for node in self.nodes.values() if node.node_type == node_type]