import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Example dataset: vectorized features for each item (node)
features = np.array([
    [0.1, 0.2, 0.3],
    [0.2, 0.1, 0.4],
    [0.4, 0.4, 0.1],
    # Add more vectors for each item
])

# Calculate similarity matrix
similarity = cosine_similarity(features)
print(similarity)
# Create graph
G = nx.Graph()

# Add nodes
for i in range(len(features)):
    G.add_node(i)

# Add edges based on similarity threshold
threshold = 0.5
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        if similarity[i][j] > threshold:
            G.add_edge(i, j, weight=similarity[i][j])
print(G)

# Visualize the graph
pos = nx.spring_layout(G)  # Positions nodes using Fruchterman-Reingold
nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", edge_color="gray")
plt.show()
