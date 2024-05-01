import networkx as nx
import matplotlib.pyplot as plt

# Define the directed graph
megaman_directed_graph = {
    'A': {'B': 420, 'C': 725},
    'B': {'C': 300, 'D': 1300},
    'C': {'D': 900, 'E': 1300},
    'D': {'E': 360},
    'E': {'F': 360},
    'F': {'A': 480, 'C': 600, 'D': 480}
}

# Create a directed graph object
G = nx.DiGraph()

# Add nodes and edges to the graph
for node, neighbors in megaman_directed_graph.items():
    G.add_node(node)  # Add node
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)  # Add edge with weight

# Define the shell layout based on node order
node_order = ['A', 'B', 'C', 'D', 'E', 'F']
pos = nx.shell_layout(G, nlist=[node_order])

# Plot the directed graph
plt.figure(figsize=(10, 5))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')  # Draw nodes
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='b')  # Draw edges
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')  # Draw node labels
edge_labels = {(n1, n2): d['weight'] for n1, n2, d in G.edges(data=True)}  # Extract edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')  # Draw edge labels
plt.title('Customized Megaman Graph')
plt.axis('off')  # Turn off axis
plt.show()

