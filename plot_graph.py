import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(g, title):
    """
    Visualize the graph with artists' names.

    :param G: nx.Graph(), the graph we want to visualize.
    :param title: the title for our plot.
    """
    # Extract the name labels of the artists
    labels = {node: data['name'] for node, data in g.nodes(data=True)}
    # Use spring layout for better visualization
    pos = nx.spring_layout(g)
    # Plot and draw the graph
    plt.figure(figsize=(20, 16))
    nx.draw(g, pos, labels=labels, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=200, font_size=8, font_weight='bold')
    plt.title(title)
    plt.show()

import networkx as nx
import matplotlib.pyplot as plt

# Retrieve graph
gb = nx.read_graphml("./graphs/gB")
# Plot
plot_graph(gb, "Taylor Swift BFS graph")
