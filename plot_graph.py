import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(g, file_path):
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
    # Store the plot in png
    plt.savefig(file_path, format='png', bbox_inches='tight')
    #plt.show()

import networkx as nx
import matplotlib.pyplot as plt

# Retrieve graph
gb = nx.read_graphml("./graphs/gB")
gbp = nx.read_graphml("./graphs/gBp")
gd = nx.read_graphml("./graphs/gD")
gdp = nx.read_graphml("./graphs/gDp")
gbp_prunned = nx.read_graphml("./graphs/gBp_prunned")
gdp_prunned = nx.read_graphml("./graphs/gDp_prunned")
# Plot
#plot_graph(gb, "./graphs/png_gB.png")
#plot_graph(gbp, "./graphs/png_gBp.png")
#plot_graph(gd, "./graphs/png_gD.png")
#plot_graph(gdp, "./graphs/png_gDp.png")
plot_graph(gbp_prunned, "./graphs/png_gBp_prunned.png")
plot_graph(gdp_prunned, "./graphs/png_gDp_prunned.png")


