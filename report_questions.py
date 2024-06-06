import statistics
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from Lab_AGX_202324_P2_skeleton import prune_low_weight_edges
from Lab_AGX_202324_P4_skeleton import find_node_by_attribute

# ----------------- PART 1 --------------------- #

def dataset_info(df):
    num_songs = len(df)
    num_artists = df['artist_id'].nunique()
    num_albums = df['album_id'].nunique()
    print("Number of songs:", num_songs)
    print("Number of different artists:", num_artists)
    print("Number of different albums:", num_albums)


def graph_info(G):
    print("Order:", G.order())
    print("Size:", G.size())

    if isinstance(G, nx.DiGraph):

        # Calculate in-degrees and out-degrees
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]

        # Calculate minimum, maximum, and median of in-degrees
        min_in_degree = min(in_degrees)
        max_in_degree = max(in_degrees)
        median_in_degree = statistics.median(in_degrees)

        # Calculate minimum, maximum, and median of out-degrees
        min_out_degree = min(out_degrees)
        max_out_degree = max(out_degrees)
        median_out_degree = statistics.median(out_degrees)

        print("\nIN-DEGREE:")
        print("Minimum:", min_in_degree)
        print("Maximum:", max_in_degree)
        print("Median:", median_in_degree)
        #print("In-degrees:", sorted(in_degrees))

        print("\nOUT-DEGREE:")
        print("Minimum:", min_out_degree)
        print("Maximum:", max_out_degree)
        print("Median:", median_out_degree)
        #print("Out-degrees:", sorted(out_degrees))



# ----------------- PART 2 --------------------- #
def analyze_graph_components(graph, name):
    """
    Analyze the connected components of a graph, whether directed or undirected.

    For directed graphs, it analyzes weakly connected components and strongly connected components.
    For undirected graphs, it analyzes connected components.

    :param graph: A networkx Graph (undirected) or DiGraph (directed).
    :param name: Name of the graph for printing purposes.
    :return: None
    """
    if isinstance(graph, nx.DiGraph):
        # Directed graph: Calculate weakly and strongly connected components
        wcc = list(nx.weakly_connected_components(graph))
        num_wcc = len(wcc)
        
        scc = list(nx.strongly_connected_components(graph))
        num_scc = len(scc)
        
        print(f"{name}: Num weakly connected components: {num_wcc}")
        print(f"{name}: Num strongly connected components: {num_scc}")
    elif isinstance(graph, nx.Graph):
        # Undirected graph: Calculate connected components
        cc = list(nx.connected_components(graph))
        num_cc = len(cc)
        
        print(f"{name}: Num connected components: {num_cc}")
    else:
        print(f"{name}: Unsupported graph type")

def gw_report(gw):
    # (a) Finding the most and least similar artist pairs
    edges = gw.edges(data=True)
    most_similar_artists = max(edges, key=lambda x: x[2]['weight'])
    least_similar_artists = min(edges, key=lambda x: x[2]['weight'])

    names_most_similar_artists = find_name_by_id(gw,most_similar_artists)
    names_least_similar_artists = find_name_by_id(gw,least_similar_artists)

    print(f"Most similar artists: {names_most_similar_artists}")
    print(f"Least similar artists: {names_least_similar_artists}")

    # (b) Calculating the weighted degree centrality
    weighted_degrees = dict(gw.degree(weight='weight'))
    most_similar_to_all = max(weighted_degrees, key=weighted_degrees.get)
    least_similar_to_all = min(weighted_degrees, key=weighted_degrees.get)

    name_most_similar_to_all = find_name_by_id(gw, [most_similar_to_all])
    name_least_similar_to_all = find_name_by_id(gw, [least_similar_to_all])

    print(f"Most similar to all: {name_most_similar_to_all}")
    print(f"Least similar to all: {name_least_similar_to_all}")

# ----------------- PART 3 --------------------- #

def find_dominating_set(graph):
    # Calculate betweenness centrality for each node
    betweenness = nx.betweenness_centrality(graph)
    
    # Initialize the set of nodes to cover and the dominating set
    nodes_to_cover = set(graph.nodes())
    dominating_set = set()

    while nodes_to_cover:
        # Find the node with the highest betweenness centrality
        max_betweenness_node = max(nodes_to_cover, key=lambda n: betweenness[n])
        
        # Add this node to the dominating set
        dominating_set.add(max_betweenness_node)
        
        # Remove the node and its neighbors from the set of nodes to be covered
        nodes_to_cover -= set(graph.neighbors(max_betweenness_node))
        nodes_to_cover.discard(max_betweenness_node)
        
        # Remove the selected node from betweenness dictionary to avoid selecting it again
        del betweenness[max_betweenness_node]

    return dominating_set

def select_top_nodes_by_betweenness(graph, budget, cost_per_artist):
    # Calculate betweenness centrality for each node
    betweenness = nx.betweenness_centrality(graph)
    
    # Sort nodes by betweenness centrality in descending order
    sorted_nodes = sorted(betweenness, key=betweenness.get, reverse=True)
    
    # Determine the number of artists we can afford with the given budget
    num_artists = budget // cost_per_artist
    
    # Select the top nodes based on the budget
    selected_artists = sorted_nodes[:num_artists]

    # Get the names of the artists
    names_selected_artists = find_name_by_id(graph, selected_artists)
    
    # Get the betweenness centrality values of the selected artists
    selected_artists_betweenness = [betweenness[artist] for artist in selected_artists]

    return names_selected_artists, selected_artists_betweenness

# ----------------- PART 4 --------------------- #
def find_name_by_id(graph: nx.Graph, ids:list):
    """
    Find the node Name by its ID.

    :param graph: A NetworkX graph.
    :param attribute: The attribute name to search for.
    :param value: The attribute value to match.
    :return: The node ID with the matching attribute value, or None if not found.
    """
    new_list = []
    for id in ids:
        for node, attrs in graph.nodes(data=True):
            if node == id:
                new_list.append(attrs["name"])
    return new_list

def plot_connected_component_sizes(graph: nx.Graph, thresholds, filename):
    """
    Show plot of evolution of biggest connected component size 
    with different min_percentiles thresholds.

    :param graph: A Networkx graph.
    :param thresholds: a list of thresholds.
    :param filename: str of filename.
    """
    sizes = []

    for threshold in thresholds:
        pruned_graph = prune_low_weight_edges(graph.copy(), min_percentile=threshold)
        largest_cc = max(nx.connected_components(pruned_graph), key=len)
        sizes.append(len(largest_cc))

    plt.plot(thresholds, sizes)
    plt.xlabel('Threshold')
    plt.ylabel('Size of the Largest Connected Component')
    plt.title('Evolution of Largest Connected Component with Threshold')
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.show()

def find_optimal_percentile(graph: nx.Graph, start_percentile=90, step=1):
    """
    Find the optimal percentile to prune edges of a weighted similarity graph to preserve the size of the largest connected component.

    :param graph: Weighted similarity graph (NetworkX).
    :param start_percentile: Starting percentile value.
    :param step: Step size for decreasing the percentile.
    :return: Optimal percentile value.
    """
    original_largest_cc_size = len(max(nx.connected_components(graph), key=len))
    print("Original size of the larges connected component: ", original_largest_cc_size)
    percentile = start_percentile

    while True:
        pruned_graph = prune_low_weight_edges(graph.copy(), min_percentile=percentile)
        largest_cc_size = len(max(nx.connected_components(pruned_graph), key=len))
        print(f"Largest conected component size after prunning with percentile {percentile}: {largest_cc_size}")

        if largest_cc_size == original_largest_cc_size:
            percentile += step
        else:
            break

    return percentile

# ------------ ANSWER QUESTIONS------------------ #
if __name__ == "__main__":

    # Import graphs and datasets
    gb = nx.read_graphml("./graphs/gB")
    gd = nx.read_graphml("./graphs/gD")
    gbp = nx.read_graphml("./graphs/gBp")
    gdp = nx.read_graphml("./graphs/gDp")
    gbp_prunned = nx.read_graphml("./graphs/gBp_prunned")
    gdp_prunned = nx.read_graphml("./graphs/gDp_prunned")
    songs = pd.read_csv("./graphs/songs_updated.csv")
    gw = nx.read_graphml("./graphs/gw")

    # PART 1: DATA ADQUISITION
    print("#-------------PART 1-----------------#\n")
    print("#----------Info about GB:------------#\n")
    graph_info(gb)
    print("\n#----------Info about GD:------------#")
    graph_info(gd)
    print("\n#----------Info about GBp:------------#")
    graph_info(gbp)
    graph_info(gbp_prunned)
    print("\n#----------Info about GDp:------------#")
    graph_info(gdp)
    graph_info(gdp_prunned)
    print("\n#----------Info about songs:-----------#")
    dataset_info(songs)

    # PART 2: DATA ADQUISITION
    print("\n#---------------PART 2-------------------#\n")
    # 1/2. Strong / weak connected components
    print("#----------Connected components---------#\n")
    analyze_graph_components(gb, "gb")
    analyze_graph_components(gd, "gd")
    analyze_graph_components(gbp, "gbp")
    analyze_graph_components(gdp, "gdp")
    analyze_graph_components(gbp_prunned, "gbp prunned")
    analyze_graph_components(gdp_prunned, "gdp prunned")
    # 3. gw report
    print("\n#----------------gW report---------------#\n")
    gw_report(gw)

    # PART 3: DATA PREPROCESSING
    print("\n#-------------PART 3-----------------#\n")
    # 1/2/3/4/5. Used functions in notebook 3.
    # 6. Advertising campaign
    # a)
    print("\n#-------------6.a)-----------------#\n")
    dominating_set_gB = find_dominating_set(gb)
    print(f"Number of dominating nodes of gB: {len(dominating_set_gB)}")
    dominating_set_gD = find_dominating_set(gd)
    print(f"Number of dominating nodes of gD: {len(dominating_set_gD)}")
    cost_gB = len(dominating_set_gB) * 100
    cost_gD = len(dominating_set_gD) * 100
    print(f"Minimum cost for gB: {cost_gB} euros")
    print(f"Minimum cost for gD: {cost_gD} euros")

    print("\n#-------------6.b)-----------------#\n")
    # Selecting top nodes for gB and gD
    selected_artists_gB, betweenness_gb = select_top_nodes_by_betweenness(gb, budget=400, cost_per_artist=100)
    selected_artists_gD, betweenness_gd = select_top_nodes_by_betweenness(gd, budget=400, cost_per_artist=100)
    print(f"Selected artists for gB: {selected_artists_gB} and their betweenness: {betweenness_gb}")
    print(f"Selected artists for gD: {selected_artists_gD} and their betweenness: {betweenness_gd}")

    # 7.
    print("\n#--------------7------------------#\n")
    start_artist, end_artist = "Taylor Swift", "THE DRIVER ERA"
    start_artist_id = find_node_by_attribute(gb, attribute="name", value=start_artist)
    end_artist_id = find_node_by_attribute(gb, attribute="name", value=end_artist)

    try: # Find the shortest path
        shortest_path = nx.shortest_path(gb, source=start_artist_id, target=end_artist_id)
        num_hops = len(shortest_path) - 1
        print(f"Minimum number of hops from '{start_artist}' to '{end_artist}': {num_hops}")
        print(f"Path: {find_name_by_id(gb,shortest_path)}")
    except nx.NetworkXNoPath:
        print(f"There is no path from '{start_artist}' to '{end_artist}' in the graph.")
    except nx.NodeNotFound as e:
        print(e)

    # PART 4: DATA PREPROCESSING
    print("\n#-------------PART 4-----------------#\n")
    print("\n#-------------1.c)-----------------#\n")
    # 1.c)
    id_Taylor = "06HL4z0CvFAxyc27GXpf02"
    id_most_similar = "6KImCVD70vtIoJWnq6nGn3"
    id_less_similar = "25uiPmTg16RbhZWAqwLBy5"
    # Most similar
    distance_gb_most = nx.shortest_path(gb, source=id_Taylor, target=id_most_similar)
    names_path = find_name_by_id(gb, distance_gb_most)
    print("Distance gB between Taylor and Harry:",distance_gb_most, names_path, len(distance_gb_most))
    distance_gd_most = nx.shortest_path(gd, source=id_Taylor, target=id_most_similar)
    names_path = find_name_by_id(gd, distance_gd_most)
    print("Distance gD between Taylor and Harry:",distance_gd_most, names_path, len(distance_gd_most))
    # Less similar
    distance_gb_less = nx.shortest_path(gb, source=id_Taylor, target=id_less_similar)
    names_path = find_name_by_id(gb, distance_gb_less)
    print("Distance gB between Taylor and Charli XCX:",distance_gb_less, names_path, len(distance_gb_less))
    distance_gd_less = nx.shortest_path(gd, source=id_Taylor, target=id_less_similar)
    names_path = find_name_by_id(gd, distance_gd_less)
    print("Distance gD between Taylor and Charli XCX:",distance_gd_less, names_path, len(distance_gd_less))
    
    print("\n#-------------1.d)-----------------#\n")
    # 1.d)
    print(f"Initial number of nodes: {len(gw.nodes())}, Initial number of edges: {len(gw.edges())}")
    print("\nStarting with percentile 50...")
    optimal_percentile = find_optimal_percentile(gw, start_percentile=50)
    print("Optimal percentile:", optimal_percentile-1)
    print("\nCreating gw prunned...")
    gw_prunned = prune_low_weight_edges(gw, min_weight=None, min_percentile=optimal_percentile-1, out_filename="./graphs/gw_prunned")
    print(f"Final number of nodes: {len(gw_prunned.nodes())}, Final number of edges: {len(gw_prunned.edges())}")
    
    print("\n#-------------Ex.4 e)-----------------#\n")
    # e) from Part 4 ex4: Prune low weight edges
    thresholds = [90,91,92,93,94,95,96,97,98,99]
    plot_connected_component_sizes(gw, thresholds, './graphs/plots/connected_component_sizes.png')