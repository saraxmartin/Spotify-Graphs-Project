import statistics
import networkx as nx
import pandas as pd
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
    

# ----------------- PART 3 --------------------- #


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


# ------------ ANSWER QUESTIONS------------------ #
if __name__ == "__main__":

    # Import graphs and datasets
    gb = nx.read_graphml("./graphs/gB")
    gd = nx.read_graphml("./graphs/gD")
    gbp = nx.read_graphml("./graphs/gBp")
    gdp = nx.read_graphml("./graphs/gDp")
    gbp_prunned = nx.read_graphml("./graphs/gBp_prunned")
    gdp_prunned = nx.read_graphml("./graphs/gDp_prunned")
    songs = pd.read_csv("./graphs/songs.csv")

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
    # datase_info(songs)

    # PART 2: DATA ADQUISITION
    print("\n#-------------PART 2-----------------#\n")
    # 1/2. Strong / weak connected components
    analyze_graph_components(gb, "gb")
    analyze_graph_components(gd, "gd")
    analyze_graph_components(gbp, "gbp")
    analyze_graph_components(gdp, "gdp")
    analyze_graph_components(gbp_prunned, "gbp prunned")
    analyze_graph_components(gdp_prunned, "gdp prunned")
    # ...

    # PART 3: DATA PREPROCESSING
    print("\n#-------------PART 3-----------------#\n")
    # 1/2/3/4/5. Used functions in notebook 3.
    # ...

    # PART 4: DATA PREPROCESSING
    print("\n#-------------PART 4-----------------#\n")
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
