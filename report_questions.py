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
    Analyze the weakly connected components and strongly connected components of a directed graph.

    :param graph: A networkx DiGraph (directed graph).
    :return: Number of weakly connected components and strongly connected components.
    """
    # Calculate weakly connected components
    wcc = list(nx.weakly_connected_components(graph))
    num_wcc = len(wcc)
    {}
    # Calculate strongly connected components
    scc = list(nx.strongly_connected_components(graph))
    num_scc = len(scc)
    
    print(f"{name}: Num weak connected components: ",num_wcc)
    print(f"{name}: Num strong connected components: ",num_scc)
    

# ----------------- PART 3 --------------------- #




# ----------------- PART 4 --------------------- #





# ------------ ANSWER QUESTIONS------------------ #
if __name__ == "__main__":

    # Import graphs and datasets
    gb = nx.read_graphml("./graphs/gB")
    gd = nx.read_graphml("./graphs/gD")
    #songs = pd.read_csv("./songs.csv")

    # PART 1: DATA ADQUISITION
    print("#-------------PART 1-----------------#\n")
    print("#----------Info about GB:------------#\n")
    graph_info(gb)
    print("\n#----------Info about GD:------------#")
    graph_info(gd)

    print("\n#----------Info about songs:-----------#")
    # datase_info(songs)

    # PART 2: DATA ADQUISITION
    print("\n#-------------PART 2-----------------#\n")
    analyze_graph_components(gb, "gb")
    analyze_graph_components(gd, "gd")

