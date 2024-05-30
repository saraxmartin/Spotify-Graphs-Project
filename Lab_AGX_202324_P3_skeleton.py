import networkx as nx
from networkx.algorithms.community import girvan_newman
import community as community_louvain
import csv

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #

def save_dict_to_csv(dictionary: dict, filename:str):
    """
    Save dictionary to a CSV file.
    :param: dictionary
    :param: filename (str)
    """
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            writer.writerow([key, value])

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def num_common_nodes(list_graphs: list) -> int:
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    nodes_graphs = [] 
    # Get list of sets of unique nodes in each graph
    for graph in list_graphs:
        nodes = set(graph.nodes)
        nodes_graphs.append(nodes)

    common_nodes = set.intersection(*nodes_graphs)

    num_common_nodes = len(common_nodes)

    return num_common_nodes
    # ----------------- END OF FUNCTION --------------------- #


def get_degree_distribution(g: nx.Graph, filename:str) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    degree_distribution = {}
    for node in g.nodes():
        degree = g.degree[node]
        if degree not in degree_distribution.keys():
            degree_distribution[degree] = 1
        else:
            degree_distribution[degree] += 1

    # Save dictionary to CSV for future use
    save_dict_to_csv(degree_distribution, filename)
    
    return degree_distribution

    # ----------------- END OF FUNCTION --------------------- #


def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if metric=='degree':
      centrality = nx.degree_centrality(g)
    elif metric=='betweenness':
      centrality = nx.betweenness_centrality(g)
    elif metric=='closeness':
      centrality = nx.closeness_centrality(g)
    elif metric=='eigenvector':
      centrality = nx.eigenvector_centrality(g)

    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    top_k_nodes = sorted_nodes[:num_nodes]

    return top_k_nodes
    # ----------------- END OF FUNCTION --------------------- #


def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    cliques = [clique for clique in nx.enumerate_all_cliques(g) if len(clique) >= min_size_clique]
    nodes = list({node for clique in cliques for node in clique})

    return (cliques,nodes)
    # ----------------- END OF FUNCTION --------------------- #


def detect_communities(g: nx.Graph, method: str) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if method == 'girvan-newman':
        comp = girvan_newman(g)
        # Get the first partition (after the first removal of edges)
        communities = next(comp)
        # Convert to a list of lists
        communities = [list(c) for c in communities]
        # Calculate modularity
        modularity = nx.algorithms.community.quality.modularity(g, communities)
    
    elif method == 'louvain':
        partition = community_louvain.best_partition(g)
        # Convert partition dictionary to list of lists
        communities = {}
        for node, community in partition.items():
            communities.setdefault(community, []).append(node)
        communities = list(communities.values())
        # Calculate modularity
        modularity = community_louvain.modularity(partition, g)
    
    else:
        raise ValueError("Method must be 'girvan-newman' or 'louvain'")

    return communities, modularity

    # ----------------- END OF FUNCTION --------------------- #


if __name__ == '__main__':
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    #! pip install python-louvain

    gb = nx.read_graphml("./graphs/gB")
    gd = nx.read_graphml("./graphs/gD")
    gb2 = nx.read_graphml("./graphs/gBp")
    gd2 = nx.read_graphml("./graphs/gDp")

    # Common nodes
    print("Number of nodes of gB: ", len(gb.nodes))
    print("Number of nodes of gB bidir: ", len(gb2.nodes))
    print("Number of nodes if gD: ", len(gd.nodes))
    print("Number of nodes of gD bidir: ", len(gd2.nodes))

    n_common_nodes = num_common_nodes([gb,gd])
    print("Number of common nodes of gB and gD bidir: ", n_common_nodes)

    # Degree distribution
    degree_distribution_gb = get_degree_distribution(gb, filename="./degree_distribution/dict_gb.csv")
    print("Degree distribution of gB: ", degree_distribution_gb)
    degree_distribution_gb2 = get_degree_distribution(gb2, filename="./degree_distribution/dict_gbp.csv")
    print("Degree distribution of gB bidir: ", degree_distribution_gb2)
    degree_distribution_gd = get_degree_distribution(gd, filename="./degree_distribution/dict_gd.csv")
    print("Degree distribution of gD: ", degree_distribution_gd)
    degree_distribution_gd2 = get_degree_distribution(gd2, filename="./degree_distribution/dict_gdp.csv")
    print("Degree distribution of gD bidir: ", degree_distribution_gb2)

    # K most central node
    # ...

    # Find cliques 
    # ...

    # Detect communities
    communities_gb, modularity_gb = detect_communities(gb2, method='girvan-newman')
    print(f"Number of communities with Girvan-Newman: {len(communities_gb)}. Modularity: {modularity_gb}")
    communities_gb2, modularity_gb2 = detect_communities(gb2, method='louvain')
    print(f"Number of communities with Louvain: {len(communities_gb2)}. Modularity: {modularity_gb2}")

    # ------------------- END OF MAIN ------------------------ #
