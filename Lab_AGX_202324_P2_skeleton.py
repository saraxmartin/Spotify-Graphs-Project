import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #

def both_graphs_artists(mean_features_artists: pd.DataFrame, gb: nx.DiGraph, gd: nx.DiGraph, out_filename: str)-> pd.DataFrame:
    """
    Get a Dataframe with the mean features of only artists that belong in both graphs

    :param mean_features_artists: a pandas DataFrame.
    :param gb: graph BFS
    :param gd: graph DFS
    :param out_filename: name of the file that will be saved.
    :return: updated_df: updated pandas DatFrame
    """
    # Get common nodes of both graphs
    common_nodes = set(gb.nodes()).intersection(gd.nodes())
    # Filter DataFrame
    updated_df = mean_features_artists[mean_features_artists['artist_id'].isin(common_nodes)]
    # Save new dataframe
    updated_df.to_csv(out_filename, index=False)

    return updated_df
# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    g_undirected = nx.Graph()

    for node, attrs in g.nodes(data=True):
        g_undirected.add_node(node, **attrs)

    for u,v in g.edges():
      if g.has_edge(v,u):
         g_undirected.add_edge(u,v)

    nx.write_graphml(g_undirected, out_filename)

    return g_undirected
    # ----------------- END OF FUNCTION --------------------- #


def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    removed_nodes = []
    # Obtain a dictionary with key:node, value:degree
    degrees = dict(g.degree())
    # Delete nodes with degree < min_degree
    for node, degree in degrees.items():
        if degree < min_degree:
            # Prune the node from the graph.
            removed_nodes.append(node)
            g.remove_node(node)

    # Remove 0 degree nodes
    degrees = dict(g.degree())
    for node, degree in degrees.items():
        if degree == 0:
            # Prune the node from the graph.
            removed_nodes.append(node)
            g.remove_node(node)

    print("List of removed nodes: ", removed_nodes)
    nx.write_graphml(g, out_filename)
    return g
    # ----------------- END OF FUNCTION --------------------- #


def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if min_weight==None and min_percentile==None:
      raise ValueError("One of the parameters [min_weight, min_percentile] should be specified")
    elif min_weight is not None and min_percentile is not None:
      raise ValueError("Only one of the parameters [min_weight, min_percentile] should be specified")

    # Create a list of edges to prune
    edges_to_prune = []

    # Get edge weights
    edge_weights = [data['weight'] for u, v, data in g.edges(data=True)]

    for u, v, data in g.edges(data=True):
        weight = data.get('weight', None)

        if min_weight is not None and weight is not None:
            if weight < min_weight:
                edges_to_prune.append((u, v))

        elif min_percentile is not None and weight is not None:
            weight_percentile = np.percentile(edge_weights, min_percentile)
            if weight < weight_percentile:
                edges_to_prune.append((u, v))

    if min_weight is not None:
        print(f"Number of edges to prune with min_weight {min_weight}: ",len(edges_to_prune))
    elif min_percentile is not None:
        print(f"Number of edges to prune with min_percentile {min_percentile}: {len(edges_to_prune)}. The min_weight is {weight_percentile}.")

    print(f"Number of edges initially: {len(g.edges())}; Number of edges after deletion: {len(g.edges()) - len(edges_to_prune)}.")
    # Remove edges from the graph
    for u, v in edges_to_prune:
        g.remove_edge(u, v)

    # Remove zero-degree nodes
    zero_degree_nodes = [node for node, degree in dict(g.degree()).items() if degree == 0]
    g.remove_nodes_from(zero_degree_nodes)

    if out_filename is not None:
        nx.write_graphml(g, out_filename)

    return g
    # ----------------- END OF FUNCTION --------------------- #

def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    mean_df = tracks_df.loc[:,['artist_name','artist_id','duration','popularity','danceability','acousticness','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence']]
    mean_df = mean_df.groupby(['artist_name','artist_id']).mean()
    mean_df = mean_df.reset_index()
    return mean_df
    # ----------------- END OF FUNCTION --------------------- #


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    G = nx.Graph()

    # Create all nodes
    for i in range(len(artist_audio_features_df)):
      artist = artist_audio_features_df.iloc[i]
      if not G.has_node(artist['artist_id']):
        G.add_node(artist['artist_id'], name = artist['artist_name'])

    # Create edges
    for i in range(len(artist_audio_features_df)):
      artist = np.array(artist_audio_features_df.iloc[i,2:]).reshape(1, -1)

      for j in range(i+1, len(artist_audio_features_df)):
        artist2 = np.array(artist_audio_features_df.iloc[j,2:]).reshape(1, -1)
        if similarity=='cosine':
          sim = cosine_similarity(artist,artist2)[0][0]
        elif similarity=='euclidean':
          dist = euclidean_distances(artist,artist2)[0][0]
          sim = 1 / (1 + dist)

        artist_id_i = artist_audio_features_df.iloc[i]['artist_id']
        artist_id_j = artist_audio_features_df.iloc[j]['artist_id']
        G.add_edge(artist_id_i, artist_id_j, weight=sim)

    nx.write_graphml(G, out_filename)

    return G
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    # Get undirected graphs of gB and gD
    gb = nx.read_graphml("./graphs/gB")
    gd = nx.read_graphml("./graphs/gD")
    gb2 = retrieve_bidirectional_edges(gb, "./graphs/gBp")
    gd2 = retrieve_bidirectional_edges(gd, "./graphs/gDp")

    # Prune low degree nodes
    gb2 = nx.read_graphml("./graphs/gBp")
    gb2_prunned = prune_low_degree_nodes(gb2, min_degree=1, out_filename="./graphs/gBp_prunned")
    gd2 = nx.read_graphml("./graphs/gDp")
    gd2_prunned = prune_low_degree_nodes(gd2, min_degree=1, out_filename="./graphs/gDp_prunned")


    # Get undirected graph gw
    songs_df = pd.read_csv("./graphs/songs_updated.csv")
    mean_audio_features_df = compute_mean_audio_features(songs_df)
    mean_audio_features_df.to_csv("./graphs/mean_audio_features_songs.csv", index=False)
    mean_feat_updated = both_graphs_artists(mean_audio_features_df, gb, gd, out_filename="./graphs/mean_audio_features_updated.csv")
    gw = create_similarity_graph(mean_feat_updated, similarity="cosine",out_filename="./graphs/gw")

    
    # ------------------- END OF MAIN ------------------------ #
