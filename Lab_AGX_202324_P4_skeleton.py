import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
import networkx as nx

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def load_dict_from_csv(filename:str) -> dict:
    """
    Load dictionary from csv.
    :param: filename (str) where the dictionary is stored
    :returns: dictionary retrieved.
    """
    dictionary = {}
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            key, value = int(row[0]), int(row[1])
            dictionary[key] = value
    return dictionary

def most_less_similar_node(graph: nx.Graph, node: int):
    # Check if the node exists in the graph
    if node not in graph:
        raise ValueError(f"Node {node} not found in the graph.")
    
    # Get all edges connected to the given node
    edges = graph[node].items()
    
    # Find the edge with the highest weight
    most_similar = max(edges, key=lambda x: x[1]['weight'])
    less_similar = min(edges, key=lambda x: x[1]['weight'])
    
    # Get the most similar node ID and its similarity score
    most_similar_node_id = most_similar[0]
    similarity_score = most_similar[1]['weight']
    less_similar_node_id = less_similar[0]
    less_similarity_score = less_similar[1]['weight']
    
    # Get the 'name' attribute of the most similar node
    most_similar_node_name = graph.nodes[most_similar_node_id].get('name', 'Unknown')
    less_similar_node_name = graph.nodes[less_similar_node_id].get('name', 'Unknown')
    
    # Return the most similar node ID, similarity score, and name
    return (most_similar_node_id, similarity_score, most_similar_node_name), (less_similar_node_id, less_similarity_score, less_similar_node_name)

def find_node_by_attribute(graph: nx.Graph, attribute: str, value):
    """
    Find the node ID by a given attribute value.

    :param graph: A NetworkX graph.
    :param attribute: The attribute name to search for.
    :param value: The attribute value to match.
    :return: The node ID with the matching attribute value, or None if not found.
    """
    for node, attrs in graph.nodes(data=True):
        if attribute in attrs and attrs[attribute] == value:
            return node
    return None

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def plot_degree_distribution(degree_dict: dict, title:str, filename:str, normalized: bool = False, loglog: bool = False) -> None:
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    # Extract degrees and their counts
    degrees = list(degree_dict.keys())
    counts = list(degree_dict.values())
    
    if normalized:
        total_counts = sum(counts)
        probabilities = [count / total_counts for count in counts]
        y_values = probabilities
        ylabel = 'Probability'
    else:
        y_values = counts
        ylabel = 'Count'
    
    plt.figure(figsize=(10, 6))
    if loglog:
        plt.loglog(degrees, y_values, 'bo')
        plt.xlabel('Degree (log scale)')
        plt.ylabel(f'{ylabel} (log scale)')
        plt.title(f'Degree Distribution {title} (log-log scale)')
    else:
        plt.plot(degrees, y_values, 'bo')
        plt.xlabel('Degree')
        plt.ylabel(ylabel)
        plt.title(f'Degree Distribution {title}')
    
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.show()

    # ----------------- END OF FUNCTION --------------------- #


def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str, filename:str) -> None:
    """
    Plot a (single) figure with a plot of mean audio features of two different artists.

    :param artists_audio_feat: dataframe with mean audio features of artists.
    :param artist1_id: string with id of artist 1.
    :param artist2_id: string with id of artist 2.
    :return: None
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    feat1 = artists_audio_feat[artists_audio_feat['artist_id']==artist1_id]
    feat2 = artists_audio_feat[artists_audio_feat['artist_id']==artist2_id]

    # Ensure we have only one row per artist
    feat1 = feat1.iloc[0]
    feat2 = feat2.iloc[0]

    name1 = feat1['artist_name']
    name2 = feat2['artist_name']

    # Drop the 'artist_id' and 'artist_name' columns for plotting
    feat1 = feat1.drop(['artist_id', 'artist_name'])
    feat2 = feat2.drop(['artist_id', 'artist_name'])

    # Convert the 'duration' feature from seconds to hours
    feat1['duration'] /= 3600
    feat2['duration'] /= 3600


    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    indices = range(len(feat1.index))
    bar_width = 0.35
    bars1 = ax.bar(indices, feat1.values, bar_width, label=name1, alpha=0.7, color='blue')
    bars2 = ax.bar([i + bar_width for i in indices], feat2.values, bar_width, label=name2, alpha=0.7, color='orange')

    # Adding labels, title, and legend
    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.set_title('Audio Features Comparison')
    ax.set_xticks([i + bar_width / 2 for i in indices])
    ax.set_xticklabels(feat1.index, rotation=45)
    ax.legend()
    ax.grid(True)

    # Add numbers on top of the bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 2), ha='center', va='bottom')


    plt.savefig(filename, format='png', bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    # ----------------- END OF FUNCTION --------------------- #


def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    artist_names = artist_audio_features_df['artist_name']
    artist_audio_features_df.set_index('artist_id', inplace=True)
    artist_audio_features_df = artist_audio_features_df.drop(['artist_name'], axis=1)

    if similarity=="cosine":
        sim_matrix = cosine_similarity(artist_audio_features_df)
    else:
        euclidean_dist_matrix = euclidean_distances(artist_audio_features_df)
        # Convert Euclidean distances to similarity scores
        sim_matrix = 1 / (1 + euclidean_dist_matrix)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(sim_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    heatmap.set_xticklabels(artist_names, rotation=45)
    heatmap.set_yticklabels(artist_names, rotation=0)

    plt.title('Correlation Heatmap of Audio Features')
    plt.xlabel('Audio Features')
    plt.ylabel('Audio Features')

    plt.savefig(out_filename, format='png', bbox_inches='tight')

    plt.show()
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    # a. Plot degree distribution
    # gB
    #dict_gb = load_dict_from_csv("./degree_distribution/dict_gb.csv")
    #plot_degree_distribution(dict_gb, title= "GB", filename="./degree_distribution/degree_distr_gb1.png")
    #plot_degree_distribution(dict_gb, title= "GB", filename="./degree_distribution/degree_distr_gb2.png", normalized=True, loglog=True)
    #dict_gb_bidir = load_dict_from_csv("./degree_distribution/dict_gb_bidir.csv")
    #plot_degree_distribution(dict_gb, title= "GB bidir", filename="./degree_distribution/degree_distr_gbp1.png")
    #plot_degree_distribution(dict_gb, title= "GB bidir", filename="./degree_distribution/degree_distr_gbp2.png", normalized=True, loglog=True)
    #dict_gbp_prunned = load_dict_from_csv("./degree_distribution/dict_gbp_prunned.csv")
    #plot_degree_distribution(dict_gbp_prunned, title= "GB bidir prunned", filename="./degree_distribution/degree_distr_gbp_prunned1.png")
    #plot_degree_distribution(dict_gbp_prunned, title= "GB bidir prunned", filename="./degree_distribution/degree_distr_gbp_prunned2.png", normalized=True, loglog=True)
    # gD
    #dict_gd = load_dict_from_csv("./degree_distribution/dict_gd.csv")
    #plot_degree_distribution(dict_gd, title= "GD", filename="./degree_distribution/degree_distr_gd1.png")
    #plot_degree_distribution(dict_gd, title= "GD", filename="./degree_distribution/degree_distr_gd2.png", normalized=True, loglog=True)
    #dict_gd_bidir = load_dict_from_csv("./degree_distribution/dict_gdp1.csv")
    #plot_degree_distribution(dict_gd, title= "GD bidir", filename="./degree_distribution/degree_distr_gdp1.png")
    #plot_degree_distribution(dict_gd, title= "GD bidir", filename="./degree_distribution/degree_distr_gdp2.png", normalized=True, loglog=True)
    #dict_gdp_prunned = load_dict_from_csv("./degree_distribution/dict_gdp_prunned.csv")
    #plot_degree_distribution(dict_gdp_prunned, title= "GD bidir prunned", filename="./degree_distribution/degree_distr_gdp_prunned1.png")
    #plot_degree_distribution(dict_gdp_prunned, title= "GD bidir prunned", filename="./degree_distribution/degree_distr_gdp_prunned2.png", normalized=True, loglog=True)
    # gw
    #dict_gw = load_dict_from_csv("./degree_distribution/dict_gw.csv")
    #plot_degree_distribution(dict_gw, title= "GW", filename="./degree_distribution/degree_distr_gw1.png")
    #plot_degree_distribution(dict_gw, title= "GW", filename="./degree_distribution/degree_distr_gw2.png", normalized=True, loglog=True)

    # b/c. Plot audio features
    """gw = nx.read_graphml("./graphs/gw")
    # Find most and less similar node to Taylor Swift
    artist1_id = find_node_by_attribute(gw, attribute="name", value="Taylor Swift")
    print(artist1_id)
    (artist2_id, similarity_score_a2, artist2_name),(artist3_id, similarity_score_a3, artist3_name) = most_less_similar_node(gw, node=artist1_id)
    print("Most similar artist to Taylor Swift:", artist2_id, similarity_score_a2, artist2_name)
    print("Less similar artist to Taylor Swift:",artist3_id, similarity_score_a3, artist3_name)
    # Create audio features plots
    mean_audio_feat = pd.read_csv("./graphs/mean_audio_features.csv")
    plot_audio_features(mean_audio_feat, artist1_id, artist2_id, f"./mean_audio_features/mean_audio_feat_Taylor_{artist2_name}.png")
    plot_audio_features(mean_audio_feat, artist1_id, artist3_id, f"./mean_audio_features/mean_audio_feat_Taylor_{artist3_name}.png")"""
    

    # d. Plot similarity heatmap
    mean_audio_feat = pd.read_csv("./graphs/mean_audio_features.csv")
    plot_similarity_heatmap(mean_audio_feat, similarity="cosine", out_filename="./graphs/plots/similarity_heatmap_cosine.png")
    
    # e. Plot in plot_graphs
    # ------------------- END OF MAIN ------------------------ #
