import pandas as pd
import matplotlib.pyplot as plt
import csv

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


def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
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

    plt.savefig('audio_features_comparison.png')

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
    pass
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    # Plot degree distribution
    dict_gb = load_dict_from_csv("./degree_distribution/dict_gb.csv")
    plot_degree_distribution(dict_gb, title= "GB", filename="./degree_distribution/degree_distr_gb1.png")
    plot_degree_distribution(dict_gb, title= "GB", filename="./degree_distribution/degree_distr_gb2.png", normalized=True, loglog=True)
    dict_gb_bidir = load_dict_from_csv("./degree_distribution/dict_gb_bidir.csv")
    plot_degree_distribution(dict_gb, title= "GB bidir", filename="./degree_distribution/degree_distr_gb_bidir1.png")
    plot_degree_distribution(dict_gb, title= "GB bidir", filename="./degree_distribution/degree_distr_gb_bidir2.png", normalized=True, loglog=True)
    dict_gd = load_dict_from_csv("./degree_distribution/dict_gd.csv")
    plot_degree_distribution(dict_gd, title= "GD", filename="./degree_distribution/degree_distr_gd1.png")
    plot_degree_distribution(dict_gd, title= "GD", filename="./degree_distribution/degree_distr_gd2.png", normalized=True, loglog=True)
    dict_gd_bidir = load_dict_from_csv("./degree_distribution/dict_gd_bidir.csv")
    plot_degree_distribution(dict_gd, title= "GD bidir", filename="./degree_distribution/degree_distr_gd_bidir1.png")
    plot_degree_distribution(dict_gd, title= "GD bidir", filename="./degree_distribution/degree_distr_gd_bidir2.png", normalized=True, loglog=True)
    
    # Plot audio features
    # ...
    # Plot similarity measure
    # ...
    # ------------------- END OF MAIN ------------------------ #
