import pandas as pd

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def plot_degree_distribution(degree_dict: dict, normalized: bool = False, loglog: bool = False) -> None:
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    pass
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
    pass
    # ------------------- END OF MAIN ------------------------ #
