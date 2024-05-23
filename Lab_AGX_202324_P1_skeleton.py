import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def addNode(g: nx.DiGraph, artist: str) -> nx.DiGraph:
    """
    Add a node to the graph with all its information.
    :param g: nx.Graph(), the graph we want to visualize.
    :param artist: artist IP
    """
    g.add_node(artist,
                name = sp.artist(artist)["name"],
                followers = sp.artist(artist)["followers"]["total"],
                popularity = sp.artist(artist)["popularity"],
                genres = ", ".join(sp.artist(artist)["genres"]))

    return g

def visualize_graph(G: nx.DiGraph, title: str):
    """
    Visualize the graph with artists' names.

    :param G: nx.Graph(), the graph we want to visualize.
    :param title: the title for our plot.
    """
    # Extract the name labels of the artists
    labels = {node: data['name'] for node, data in G.nodes(data=True)}
    # Use spring layout for better visualization
    pos = nx.spring_layout(G)
    # Plot and draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=500, font_size=12, font_weight='bold')
    plt.title(title)
    plt.show()


def related_artists(sp: spotipy.client.Spotify, artist_id: str) -> list:
    """
    Get a list of related artists to an artist.

    :param sp: spotipy client object.
    :param artist_id: artist id.
    :return: artists_ids: list of tuple of format: (new_artist, root_artist)
    """

    # Get dictionary with the related artists
    artists = sp.artist_related_artists(artist_id)
    # Extract the artists' ids
    artists_ids = [(artist["id"],artist_id) for artist in artists["artists"]]

    return artists_ids

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def search_artist(sp: spotipy.client.Spotify, artist_name: str) -> str:
    """
    Search for an artist in Spotify.

    :param sp: spotipy client object
    :param artist_name: name to search for.
    :return: spotify artist id.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    results = sp.search(q='artist:' + artist_name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        artist = items[0]
        return artist['id']
    else:
        return None
    # ----------------- END OF FUNCTION --------------------- #


def crawler(sp: spotipy.client.Spotify, seed: str, max_nodes_to_crawl: int, strategy: str = "BFS",
            out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.

    :param sp: spotipy client object
    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    # Initialize the graph with the seed node
    G = nx.DiGraph()
    G = addNode(G,seed)

    # Initialize lists of visited and queue
    created = [seed] # Stores created nodes (not explored)
    visited = [seed] # Stores already visited/explored artists
    queue = [] # Stores artists to visit

    # Add to the queue the seed artist related artists
    son_artists = related_artists(sp, seed)
    queue.extend(son_artists)

    # Add son artist of the seed artists to the graph
    for artist in son_artists:
        G = addNode(G,artist)
        G.add_edge(seed,artist)
        created.append(artist)

    seed_name = G.nodes[seed]["name"]

    with tqdm(total=max_nodes_to_crawl, desc=f"Creating {strategy} graph from {seed_name}") as pbar:
        while (len(visited) < max_nodes_to_crawl+1) and (len(queue) != 0):

            # Get the first artist of the queue and its root artist
            current_artist = queue[0][0]
            root_artist = queue[0][1]

            # Add an edge between current_artist and next_artist
            #G.add_edge(root_artist,current_artist) #edge ya esta
            
            # Pop artist from queue
            queue = queue[1:]
            
            # Get related artists of next_artist
            son_artists = related_artists(sp, current_artist)

            if current_artist not in visited:
                # Add new artists to visited
                visited.append(current_artist)

                if strategy == "BFS":
                    # Add the related artists of the new artists to the end of the queue
                    queue.extend(son_artists)

                elif strategy == "DFS":
                    # Add the related artists of the new artists to the beggining of the queue
                    queue = son_artists + queue
            
                # Add son artists to graph
                for artist in son_artists:
                    if artist not in created:
                        G = addNode(G,artist)
                    else:
                        created.append(artist)
                    G.add_edge(current_artist,artist)


            # Update progress bar
            pbar.update(1)

    # Save file as graphml
    nx.write_graphml(G, out_filename)

    return G
    # ----------------- END OF FUNCTION --------------------- #


def get_track_data(sp: spotipy.client.Spotify, graphs: list, out_filename: str) -> pd.DataFrame:
    """
    Get track data for each visited artist in the graph.

    :param sp: spotipy client object
    :param graphs: a list of graphs with artists as nodes.
    :param out_filename: name of the csv output file.
    :return: pandas dataframe with track data.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    artists = []

    song_name, song_id, song_duration = [], [], []
    acoustic, dance, energy, instrumental, liveness, loudness, speech, tempo, valence, popularity  = [],[],[],[],[],[],[],[],[],[]
    album_name, album_id, album_date = [],[],[]
    artist_name, artist_id = [],[]

    for graph in graphs:
      artists.extend(list(graph.nodes()))

    for art_id in artists:
      for track in sp.artist_top_tracks(art_id, country='ES')['tracks']:
        artist = sp.artist(art_id)['name']
        track_id = track['id']
        song_name.append(track['name'])
        song_id.append(track_id)
        popularity.append(track['popularity'])

        feat = sp.audio_features(track_id)[0]
        song_duration.append(feat['duration_ms'])
        acoustic.append(feat['acousticness'])
        dance.append(feat['danceability'])
        energy.append(feat['energy'])
        speech.append(feat['speechiness'])
        instrumental.append(feat['instrumentalness'])
        loudness.append(feat['loudness'])
        tempo.append(feat['tempo'])
        liveness.append(feat['liveness'])
        valence.append(feat['valence'])

        album_name.append(track['album']['name'])
        album_id.append(track['album']['id'])
        album_date.append(track['album']['release_date'])

        artist_name.append(artist)
        artist_id.append(art_id)

    df = pd.DataFrame({'song_name':song_name,
                      'song_id':song_id,
                      'duration':song_duration,
                      'popularity':popularity,
                      'danceability':dance,
                      'acousticness':acoustic,
                      'energy':energy,
                      'instrumentalness':instrumental,
                      'liveness':liveness,
                      'loudness':loudness,
                      'speechiness':speech,
                      'tempo':tempo,
                      'valence':valence,
                      'album_name':album_name,
                      'album_id':album_id,
                      'album_release_date':album_date,
                      'artist_name':artist_name,
                      'artist_id':artist_id
                      })

    out_filename = out_filename + ".csv"
    df.to_csv(out_filename, index=False)

    return df
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":

    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #

    # Get Spotify client object
    CLIENT_ID = "59435f3767f5407395e8a21c91f1b719"
    CLIENT_SECRET = "aa930752eeeb4e1ab36bd7bfff2cd0ff"
    auth_manager = SpotifyClientCredentials (client_id = CLIENT_ID, client_secret = CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Get the artist ID
    seed = search_artist(sp,"Taylor Swift")

    # Create and visualize the BFS graph
    gb = crawler(sp, seed, max_nodes_to_crawl=25, strategy="BFS", out_filename="./graphs/gB")
    visualize_graph(gb, title="BFS Taylor Swift graph")

    # Create and visualize the DFS graph
    gd = crawler(sp, seed, max_nodes_to_crawl=100, strategy="DFS", out_filename="./graphs/gD")
    visualize_graph(gd, title="DFS Taylor Swift graph")

    # Obtain dataset of songs from artists of previous graphs
    D = get_track_data(sp, graphs=[gb,gd], out_filename="gB_TaylorSwift")

    # Create BFS graph for Pastel Ghost
    seed = search_artist(sp,"Pastel Ghost")
    hb = crawler(sp, seed, max_nodes_to_crawl=100, strategy="BFS", out_filename="./graphs/hB")
    visualize_graph(hb, title="BFS Pastel Ghost graph")

    # ------------------- END OF MAIN ------------------------ #
