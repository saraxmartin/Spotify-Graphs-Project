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
    plt.figure(figsize=(20, 18))
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

    # Initialize queue and visited lists
    queue = [seed]
    visited = set()

    with tqdm(total=max_nodes_to_crawl, desc=f"Creating {strategy} graph from {seed}") as pbar:
        while (len(visited) < max_nodes_to_crawl) and (len(queue) != 0):
            
            # Get the first artist of the queue
            current_artist = queue[0]
            # Pop artist from queue
            queue = queue[1:]

            if current_artist in visited: # Only add edges but not touch the queue
                son_artists = related_artists(sp, current_artist)
                for artists in son_artists:
                    related_id = artists[0]
                    if not G.has_edge(current_artist, related_id):
                        G.add_edge(current_artist, related_id)


            elif current_artist not in visited:
                # Add new artists to visited
                visited.add(current_artist)
                # Create a node with the new artist
                try:
                    G = addNode(G,current_artist)
                except:
                    print("exception:",current_artist)
                # Get its son artists
                son_artists = related_artists(sp, current_artist)

                # Iterate son artists and append to queue if not visited
                for artists in son_artists:
                    related_id = artists[0]
                    
                    # If son artists hasn't been visited yet and its not in the queue
                    if (related_id not in visited) and (related_id not in queue):
                        # Add node
                        G = addNode(G,related_id)
                        # Add to queue
                        if strategy == "BFS":
                            queue.append(related_id)
                        elif strategy == "DFS":
                            queue.insert(0,related_id) #= [related_id] + queue
                    
                    # Add edges
                    if not G.has_edge(current_artist,related_id):
                        G.add_edge(current_artist,related_id)
                        
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
      print('extending graph...')

    for art_id in artists:
      print(art_id)
      artist = sp.artist(art_id)['name']
      print(artist)
      top_tracks = sp.artist_top_tracks(art_id, country='ES')['tracks']
      track_ids = [track['id'] for track in top_tracks]
      audio_features = sp.audio_features(track_ids)
      
      for track, feat in zip(top_tracks, audio_features):
        song_name.append(track['name'])
        song_id.append(track['id'])
        popularity.append(track['popularity'])

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
    # CLIENT_ID = "59435f3767f5407395e8a21c91f1b719"
    # CLIENT_ID = "1bc3dfa825e14b1c9e79c0a5ad59d3d8"
    CLIENT_ID = "c2530bbdac80448191d16672a06e625e" # amelia new
    
    # CLIENT_SECRET = "aa930752eeeb4e1ab36bd7bfff2cd0ff"
    # CLIENT_SECRET = "f3b8b67e5ce74c199b0292e40750fe58"
    CLIENT_SECRET = "b395386b3d894efc98819a4c9f6e3a7c"

    auth_manager = SpotifyClientCredentials (client_id = CLIENT_ID, client_secret = CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Get the artist ID
    seed = search_artist(sp,"Taylor Swift")

    # Create and visualize the BFS graph
    #gb = crawler(sp, seed, max_nodes_to_crawl=100, strategy="BFS", out_filename="./graphs/gB")
    #visualize_graph(gb, title="BFS Taylor Swift graph")

    # Create and visualize the DFS graph
    # gd = crawler(sp, seed, max_nodes_to_crawl=100, strategy="DFS", out_filename="./graphs/gD")
    # visualize_graph(gd, title="DFS Taylor Swift graph")

    gb = nx.read_graphml("./graphs/gB")
    gd = nx.read_graphml("./graphs/gD")

    gproba = crawler(sp, seed, max_nodes_to_crawl=10, strategy="BFS", out_filename="./graphs/gproba")

    # Obtain dataset of songs from artists of previous graphs
    # D = get_track_data(sp, graphs=[gb,gd], out_filename="gB_TaylorSwift")
    # D = get_track_data(sp, graphs=[gd], out_filename="gB_TaylorSwift")

    # Create BFS graph for Pastel Ghost
    #seed = search_artist(sp,"Pastel Ghost")
    #hb = crawler(sp, seed, max_nodes_to_crawl=100, strategy="BFS", out_filename="./graphs/hB")
    #visualize_graph(hb, title="BFS Pastel Ghost graph")

    # ------------------- END OF MAIN ------------------------ #
