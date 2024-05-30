import statistics

def dataset_info(df):
  num_songs = len(df)
  num_artists = df['artist_id'].nunique()
  num_albums = df['album_id'].nunique()
  print("Number of songs:", num_songs)
  print("Number of different artists:", num_artists)
  print("Number of different albums:", num_albums)
  return

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
  print("In-degrees:", sorted(in_degrees))

  print("\nOUT-DEGREE:")
  print("Minimum:", min_out_degree)
  print("Maximum:", max_out_degree)
  print("Median:", median_out_degree)
  print("Out-degrees:", sorted(out_degrees))

  return