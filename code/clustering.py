import pandas as pd
from kmeans.kmeans import cluster_kmeans
from plotting.plotting import plot_tsne_2d, plot_data_3d

seeds = pd.read_csv('../data/seeds_dataset.txt',
                    sep='\t', 
                    names=['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry_coeff', 'length_groove', 'label']
                   )

seeds_labeled = cluster_kmeans(seeds[seeds.columns[:-1]], 3, returns='labeled', state=0)

print(seeds_labeled)

plot_data_3d(
    seeds_labeled[:, :-1], 
    seeds_labeled[:, -1], 
    features=[0, 1, 2], 
    feature_names=['area', 'lenght', 'width'], 
    title='Seeds', 
    size=(8, 8), 
    azimuth=100 , 
    elevation=40 , 
    returns='plot'
  )

plot_tsne_2d(seeds_labeled[:, :-1], seeds_labeled[:, -1], title='Seeds')