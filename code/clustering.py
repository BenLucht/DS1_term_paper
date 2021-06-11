# pylint: disable=F0401

import pandas as pd
from kmeans.kmeans import cluster_kmeans
from plotting.plotting import tsne_transform, plot_tsne_2d, plot_data_3d, plot_indices
from evaluation.evaluation import get_indices, get_all_indices

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
    feature_names=['area', 'perimeter', 'compactness'], 
    title='Seeds', 
    size=(8, 8), 
    azimuth=100 , 
    elevation=40 , 
    returns='plot'
  )

X = tsne_transform(seeds_labeled[:, :-1], state=0)

plot_tsne_2d(X, seeds_labeled[:, -1], title='Seeds', state=0)

print(get_indices(seeds_labeled[:, :-1], seeds_labeled[:, -1]))

plot_indices(get_all_indices(cluster_kmeans, seeds_labeled[:, :-1], [n for n in range(2, 10)]))