import pandas as pd
from kmeans.kmeans import cluster_kmeans

seeds = pd.read_csv('../data/seeds_dataset.txt',
                    sep='\t', 
                    names=['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry_coeff', 'length_groove', 'label']
                   )

seeds_labeled = cluster_kmeans(seeds[seeds.columns[:-1]], 3, returns='labeled', state=0)

print(seeds_labeled)