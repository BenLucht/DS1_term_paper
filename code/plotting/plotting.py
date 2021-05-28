from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne_2d(data, labels, title='', size=(12, 12), returns='plot'):
  '''
  data:     n x m matrix of data (n observations, m features)
  labels:   label array for all observations
  title:    title for the plot (default '')
  size:     size-tuple for the plot (width, height) in inches (default (12,12))
  returns:  'plot' returns nothing and just plots
            'fig' returns figure object
  '''

  data_projected_2d = TSNE(n_components=2).fit_transform(data)

  labeled_data = np.c_[data_projected_2d, labels]

  fig = plt.figure(figsize=size)
  ax = fig.add_subplot(111)

  # list of tuples of (color, marker, label) for each possible cluster
  clusters = [
    ('r', 'o', 0), 
    ('b', '^', 1), 
    ('g', 'x', 2), 
    ('m', '.', 3), 
    ('y', '*', 4), 
    ('cyan', '+', 5)
  ]

  # number of clusters
  n = len(set(labels))

  # adds seperate scatter for each cluster 
  for color, marker, label in clusters[0:n]:
    current_cluster = labeled_data[labeled_data[:,-1] == label]

    ax.scatter(
      current_cluster[:, 0], 
      current_cluster[:, 1], 
      c=color, 
      marker=marker
    )

  ax.set_title(title)

  if returns == 'plot':
    plt.show()
  else:
    return fig