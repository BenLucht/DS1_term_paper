from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# list of tuples of (color, marker, label) for each possible cluster
clusters = [
  ('r', 'o', 0), 
  ('b', '^', 1), 
  ('g', 'x', 2), 
  ('m', '.', 3), 
  ('y', '*', 4), 
  ('c', '+', 5),
  ('maroon', '>', 6),
  ('lime', 'D', 7),
  ('gold', 'H', 8),
  ('indigo', '1', 9),
  ('aquamarine', 'v', 10),
]

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


def plot_data_3d(
    data, 
    labels, 
    features=[0, 1, 2], 
    feature_names=['', '', ''], 
    title='', 
    size=(12, 12), 
    azimuth=-60 , 
    elevation=30 , 
    returns='plot'
  ):
  '''
  data:           n x m matrix of data (n observations, m features)
  labels:         label array for all observations
  features:       list of feature column indices (default first three)
  feature_names:  list of feature names (default '')
  title:          title for the plot (default '')
  size:           size-tuple for the plot (width, height) in inches (default (12,12))
  azimuth:        angle around z-axis (default -60)
  elevation:      elevation above 'ground plane' (default 30)
  returns:        'plot' returns nothing and just plots
                  'fig' returns figure object
  '''

  # seems somewhat redundant, but wanted to stay consistent with other functions
  labeled_data = np.c_[data, labels]

  fig = plt.figure(figsize=size)

  # number of clusters
  n = len(set(labels))

  ax = fig.add_subplot(111, projection='3d', azim=azimuth, elev=elevation)

  for color, marker, label in clusters[0:n]:
    current_cluster = labeled_data[labeled_data[:,-1] == label]

    ax.scatter(
      current_cluster[:, 0], 
      current_cluster[:, 1], 
      current_cluster[:, 2], 
      c=color, 
      marker=marker
    )

  ax.set_zlabel(feature_names[0])
  ax.set_ylabel(feature_names[1])
  ax.set_xlabel(feature_names[2])
  ax.set_title('')

  if returns == 'plot':
    plt.show()
  else:
    return fig