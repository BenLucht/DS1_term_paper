from sklearn.cluster import SpectralClustering
import numpy as np

def cluster_spectral(data, k, returns='labels'):
  '''
  data:     n x m - matrix of input data (n observations, m features)
  k:        number of clusters
  returns:  'estimator' returns fitted estimator,
            'labeled' returns labeled data matrix,
            'labels' returns labels-array only (default)
    '''

  spectral = SpectralClustering(n_clusters=k,
                                affinity='nearest_neighbors',
                                n_init=100
                                )

  spectral_labels = spectral.fit_predict(data)

  if returns == 'labels':
    return spectral_labels
  elif returns == 'labeled':
    return np.c_[data, spectral_labels]
  else:
    return spectral