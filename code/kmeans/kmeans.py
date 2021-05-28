from sklearn.cluster import KMeans
import numpy as np

def cluster_kmeans(data, k, returns='labels', state=None):
  '''
  data:     n x m - matrix of input data (n observations, m features)
  k:        number of clusters
  returns:  'estimator' returns fitted estimator, 
            'labeled' returns labeled data matrix, 
            'labels' returns labels-array only (default)
  state:    random initialization (default 'None')
  '''

  estimator_fit = KMeans(n_clusters=k, random_state=state).fit(data)

  if returns == 'labels':
    return estimator_fit.labels_
  elif returns == 'labeled':
    return np.c_[data, estimator_fit.labels_]
  else:
    return estimator_fit