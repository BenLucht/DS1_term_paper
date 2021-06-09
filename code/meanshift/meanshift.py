from sklearn.cluster import MeanShift
import numpy as np

def cluster_meanshift(data, bandwidth, returns='labels'):
  '''
  data:       n x m - matrix of input data (n observations, m features)
  bandwidth:  kernel's bandwidth or radius
  returns:    'estimator' returns fitted estimator, 
              'labeled' returns labeled data matrix, 
              'labels' returns labels-array only (default)
  '''

  estimator_fit = MeanShift(bandwidth=bandwidth).fit(data)

  if returns == 'labels':
    return estimator_fit.labels_
  elif returns == 'labeled':
    return np.c_[data, estimator_fit.labels_]
  else:
    return estimator_fit