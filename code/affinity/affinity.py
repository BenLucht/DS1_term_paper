from sklearn.cluster import AffinityPropagation
import numpy as np
def cluster_affinity(data, preference, returns='labels'):
  '''
  data:      panda dataframe
  preference:  numbers of exemplars 
  returns:    'labeled' returns labeled data matrix, 
              'labels' returns labels-array only (default)
  '''

  print(preference)

  af = AffinityPropagation( verbose=False, max_iter=1200).fit(data)

  if returns == 'labels':
    return af.labels_
  elif returns == 'labeled':
    return np.c_[data, af.labels_]
  else:
    return af