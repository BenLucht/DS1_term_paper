# dunn index not in sklearn, found at https://github.com/jqmviegas/jqm_cvi
#from dunn import dunn, dunn_fast
import pandas as pd
from sklearn import metrics

def get_indices(data, labels):
  '''
  data:     n x m matrix of data (n observations, m features)
  labels:   list of n cluster labels
  '''
  indices = pd.DataFrame(columns=['Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette', 'Dunn'])
      
  ch = metrics.calinski_harabasz_score(data, labels)
  db = metrics.davies_bouldin_score(data, labels)
  si = metrics.silhouette_score(data, labels)
  #dn = dunn_fast(data, labels)

  indices = indices.append({'Calinski-Harabasz': ch,
                            'Davies-Bouldin': db,
                            'Silhouette': si,
                            'Dunn': 0}, 
                            ignore_index=True)

  return indices

def get_all_indices(algorithm, data, parameter_range):
    '''
    algorithm:        the function performing clustering, taking data, one parameter
                      and only default parameters, returning list of labels
    data:             n x m matrix of data (n observations, m features)
    parameter_range:  list of values to iterate over
    '''

    indices_list = []
    indices_df =  pd.DataFrame(columns=['Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette', 'Dunn'])
    
    for p in parameter_range:
        indices_list.append(get_indices(data, algorithm(data, p, returns='labels', state=0)))
        
    indices_df = pd.concat(indices_list).reset_index(drop=True)
    indices_df['Parameter'] = parameter_range
        
    return indices_df