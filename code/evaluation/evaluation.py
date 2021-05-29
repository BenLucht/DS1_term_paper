# dunn index not in sklearn, found at https://github.com/jqmviegas/jqm_cvi
#from dunn import dunn, dunn_fast
import pandas as pd
from sklearn import metrics

def get_indices(data, labels):
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