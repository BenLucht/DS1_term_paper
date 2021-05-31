# pylint: disable=E0611
# pylint: disable=F0401
# disables pylint errors for no name in module and unable to import

import streamlit as st
import pandas as pd
from code.kmeans.kmeans import cluster_kmeans
from code.plotting.plotting import plot_tsne_2d

# LOAD DATA
seeds = pd.read_csv('data/seeds_dataset.txt', 
                    sep='\t', 
                    names=['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry_coeff', 'length_groove', 'label']
                   )

customers = pd.read_csv('data/Mall_Customers.csv')
customers.columns = ['id', 'gender', 'age', 'income', 'spending_score']

# CONTROLS
selected_data = st.sidebar.selectbox(
    'Which dataset would you like to choose?',
    ('None', 'seeds', 'mall', 'food', 'wine'))

option = st.sidebar.selectbox(
    'Which algorithm would you like to choose?',
    ('None', 'k-means', 'meanshift', 'spectral', 'affinity prop'))

if selected_data != 'None':
    st.write('You selected:', selected_data)

    if option == 'None':
        st.write('Please select an algorithm.')

    elif option != 'None':
        st.write('You selected:', option)
        
else:
    st.write('Please select a dataset.')


# LOGIC
if option == 'k-means':

  n = st.sidebar.slider('How many clusters would you like?', 2, 8, 3)
  st.sidebar.write("I want ", n, 'clusters.')

  if st.sidebar.button('Calculate!'):
      if selected_data == 'seeds':
          X = seeds[seeds.columns[:-1]]
      elif selected_data == 'mall':
          X = customers[['age', 'income', 'spending_score']]

      kmeans_labels = cluster_kmeans(X, n, state=0)

      clusters_plot = plot_tsne_2d(X, kmeans_labels, title='', size=(12, 12), state=0, returns='fig')

      with st.spinner('Plotting data ...'):
          st.pyplot(clusters_plot)

  else:
      st.write('Please specify your desired parameters.')

elif option == 'meanshift':
  st.write('Implementation coming soon!')

elif option == 'spectral':
  st.write('Implementation coming soon!')

elif option == 'affinity prop':
  st.write('Implementation coming soon!')