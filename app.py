# pylint: disable=E0611
# pylint: disable=F0401
# disables pylint errors for no name in module and unable to import

import streamlit as st
import pandas as pd
from code.kmeans.kmeans import cluster_kmeans
from code.meanshift.meanshift import cluster_meanshift
from code.plotting.plotting import plot_tsne_2d

# LOAD DATA
seeds = pd.read_csv('data/seeds_dataset.txt', 
                    sep='\t', 
                    names=['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry_coeff', 'length_groove', 'label']
                   )

customers = pd.read_csv('data/Mall_Customers.csv')
customers.columns = ['id', 'gender', 'age', 'income', 'spending_score']
housing = pd.read_csv('data/Boston-house-price-data.csv')
redwine = pd.read_csv('data/winequality-red.csv')

# CONTROLS
selected_data = st.sidebar.selectbox(
    'Which dataset would you like to choose?',
    ('None', 'seeds', 'mall', 'house pricing', 'wine'))

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
    # set bandwidth parameter ranges
    if selected_data == 'seeds':
        min_bandwidth = 1
        max_bandwidth = 6
        default_bandwidth = 2
    elif selected_data == 'mall':
        min_bandwidth = 10
        max_bandwidth = 50
        default_bandwidth = 22
    elif selected_data == 'house pricing':
        min_bandwidth = 80
        max_bandwidth = 150
        default_bandwidth = 120
    elif selected_data == 'wine':
        min_bandwidth = 10
        max_bandwidth = 40
        default_bandwidth = 22
    bw = st.sidebar.slider('Which bandwidth would you like to choose?', min_bandwidth, max_bandwidth, default_bandwidth)
    st.sidebar.write("I select a bandwidth of ", bw)

    if st.sidebar.button('Calculate!'):
        if selected_data == 'seeds':
            X = seeds[seeds.columns[:-1]]
        elif selected_data == 'mall':
            X = customers[['age', 'income', 'spending_score']]
        elif selected_data == 'house pricing':
            X = housing
        elif selected_data == 'wine':
            X = redwine

        meanshift_labels = cluster_meanshift(X, bw)

        clusters_plot = plot_tsne_2d(X, meanshift_labels, title='', size=(12, 12), state=0, returns='fig')

        with st.spinner('Plotting data ...'):
            st.pyplot(clusters_plot)

    else:
        st.write('Please specify your desired parameters.')


elif option == 'spectral':
  st.write('Implementation coming soon!')

elif option == 'affinity prop':
  st.write('Implementation coming soon!')