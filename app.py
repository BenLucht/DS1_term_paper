# pylint: disable=E0611
# pylint: disable=F0401
# disables pylint errors for no name in module and unable to import

import streamlit as st
import pandas as pd
from code.kmeans.kmeans import cluster_kmeans
from code.affinity.affinity import cluster_affinity
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
    ('None', 'Seeds', 'Mall Customers', 'House Pricing', 'Wine Quality'))

option = st.sidebar.selectbox(
    'Which algorithm would you like to choose?',
    ('None', 'K-Means', 'Mean Shift', 'Spectral Clustering', 'Affinity Propagation'))

if selected_data != 'None':
    st.write('You selected:', selected_data)

    if option == 'None':
        st.write('Please select an algorithm.')

    elif option != 'None':
        st.write('You selected:', option)
        
else:
    st.write('Please select a dataset.')


# LOGIC
if option == 'K-Means':

    n = st.sidebar.slider('How many clusters would you like?', 2, 8, 3)
    st.sidebar.write("I want ", n, 'clusters.')

    if st.sidebar.button('Calculate!'):
        if selected_data == 'Seeds':
            X = seeds[seeds.columns[:-1]]
        elif selected_data == 'Mall Customers':
            X = customers[['age', 'income', 'spending_score']]
        elif selected_data == 'House Pricing':
            X = housing
        elif selected_data == 'Wine Quality':
            X = redwine

        kmeans_labels = cluster_kmeans(X, n, state=0)

        clusters_plot = plot_tsne_2d(X, kmeans_labels, title='', size=(12, 12), state=0, returns='fig')

        with st.spinner('Plotting data ...'):
            st.pyplot(clusters_plot)

    else:
        st.write('Please specify your desired parameters.')

elif option == 'Mean Shift':
    # set bandwidth parameter ranges
    if selected_data == 'Seeds':
        min_bandwidth = 1
        max_bandwidth = 6
        default_bandwidth = 2
    elif selected_data == 'Mall Customers':
        min_bandwidth = 10
        max_bandwidth = 50
        default_bandwidth = 22
    elif selected_data == 'House Pricing':
        min_bandwidth = 80
        max_bandwidth = 150
        default_bandwidth = 120
    elif selected_data == 'Wine Quality':
        min_bandwidth = 10
        max_bandwidth = 40
        default_bandwidth = 22
    bw = st.sidebar.slider('Which bandwidth would you like to choose?', min_bandwidth, max_bandwidth, default_bandwidth)
    st.sidebar.write("I select a bandwidth of ", bw)

    if st.sidebar.button('Calculate!'):
        if selected_data == 'Seeds':
            X = seeds[seeds.columns[:-1]]
        elif selected_data == 'Mall Customers':
            X = customers[['age', 'income', 'spending_score']]
        elif selected_data == 'House Pricing':
            X = housing
        elif selected_data == 'Wine Quality':
            X = redwine

        meanshift_labels = cluster_meanshift(X, bw)

        clusters_plot = plot_tsne_2d(X, meanshift_labels, title='', size=(12, 12), state=0, returns='fig')

        with st.spinner('Plotting data ...'):
            st.pyplot(clusters_plot)

    else:
        st.write('Please specify your desired parameters.')


elif option == 'Spectral Clustering':
  st.write('Implementation coming soon!')

elif option == 'Affinity Propagation':
    bw = st.sidebar.slider('Which preference would you like to choose?', -10000, -50, -7000)
    st.sidebar.write("I select a preference of ", bw)
    if st.sidebar.button('Calculate!'):
        if selected_data == 'Seeds':
            X = seeds[seeds.columns[:-1]]
        elif selected_data == 'Mall Customers':
            X = customers[['age', 'income', 'spending_score']]
        elif selected_data == 'House Pricing':
            X = housing
        elif selected_data == 'Wine Quality':
            X = redwine


    
        affinity_labes = cluster_affinity(X, bw)

        clusters_plot = plot_tsne_2d(X, affinity_labes, title='', size=(12, 12), state=0, returns='fig')

        with st.spinner('Plotting data ...'):
            st.pyplot(clusters_plot)

    else:
        st.write('Please specify your desired parameters.  ')