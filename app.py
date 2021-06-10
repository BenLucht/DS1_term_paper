# pylint: disable=E0611
# pylint: disable=F0401
# disables pylint errors for no name in module and unable to import

import streamlit as st
import pandas as pd
from code.kmeans.kmeans import cluster_kmeans
from code.affinity.affinity import cluster_affinity
from code.meanshift.meanshift import cluster_meanshift
from code.plotting.plotting import plot_tsne_2d, tsne_transform

# cached wrapper for tsne_transform function
@st.cache
def tsne_transform_2(data, state=None):
    print('transforming')
    return tsne_transform(data, state=state)

# LOAD DATA
seeds = pd.read_csv('data/seeds_dataset.txt', 
                    sep='\t', 
                    names=['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry_coeff', 'length_groove', 'label']
                   )
seeds = seeds[seeds.columns[:-1]]
customers = pd.read_csv('data/Mall_Customers.csv')
customers.columns = ['id', 'gender', 'age', 'income', 'spending_score']
housing = pd.read_csv('data/Boston-house-price-data.csv')
redwine = pd.read_csv('data/winequality-red.csv')

# calculate cached version of tsne transformed data 
# to speed up interactions
seeds_tsne = tsne_transform_2(seeds[seeds.columns[:-1]], state=0)
customers_tsne = tsne_transform_2(customers[['age', 'income', 'spending_score']], state=0)
housing_tsne = tsne_transform_2(housing, state=0)
redwine_tsne = tsne_transform_2(redwine, state=0)

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
            X = seeds
            X_plot = seeds_tsne
        elif selected_data == 'Mall Customers':
            X = customers[['age', 'income', 'spending_score']]
            X_plot = customers_tsne
        elif selected_data == 'House Pricing':
            X = housing
            X_plot = housing_tsne
        elif selected_data == 'Wine Quality':
            X = redwine
            X_plot = redwine_tsne

        kmeans_labels = cluster_kmeans(X, n, state=0)

        clusters_plot = plot_tsne_2d(X_plot, kmeans_labels, title='', size=(12, 12), state=0, returns='fig')

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
            X = seeds
            X_plot = seeds_tsne
        elif selected_data == 'Mall Customers':
            X = customers[['age', 'income', 'spending_score']]
            X_plot = customers_tsne
        elif selected_data == 'House Pricing':
            X = housing
            X_plot = housing_tsne
        elif selected_data == 'Wine Quality':
            X = redwine
            X_plot = redwine_tsne

        meanshift_labels = cluster_meanshift(X, bw)

        clusters_plot = plot_tsne_2d(X_plot, meanshift_labels, title='', size=(12, 12), state=0, returns='fig')

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
            X = seeds
            X_plot = seeds_tsne
        elif selected_data == 'Mall Customers':
            X = customers[['age', 'income', 'spending_score']]
            X_plot = customers_tsne
        elif selected_data == 'House Pricing':
            X = housing
            X_plot = housing_tsne
        elif selected_data == 'Wine Quality':
            X = redwine
            X_plot = redwine_tsne

        affinity_labels = cluster_affinity(X, bw)

        clusters_plot = plot_tsne_2d(X_plot, affinity_labels, title='', size=(12, 12), state=0, returns='fig')

        with st.spinner('Plotting data ...'):
            st.pyplot(clusters_plot)

    else:
        st.write('Please specify your desired parameters.  ')