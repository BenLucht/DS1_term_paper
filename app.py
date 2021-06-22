# pylint: disable=E0611
# pylint: disable=F0401
# disables pylint errors for no name in module and unable to import

import streamlit as st
import pandas as pd
from code.kmeans.kmeans import cluster_kmeans
from code.affinity.affinity import cluster_affinity
from code.meanshift.meanshift import cluster_meanshift
from code.spectral.spectral import cluster_spectral
from code.plotting.plotting import plot_tsne_2d, tsne_transform
from code.evaluation.evaluation import get_indices

st.set_page_config(
    page_title="Clustering Term Paper",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
customers = customers[['gender', 'age', 'income', 'spending_score']]
def convert_gender(x):
    if x == 'Male':
        return 0
    else:
        return 1
    
customers['gender'] = customers['gender'].map(convert_gender)

housing = pd.read_csv('data/Boston-house-price-data.csv')

redwine = pd.read_csv('data/winequality-red.csv')
redwine = redwine[redwine.columns[:-1]]

# calculate cached version of tsne transformed data 
# to speed up interactions
seeds_tsne = tsne_transform_2(seeds[seeds.columns[:-1]], state=0)
customers_tsne = tsne_transform_2(customers, state=0)
housing_tsne = tsne_transform_2(housing, state=0)
redwine_tsne = tsne_transform_2(redwine, state=0)


def get_slider(selected_data, option):
    if selected_data == 'Seeds':
        kmvals = [2, 8, 3]
        spvals = [2, 8, 3]
        afvals = [-1000, -50, -250]
        msvals = [1, 6, 2]
    elif selected_data == 'Mall Customers':
        kmvals = [2, 8, 2]
        spvals = [2, 8, 3]
        afvals = [-10000, -1000, -7000]
        msvals = [10, 50, 22]
    elif selected_data == 'House Pricing':
        kmvals = [2, 8, 4]
        spvals = [2, 8, 3]
        afvals = [-100000, -1000, -47000]
        msvals = [80, 150, 120]
    elif selected_data == 'Wine Quality':
        kmvals = [2, 8, 6]
        spvals = [2, 8, 3]
        afvals = [-100000, -1000, -7000]
        msvals = [10, 40, 22]

    if option == 'K-Means':
        return st.slider('How many clusters would you like?', kmvals[0], kmvals[1], kmvals[2], key='kmeans_regular')
    elif option == 'Spectral Clustering':
        return st.slider('How many clusters would you like?', spvals[0], spvals[1], spvals[2], key='spectral_regular')
    elif option == 'Affinity Propagation':
        return st.slider('Which preference would you like to choose?', afvals[0], afvals[1], afvals[2], key='affinity_regular')
    elif option == 'Mean Shift':
        return st.slider('Which bandwidth would you like to choose?', msvals[0], msvals[1], msvals[2], key='meanshift_regular')

def get_default_parameters(selected_data, algorithm):
    if selected_data == 'None':
        return [1, 10, 1]

    if selected_data == 'Seeds': 
        return {'kmeans': [2, 8, 3],
        'spectral': [2, 8, 3],
        'affinity': [-1000, -50, -250],
        'meanshift': [1, 6, 2]}[algorithm]
    elif selected_data == 'Mall Customers': 
        return {'kmeans': [2, 8, 2],
        'spectral': [2, 8, 3],
        'affinity': [-10000, -1000, -7000],
        'meanshift': [10, 50, 22]}[algorithm]
    elif selected_data == 'House Pricing':
        return {'kmeans': [2, 8, 4],
        'spectral': [2, 8, 3],
        'affinity': [-100000, -1000, -47000],
        'meanshift': [80, 150, 120]}[algorithm]
    elif selected_data == 'Wine Quality':
        return {'kmeans': [2, 8, 6],
        'spectral': [2, 8, 3],
        'affinity': [-100000, -1000, -7000],
        'meanshift': [10, 40, 22]}[algorithm]

def get_data(mode):
    if mode == 'regular':
        if selected_data == 'Seeds':
            return seeds, seeds_tsne
        elif selected_data == 'Mall Customers':
            return customers, customers_tsne
        elif selected_data == 'House Pricing':
            return housing, housing_tsne
        elif selected_data == 'Wine Quality':
            return redwine, redwine_tsne
    else:
        if selected_data_compare == 'Seeds':
            return seeds, seeds_tsne
        elif selected_data_compare == 'Mall Customers':
            return customers, customers_tsne
        elif selected_data_compare == 'House Pricing':
            return housing, housing_tsne
        elif selected_data_compare == 'Wine Quality':
            return redwine, redwine_tsne

def run_clustering(option, parameter, mode):

    X, X_plot = get_data(mode)

    if option == 'K-Means':
        cluster_labels = cluster_kmeans(X, parameter, state=0)
    elif option == 'Spectral Clustering':
        cluster_labels = cluster_spectral(X, parameter)
    elif option == 'Affinity Propagation':
        cluster_labels = cluster_affinity(X, parameter)
    elif option == 'Mean Shift':
        cluster_labels = cluster_meanshift(X, parameter)

    clusters_plot = plot_tsne_2d(X_plot, cluster_labels, title='', size=(12, 12), state=0, returns='fig')

    with st.spinner('Plotting data ...'):
        st.pyplot(clusters_plot)

    return cluster_labels

# CONTROLS
with st.sidebar.beta_expander(label='Regular', expanded=True):
    selected_data = st.selectbox(
        'Which dataset would you like to choose?',
        ('None', 'Seeds', 'Mall Customers', 'House Pricing', 'Wine Quality'), key='select_data_regular')

    option = st.selectbox(
        'Which algorithm would you like to choose?',
        ('None', 'K-Means', 'Mean Shift', 'Spectral Clustering', 'Affinity Propagation'), key='option_regular')

    parameter = get_slider(selected_data, option)

    run_regular = st.button('Calculate!')

with st.sidebar.beta_expander(label='Comparison', expanded=False):
    selected_data_compare = st.selectbox(
        'Which dataset would you like to choose?',
        ('None', 'Seeds', 'Mall Customers', 'House Pricing', 'Wine Quality'), key='select_data_comparison')

    kmvals = get_default_parameters(selected_data_compare, 'kmeans')
    spvals = get_default_parameters(selected_data_compare, 'spectral')
    afvals = get_default_parameters(selected_data_compare, 'affinity')
    msvals = get_default_parameters(selected_data_compare, 'meanshift')

    st.write('K-Means')
    kmeans_parameter = st.slider('How many clusters would you like?', kmvals[0], kmvals[1], kmvals[2], key='kmeans_comparison')
    st.write('Mean Shift')
    meanshift_parameter = st.slider('Which bandwidth would you like to choose?', msvals[0], msvals[1], msvals[2], key='meanshift_comparison')
    st.write('Affinity Propagation')
    affinity_parameter = st.slider('Which preference would you like to choose?', afvals[0], afvals[1], afvals[2], key='affinity_comparison')
    st.write('Spectral Clustering')
    spectral_parameter = st.slider('How many clusters would you like?', spvals[0], spvals[1], spvals[2], key='spectral_comparison')
    
    run_comparison = st.button('Compare!')

# if selected_data != 'None':
#     st.write('You selected:', selected_data)

#     if option == 'None':
#         st.write('Please select an algorithm.')

#     elif option != 'None':
#         st.write('You selected:', option)
        
# else:
#     st.write('Please select a dataset.')



# Main Panel

if run_regular:
    run_clustering(option, parameter, 'regular')

if run_comparison:
    col1, col2, col3, col4 = st.beta_columns(4)
    X, X_plot = get_data('comparison')
    with col1:
        st.header("K-Means")
        kmeans_labels = run_clustering('K-Means', kmeans_parameter, 'comparison')
    with col2:
        st.header("Mean Shift")
        meanshift_labels = run_clustering('Mean Shift', meanshift_parameter, 'comparison')
    with col3:
        st.header("Affinity")
        affinity_labels = run_clustering('Affinity Propagation', affinity_parameter, 'comparison')
    with col4:
        st.header("Spectral")
        spectral_labels = run_clustering('Spectral Clustering', spectral_parameter, 'comparison')


    # make comparison table
    indices_list = []
    indices_df =  pd.DataFrame(columns=['Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette', 'Dunn'])
    
    indices_list.append(get_indices(X, kmeans_labels))
    indices_list.append(get_indices(X, meanshift_labels))
    indices_list.append(get_indices(X, affinity_labels))
    indices_list.append(get_indices(X, spectral_labels))
        
    indices_df = pd.concat(indices_list).reset_index(drop=True)
    indices_df['Algorithm'] = ['K-Means', 'Mean Shift', 'Affinity', 'Spectral']
    indices_df = indices_df.set_index('Algorithm')
    indices_df = indices_df.T

    st.table(indices_df)
