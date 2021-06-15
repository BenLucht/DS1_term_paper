import streamlit as st
import pandas as pd
from code.kmeans.kmeans import cluster_kmeans
from code.affinity.affinity import cluster_affinity
from code.meanshift.meanshift import cluster_meanshift
from code.plotting.plotting import plot_tsne_2d

# LOAD DATA
seeds = pd.read_csv('data/seeds_dataset.txt',
                    sep='\t',
                    names=['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry_coeff', 'length_groove',
                           'label']
                    )

customers = pd.read_csv('data/Mall_Customers.csv')
customers.columns = ['id', 'gender', 'age', 'income', 'spending_score']

housing = pd.read_csv('data/Boston-house-price-data.csv')
redwine = pd.read_csv('data/winequality-red.csv')


@st.cache
def mydata(arg1):
    myalgo = arg1
    return myalgo

@st.cache
def myalgo(arg1):
    myalgo = arg1
    return myalgo



def kmeans(X, selected_data,  buttontext='Calculate!', j = "", parameter=None):
    if parameter == None:
        n = st.sidebar.slider('How many clusters would you like?'+j, 2, 8, 3)
        st.sidebar.write("I want ", n, 'clusters.')

        if st.sidebar.button(buttontext):


            kmeans_labels = cluster_kmeans(X, n, state=0)

            clusters_plot = plot_tsne_2d(X, kmeans_labels, title='kmeans', size=(12, 12), state=0, returns='fig')

            with st.spinner('Plotting data ...'):
                st.pyplot(clusters_plot)
                return n

        else:
            st.write('Please specify your desired parameters.')
            return n
    else:
        kmeans_labels = cluster_kmeans(X, parameter, state=0)
        clusters_plot = plot_tsne_2d(X, kmeans_labels, title='kmeans', size=(12, 12), state=0, returns='fig')
        st.pyplot(clusters_plot)
        return parameter

def meanshift(X, selected_data, buttontext='Calculate!', parameter=None):
    if parameter ==None:
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

         if st.sidebar.button(buttontext):


             meanshift_labels = cluster_meanshift(X, bw)

             clusters_plot = plot_tsne_2d(X, meanshift_labels, title='meanshift', size=(12, 12), state=0, returns='fig')

             with st.spinner('Plotting data ...'):
                 st.pyplot(clusters_plot)

         else:
             st.write('Please specify your desired parameters.')
         return bw
    else:
        meanshift_labels = cluster_meanshift(X, parameter)
        clusters_plot = plot_tsne_2d(X, meanshift_labels, title='meanshift', size=(12, 12), state=0, returns='fig')
        st.pyplot(clusters_plot)
        return parameter



def spectralClustering(X, selected_data, buttontext='Calculate!', parameter=None):
    st.write('Implementation coming soon!')
    return None

def affinityPropagation(X, selected_data, buttontext='Calculate!', parameter=None):
    if parameter == None:
        bw = st.sidebar.slider('Which preference would you like to choose for the comparison?', -10000, -50, -7000)
        st.sidebar.write("I select a preference of ", bw, "to compare")
        if st.sidebar.button(buttontext):

            affinity_labes = cluster_affinity(X, bw)

            clusters_plot = plot_tsne_2d(X, affinity_labes, title='', size=(12, 12), state=0, returns='fig')

            with st.spinner('Plotting data ...'):
                st.pyplot(clusters_plot)

        else:
            st.write('Please specify your desired parameters.  ')
        return bw
    else:
        affinity_labes = cluster_affinity(X, parameter)
        clusters_plot = plot_tsne_2d(X, affinity_labes, title='', size=(12, 12), state=0, returns='fig')
        st.pyplot(clusters_plot)
        return parameter


def getData(selected_data):
    mydata(selected_data)
    if selected_data == 'Seeds':
        X = seeds[seeds.columns[:-1]]
    elif selected_data == 'Mall Customers':
        X = customers[['age', 'income', 'spending_score']]
    elif selected_data == 'House Pricing':
        X = housing
    elif selected_data == 'Wine Quality':
        X = redwine

    return X



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
        firstinput = 1
        if option == 'K-Means':
            firstinput = kmeans(getData(selected_data), selected_data)
        elif option == 'Mean Shift':
            firstinput =meanshift(getData(selected_data), selected_data)
        elif option == 'Spectral Clustering':
            firstinput =spectralClustering(getData(selected_data), selected_data)
        elif option == 'Affinity Propagation':
            firstinput = affinityPropagation(getData(selected_data), selected_data)

        option3 = st.sidebar.selectbox('Which algorithm would you like to compare?',
                                       ('None', 'Mean Shift', 'K-Means', 'Spectral Clustering', 'Affinity Propagation'))

        if option3 != "None":


            if option3 == 'K-Means':
                if option == 'K-Means':
                    firstinput = kmeans(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Mean Shift':
                    firstinput = meanshift(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Spectral Clustering':
                    firstinput = spectralClustering(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Affinity Propagation':
                    firstinput = affinityPropagation(getData(selected_data), selected_data, parameter=firstinput)
                kmeans(getData(selected_data), selected_data, "Compare?")

            elif option3 == 'Mean Shift':
                if option == 'K-Means':
                    firstinput = kmeans(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Mean Shift':
                    firstinput = meanshift(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Spectral Clustering':
                    firstinput = spectralClustering(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Affinity Propagation':
                    firstinput = affinityPropagation(getData(selected_data), selected_data, parameter=firstinput)
                meanshift(getData(selected_data), selected_data, "Compare?")

            elif option3 == 'Spectral Clustering':
                if option == 'K-Means':
                    firstinput = kmeans(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Mean Shift':
                    firstinput = meanshift(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Spectral Clustering':
                    firstinput = spectralClustering(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Affinity Propagation':
                    firstinput = affinityPropagation(getData(selected_data), selected_data, parameter=firstinput)
                spectralClustering(getData(selected_data), selected_data, "Compare?")
            elif option3 == 'Affinity Propagation':
                if option == 'K-Means':
                    firstinput = kmeans(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Mean Shift':
                    firstinput = meanshift(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Spectral Clustering':
                    firstinput = spectralClustering(getData(selected_data), selected_data, parameter=firstinput)
                elif option == 'Affinity Propagation':
                    firstinput = affinityPropagation(getData(selected_data), selected_data, parameter=firstinput)

                affinityPropagation(getData(selected_data), selected_data, "Compare?")




else:
    st.write('Please select a dataset.')

