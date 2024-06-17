#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from itertools import combinations
from math import comb
import numpy as np
import matplotlib.pyplot as plt
import math
import operator

class parenclitic_hypergraph:
    
    def __init__(self, f_mat, new_datapoint, dist_type, \
                 theta, uniform_size, p=10):
        # Input:
        # f_mat: data matrix (num features x num instances), (n_features x N)
        # new_datapoint: single data point either normal or abnormal with n_features
        
        # dist_type: for our analysis "min" is to be set
        # but other distrances like mean, max, percentile-median can also be given as input

        # theta: threshold considered for each datapoint
        # uniform_size: cardinality of each hyperedge
        
        self.n_features = f_mat.shape[0]
        self.N = f_mat.shape[1]
        self.f_mat = f_mat
        
        self.dist_type = dist_type
        if self.dist_type=="percentile_median":
            self.p = p
            # else ignore the parameter 'p'
        
        self.theta = theta
        self.uniform_size = uniform_size
        self.new_datapoint = new_datapoint
        
        # Initializes the dictionary whose "keys" being all the possible hyperedges
        # "values" being count of number of times a feature hyperedge(i,j,k)
        # has a random number (identity m) with maximum deviation from N points.
        self.L = {h: 0 for h in self.k_uniform_hyperedges()}
        
        
    def k_uniform_hyperedges(self):
        # This function is used to generalize to k-uniform hypergraph
        # where k is not just equal to 3 referring to triadic interaction
        
        # Inputs:
        # n_features: no. of features of the data
        # uniform_size: size of hyperedges of k-uniform hypergraph
    
        # Outputs:
        # list of all k-uniform hyperedges
    
        return combinations(range(self.n_features), self.uniform_size)
    
            
    def inter_cluster_distance(self, distance_list):
        # Computes the inter-cluster distances between two clusters
        # singleton cluster which is a random number with n_features "m_random"
        
        # Single linkage
        if self.dist_type == "min":
            return np.min(distance_list)
    
        # Complete linkage
        elif self.dist_type == "max":
            return np.max(distance_list)
    
        # Average linkage
        elif self.dist_type == "mean":
            return np.mean(distance_list)

        # Sorted dist Percentile median
        # robust against noise and outliers
        elif self.dist_type == "percentile_median":
            d_p = np.percentile(np.sort(distance_list), self.p)
            D_p = distance_list[distance_list<=d_p]
            return np.median(D_p)
    
        else:
            print(" Error: type = min/max/mean/percentile_median ")

    
    def compute_pairwise_distances(self, h, m_random):
        # computing pairwise distances between N points and 1 random point 
        # in (fi, fj, fk, ...) k-projected feature space.
    
        # Inputs:
        # h: hyperedge with its elements being indices (f1, f2, f3, ...)
        # m_random: uniform random number belonging to R^(n_features)
    
        # Output:
        # dist: vector of dim(N). pairwise distance between N individuals and
        # random number m in the k-projected feature space (f1, f2, f3, ...) 
    
        # distance between 1 projected random point and N projected points

        dist = np.sqrt( np.sum(  ( self.f_mat[np.array(h),:].T - m_random[np.array(h)] )**2, \
                               axis=1   ))
        
        return dist

    def deviation(self):
        
        for h in self.k_uniform_hyperedges():
            d_list = self.compute_pairwise_distances(h, self.new_datapoint)
            dm = self.inter_cluster_distance(d_list) # scalar
            if dm>self.theta:
                #print("hit")
                self.L[h]+=1
        
        return self.L
    


# In[ ]:





# In[ ]:


# import required breast cancer dataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[ ]:


# collect all the data

cancer_dict = load_breast_cancer()

#### NOTE. ####
# do not normalise and split but first split the data, then normalise on training set
# New data:
# initialise random seed
# First we sample: 200 benign, 200 malign
# Training set: 150 benign, 150 malign, use normalisation
# Test set: 50 benign, 50 malignant

# Parenclitic, only 150 beningn training,
# Test : 50 benign, 50 malignant

data_benign = cancer_dict["data"][cancer_dict.target==0]
data_malignant = cancer_dict["data"][cancer_dict.target==1]


# In[ ]:


# function for parallelising
import multiprocess
from functools import partial

def extract_hedges(arg, f_mat, new_datapoint, dist_type,\
                    uniform_size):
    theta = arg    

    class_obj = parenclitic_hypergraph(f_mat, new_datapoint, dist_type, \
                 theta, uniform_size)

    L = class_obj.deviation()

    return list(L.values()) 


# In[ ]:





# In[ ]:


## Initialise parameters

dist_type='min'
n_features = len(cancer_dict.feature_names)

# threshold values 
init_theta = 0.005
fin_theta = 1.7
step_theta = 0.005

theta_vals = np.arange(init_theta, fin_theta+step_theta, step_theta)

n_realizations = 100
seeds = range(100, 100+n_realizations)


# In[ ]:


# save the results
import os
data_path = "./data/"
if not os.path.isdir(data_path):
    os.mkdir(data_path)


# In[ ]:


for uniform_size in [2,3]:
    print("Uniform size = %d"%uniform_size)

    n_comb = math.comb(n_features, uniform_size)

    # keep the number of test data same
    # for both malignant and benign classes.
    num_test = 50
        
    # Initialise the matrix
    # use np int8 to minimize the memory usage
    L_matrix_benign = np.zeros((n_realizations, num_test,\
                         len(theta_vals), n_comb),dtype=np.int8)
    L_matrix_malignant = np.zeros((n_realizations, num_test,\
                         len(theta_vals), n_comb),dtype=np.int8)

    for rz_ind in range(n_realizations):
        if (rz_ind+1)%10==0:
            print(rz_ind+1)

        # initialise the random seed
        np.random.seed(seeds[rz_ind])
        # Shuffle the range of integers
        # and select 200 datapoints from data_benign, data_malignant

        indices = np.random.permutation(data_benign.shape[0])[:200]
        # Select only those rows belonging to the shuffled indices
        data_benign_sampled = data_benign[indices, :]

        indices = np.random.permutation(data_malignant.shape[0])[:200]
        # Select only those rows belonging to the shuffled indices
        data_malignant_sampled = data_malignant[indices, :]

        # Split the data into training and testing sets, consider many realizations
        data_benign_train, data_benign_test = train_test_split(data_benign_sampled, \
                                                        test_size=0.25, random_state=seeds[rz_ind])

        data_malignant_train, data_malignant_test = train_test_split(data_malignant_sampled, \
                                                        test_size=0.25, random_state=seeds[rz_ind])

        # Normalise over the entire training set
        # benign_train and malignant_train combined.
        data_train = np.concatenate((data_benign_train,data_malignant_train),axis=0)
        # Shuffle the rows of the combined array
        data_train = data_train[np.random.permutation(data_train.shape[0]), :]

        # Normalise "fit" over the training set
        scaler = MinMaxScaler()

        scaler.fit(data_train)
        data_benign_train = scaler.transform(data_benign_train)


        # Finally normalise transform on test set
        data_matrix_benign = scaler.transform(data_benign_test)
        data_matrix_malignant = scaler.transform(data_malignant_test)

        for i in range(num_test):
            # collect hyperedges for benign
            paral_func1 = partial( extract_hedges, f_mat=data_benign_train.T, \
                                new_datapoint = data_matrix_benign[i],
                                dist_type=dist_type, uniform_size=uniform_size)

            pool = multiprocess.Pool( 80 )
            L_matrix_benign[rz_ind, i,:, :] = pool.map(paral_func1, theta_vals)
            pool.close()
            pool.join()
            
            # collect hyperedges for malignant
            paral_func2 = partial( extract_hedges, f_mat=data_benign_train.T, \
                                new_datapoint = data_matrix_malignant[i],
                                dist_type=dist_type, uniform_size=uniform_size)

            pool = multiprocess.Pool( 80 )
            L_matrix_malignant[rz_ind, i,:, :] = pool.map(paral_func2, theta_vals)
            pool.close()
            pool.join()

    np.savez(data_path+"extract_hedges_us%d.npz"%uniform_size, \
             benign_test = L_matrix_benign, \
             malignant_test = L_matrix_malignant)
    print("")


# In[ ]:




