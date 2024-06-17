#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import required breast cancer dataset
from sklearn.datasets import load_breast_cancer

import numpy as np
import hypernetx as hnx
from hide_warnings import hide_warnings
from itertools import combinations


# In[2]:


# save properties of each datapoint in a dataframe

cancer_dict = load_breast_cancer()

n_features = len(cancer_dict.feature_names)

data_path = "./data/"


# In[ ]:





# In[3]:


@hide_warnings    
def max_degree(hyperdict, s=2):
    H = hnx.Hypergraph(hyperdict)
    max_deg = max([H.degree(node, s=s) for node in list(H.nodes())])
    return max_deg    
    


# In[4]:


def get_max_degree(args, arr, n_features, uniform_size, s=2):
    data_ind, thresh_ind = args[0], args[1]
    
    # if there are no hyperedges, then degree is zero
    # for all feature nodes
    if ~(np.any(arr[data_ind, thresh_ind])):
        max_deg = 0
    else:
        
        all_hedges = np.array(list(combinations(range(n_features), uniform_size)))
        hedges = all_hedges[arr[data_ind, thresh_ind]!=0]

        hyperdict =[] 
        for edge in hedges:
            hyperdict.append(set(edge))

        # hyperdict is list of sets
        max_deg = max_degree(hyperdict, s)
    
    return max_deg
        
from functools import partial
import multiprocess


# In[5]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[6]:


s=2
n_realizations = 100

for uniform_size in [3]:
    print("Uniform size: ",uniform_size)

    prop_dict={}

    for datapoint_type in ["benign_test","malignant_test"]:
        print(datapoint_type)
        arr = np.load(data_path+"extract_hedges_us%d.npz"%uniform_size)[datapoint_type]

        max_deg_mat = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]))

        for i in range(n_realizations):
            if (i+1)%10==0:
                print(i+1)
            arr_rz = arr[i]
            parallel_compute = partial( get_max_degree, arr=arr_rz, n_features=n_features,
                                   uniform_size=uniform_size, s=s)

            pool = multiprocess.Pool( 80 )

            res = pool.map(parallel_compute, ((i,j) for i in range(arr_rz.shape[0]) for j in range(arr_rz.shape[1])))
            pool.close()
            pool.join()

            max_deg_mat[i,:,:] = np.array(res).reshape((arr_rz.shape[0], arr_rz.shape[1]))

        prop_dict[datapoint_type] = max_deg_mat 

    np.savez(data_path+"maxdeg_us%d.npz"%uniform_size, benign_test = prop_dict["benign_test"],
             malignant_test = prop_dict["malignant_test"])
    print("")


# In[ ]:





# In[ ]:




