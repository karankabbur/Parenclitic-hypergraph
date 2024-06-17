#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import hypernetx as hnx
from hide_warnings import hide_warnings
from itertools import combinations
from sklearn.datasets import load_breast_cancer
import operator

import json


# In[ ]:


@hide_warnings
def get_topldegree_feats(arr_1d, l, n_features, uniform_size, s):
    all_hedges = np.array(list(combinations(range(n_features), uniform_size)))
    hedges = all_hedges[arr_1d!=0]

    hyperdict =[] 
    for edge in hedges:
        hyperdict.append(set(edge))

    H = hnx.Hypergraph(hyperdict)
    
    node_degree_dict = {}
    for node in list(H.nodes()):
        node_degree_dict[node] = H.degree(node, s=s)
        
    sorted_dict = dict(sorted(node_degree_dict.items(), key=operator.itemgetter(1), reverse=True))
    
    # if number of key-value pairs are less than top l values considered.
    if len(sorted_dict.keys())<=l:
        # send the existing dictionary completely
        top_degree_dict = sorted_dict
    else:
        top_degree_dict = dict(list(sorted_dict.items())[0: l]) 
    return top_degree_dict    
    


# In[ ]:





# In[ ]:


cancer_dict = load_breast_cancer()

n_features = len(cancer_dict.feature_names)

data_path = "./data/"

s = 2
n_features = len(cancer_dict.feature_names)

# consider threshold 0.07 for both 3-uniform hypergraph and network
# set the theta max based on the plots observed from diff max degree(malignant, benign)
theta_max = 0.07

init_theta = 0.005
fin_theta = 1.7
step_theta = 0.005
theta_vals = np.arange(init_theta, fin_theta+step_theta, step_theta)

ind_max = np.argwhere(theta_vals==theta_max)[0][0]


# In[ ]:


for uniform_size in [2,3]:

    data_dict = {}
    print("uniform size: ",uniform_size)
    
    for top_l in [1,2,3,4,5]:
        print("l = ",top_l)
        data_dict["top_%d"%top_l]={}

        # Find top 'l' features at threshold 0.07
        # for each malignant and benign datapoints based on the degree
        for datapoint_type in ["benign_test", "malignant_test"]:
            print(datapoint_type)
            data_dict["top_%d"%top_l][datapoint_type]={}
            
            # 4 d   (n_realizations, n_instances, theta_vals, hyperedges triplets)
            arr4d = np.load(data_path+"extract_hedges_us%d.npz"%uniform_size)[datapoint_type]

            for rz in range(arr4d.shape[0]):

                arr3d = arr4d[rz, :, :, :]
                arr_data_hedges =arr3d[:, ind_max, :]# (n_instances, hyperedges)

                # top l degree dictionary
                # create empty dictionary
                # key: feature, value: empty list
                feat_degree_dict = {}
                for feat in cancer_dict.feature_names:
                    feat_degree_dict[feat] = []

                # loop over each instance
                for i in range(arr_data_hedges.shape[0]):
                    # don't consider if there no hedges at all
                    if np.any(arr_data_hedges[i]):
                        node_deg_dict = get_topldegree_feats(arr_data_hedges[i], top_l,\
                                                             n_features, uniform_size, s)
                        # node_deg_dict has 3 key value pairs
                        for node_ind, value in node_deg_dict.items():
                            feat_degree_dict[cancer_dict.feature_names[node_ind]].append(value)

                # Sort the dictionary in reverse order by the length of the values (lists)
                sorted_dict = dict(sorted(feat_degree_dict.items(), key=lambda x: len(x[1]), reverse=True))

                data_dict["top_%d"%top_l][datapoint_type]["rz_%d"%rz]=sorted_dict


    # Save the dictionary to a JSON file with pretty formatting
    with open(data_path+'top_feat_us%d.json'%uniform_size, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)
    print("")


# In[ ]:





# In[ ]:




