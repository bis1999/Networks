import pandas as pd 
import numpy as np 
import networkx as nx 
from tqdm import tqdm 
from sklearn.metrics import roc_auc_score

from sklearn.metrics import  auc

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler
import random
import itertools
import os
import pickle
import pandas as pd
from tqdm import tqdm
from multiprocess import Pool, cpu_count


def create_observed_set(G,alpha):
    underlying_links = list(G.edges) # G underlying graph
    observed_links = random.sample(underlying_links,int(len(underlying_links)*alpha)) # G' Observed net 0.8*|E|
    
    removed_observed_links = list(set(underlying_links) - set(observed_links)) # 0.2*|E|
    #print("Number of edges in observed_links net: ",len(observed_links))
    #print("Number of unseen edges in observed_links net: ",len(removed_observed_links))
    
    return observed_links,removed_observed_links

def create_training_set(observed_links,alpha):
    training_links = random.sample(observed_links,int(len(observed_links)*alpha)) # G'' training net 0.64|E|
    removed_train_links = list(set(observed_links) - set(training_links)) # 0.16|E| 
    #print("Number of edges in training links net: ",len(training_links))
    #print("Number of unseen edges in training net: ",len(removed_train_links))
    return training_links,removed_train_links



def G_create(G,training_links):
    nodes = list(G.nodes)
    G_train = nx.Graph()
    G_train.add_nodes_from(nodes)
    G_train.add_edges_from(training_links)
    return G_train


def create_training_df(G_train,train_links,missinglinks):
    nodes = list(G_train.nodes)
    
    all_ran_links = itertools.combinations(nodes,2) 
    all_ran_links = [tuple(sorted(t)) for t in all_ran_links]

    links_train = [tuple(sorted(t)) for t in train_links]
    non_links = list(set(all_ran_links) - set(links_train))    
    
    missinglinks = [tuple(sorted(t)) for t in missinglinks]
    
    df = pd.DataFrame()
    df["non_links"] = non_links # X''
    df["Link_present"] = 0
    df.loc[df["non_links"].isin(missinglinks),"Link_present"] =1
    
    return df






def create_cvs(G):
    observed_links,observed_missing_links = create_observed_set(G,0.8)
    train_links,train_missing_links = create_training_set(observed_links,0.8)
    Gho =  G_create(G,observed_links)
    Gtr = G_create(G,train_links)

    df_ho = create_training_df(Gho,observed_links,observed_missing_links)
    df_tr = create_training_df(Gtr,train_links,train_missing_links)


    df_ho_top = top_feats(df_ho,Gho,"non_links")
    df_tr_top = top_feats(df_tr,Gtr,"non_links")
    feature_set = df_tr_top.columns[1:]


    X_train_cv,y_train_cv,X_test_cv,y_test_cv = create_cross_validation(df_tr_top,"Link_present")
    return X_train_cv,y_train_cv,X_test_cv,y_test_cv,df_ho_top,df_tr_top




def top_feats(df,G_top,col):
    #jaccard_coefficient
    all_edges = list(df[col])
    
    
    
    
    
    jacc_coeff_obj = nx.jaccard_coefficient(G_top,all_edges)
    jacc_coeff_edges = []
    for uu,vv,jj in jacc_coeff_obj:
        jacc_coeff_edges.append(jj)
        
    df["score_jaccard"] = jacc_coeff_edges
    
    #adamic_adar_index
    adamic_adar_coeff = nx.adamic_adar_index(G_top,all_edges)
    adamic_adar_score = []
    for uu,vv,jj in adamic_adar_coeff:
        adamic_adar_score.append(jj)
    df["adamic_adar_score"] = adamic_adar_score
    
    
    resource_allocation_index_coeff =nx.resource_allocation_index(G_top,all_edges)
    resource_allocation = []
    for uu,vv,jj in resource_allocation_index_coeff:
        resource_allocation.append(jj)   
    df["resource_allocation"] = resource_allocation
    
    
    
    
    preferential_obj = nx.preferential_attachment(G_top,all_edges)
    preferential_attachment = []
    for uu,vv,jj in preferential_obj:
        preferential_attachment.append(jj)   
    df["preferential_attachment"] = preferential_attachment
    
    
    
    #num_of_neigbours
    num_of_edges = []
    for i in all_edges:
        l=list(nx.common_neighbors(G_top, i[0], i[1]))
        num_of_edges.append(len(l))
    df["num_of_neigbours"] = num_of_edges

    
        
        
    
    # Number of Degrees
    degrees = dict(G_top.degree())
    degree_1 = []
    degree_2 = []

    for i,j in all_edges:
        degree_1.append(degrees[i])
        degree_2.append(degrees[j])
    df["degree_1"] = degree_1
    df["degree_2"] = degree_2
        
    
    
    #Number of triangles
    number_of_triangles =nx.triangles(G_top)
    tri_1 = []
    tri_2 = []
    
    for i,j in all_edges:
        tri_1.append(number_of_triangles[i])
        tri_2.append(number_of_triangles[j])
    df["tri_1"] = tri_1
    df["tri_2"] = tri_2
    
    
    
    #Number of triangles
    page_rank =nx.pagerank(G_top)
    pr_1 = []
    pr_2 = []
    
    for i,j in all_edges:
        pr_1.append(page_rank[i])
        pr_2.append(page_rank[j])
    df["pr_1"] = pr_1
    df["pr_2"] = pr_2
    
    
    #Clustering Coefficients
    clustering_coeff =nx.clustering(G_top)
    clust_1 = []
    clust_2 = []
    
    for i,j in all_edges:
        clust_1.append(clustering_coeff[i])
        clust_2.append(clustering_coeff[j])
    df["clust_1"] = clust_1
    df["clust_2"] = clust_2
    
    
    #Average Neigbourhood degree
    aveg_neigh_deg =nx.clustering(G_top)
    aveg_neigh_deg_1 = []
    aveg_neigh_deg_2 = []
    
    for i,j in all_edges:
        aveg_neigh_deg_1.append(aveg_neigh_deg[i])
        aveg_neigh_deg_2.append(aveg_neigh_deg[j])
    df["aveg_neigh_deg_1"] = aveg_neigh_deg_1
    df["aveg_neigh_deg_2"] = aveg_neigh_deg_2
    
    # Degree centerlaity
    
    degree_centrality =nx.degree_centrality(G_top)
    degree_centrality_1 = []
    degree_centrality_2 = []
    
    for i,j in all_edges:
        degree_centrality_1.append(degree_centrality[i])
        degree_centrality_2.append(degree_centrality[j])
    df["degree_centrality_1"] = degree_centrality_1
    df["degree_centrality_2"] = degree_centrality_2
    
    """
    #EIgen vector centerality
    eigenvector_centrality =nx.eigenvector_centrality(G_top)
    eigenvector_centrality_1 = []
    eigenvector_centrality_2 = []
    
    for i,j in all_edges:
        eigenvector_centrality_1.append(eigenvector_centrality[i])
        eigenvector_centrality_2.append(eigenvector_centrality[j])
    df["eigenvector_centrality_1"] = eigenvector_centrality_1
    df["eigenvector_centrality_2"] = eigenvector_centrality_2"""  
    
    
    
    
    #Closness centerlaity
    closeness_centrality =nx.closeness_centrality(G_top)
    closeness_centrality_1 = []
    closeness_centrality_2 = []
    
    for i,j in all_edges:
        closeness_centrality_1.append(closeness_centrality[i])
        closeness_centrality_2.append(closeness_centrality[j])
    df["closeness_centrality_1"] = closeness_centrality_1
    df["closeness_centrality_2"] = closeness_centrality_2
    
    
    """
    katz_centrality =nx.katz_centrality(G,tol=1e-3,max_iter=100000)
    katz_centrality_1 = []
    katz_centrality_2 = []
    
    for i,j in all_edges:
        katz_centrality_1.append(katz_centrality[i])
        katz_centrality_2.append(katz_centrality[j])
    df["katz_centrality_1"] = katz_centrality_1
    df["katz_centrality_2"] = katz_centrality_2"""
    
    
        
        
        
    
    
    
    
    
    
    
    
    return df


def create_cvs(G):
    observed_links,observed_missing_links = create_observed_set(G,0.8)
    train_links,train_missing_links = create_training_set(observed_links,0.8)
    Gho =  G_create(G,observed_links)
    Gtr = G_create(G,train_links)

    df_ho = create_training_df(Gho,observed_links,observed_missing_links)
    df_tr = create_training_df(Gtr,train_links,train_missing_links)


    df_ho_top = top_feats(df_ho,Gho,"non_links")
    df_tr_top = top_feats(df_tr,Gtr,"non_links")
    feature_set = df_tr_top.columns[1:]


    X_train_cv,y_train_cv,X_test_cv,y_test_cv = create_cross_validation(df_tr_top,"Link_present")
    output_dict = {
        'X_train_cv': X_train_cv,
        'y_train_cv': y_train_cv,
        'X_test_cv': X_test_cv,
        'y_test_cv': y_test_cv,
    }

    
    return output_dict,df_ho_top,df_tr_top


import pickle  
# load the data 
infile = open('OLP_updated.pickle','rb')  
df = pickle.load(infile)  


def create_g_from_df(index):
    edges_orig  = df[df["network_index"] == index]
    edges = edges_orig["edges_id"].values[0]
    edges_lis = edges.tolist()
    G = nx.Graph()
    G.add_edges_from(edges_lis)
    return  G
network_index = df["network_index"]


network_index = network_index[:10]







    # Assuming you have a list of indices



def process_index(index):
    index_dir = "Network_datasest/{}_index".format(index)
    os.mkdir(index_dir)

    G = create_g_from_df(index)
    output_dict, df_ho_top, df_tr_top = create_cvs(G)

    with open(os.path.join(index_dir, 'out.pkl'), 'wb') as f:
        pickle.dump((output_dict), f)

    df_ho_top.to_csv(os.path.join(index_dir, 'df_ho_top.csv'), index=False)
    df_tr_top.to_csv(os.path.join(index_dir, 'df_tr_top.csv'), index=False)

if __name__ == "__main__":
    # Assuming you have a list of indices
    index_list = network_index # Replace with your list of indices

    # Determine the number of processes to use (use all available CPU cores)
    num_processes = cpu_count()

    # Create a Pool of processes
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_index, index_list), total=len(index_list)))
