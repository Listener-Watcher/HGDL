import scipy.io
import torch
import networkx as nx
import scipy as sp
import numpy as np
import torch_geometric as geo
from collections import defaultdict
def edge_index_to_dict(edge_index):
    """
    helper function that converts an edge_index tensor into a dictionary
    :param edge_index: a 2 x num_edges tensor that represents the edges in a graph
    :return: a dictionary that represents has each node as a key, and each value is a list of
                the nodes it is connected to
    """
    #num_edges = edge_index
    #print(num_edges)
    edge_dict = defaultdict(list)
    #print(edge_index)
    num_edges = len(edge_index[0])
    for i in range(num_edges):
        edge_dict[edge_index[0][i].item()].append(edge_index[1][i].item())
    return edge_dict
acm = scipy.io.loadmat('data/ACM.mat')
# paper's bag of words
paper_feat = acm['PvsT'].tocoo()
paper_feat = paper_feat.toarray()
print("paper features",paper_feat)
authors = range(acm['A'].size)
print("original authors",acm['A'])
print("authors",authors)
# # scipy csc sparse matrix format
edges = acm['PvsA'].tocoo()
edges_pa = torch.tensor(np.vstack((edges.row, edges.col)))
edges_ap = torch.tensor(np.vstack((edges.col, edges.row)))
edges = acm['PvsC'].tocoo()
edges_pc = torch.tensor(np.vstack((edges.row, edges.col)))
edges_cp = torch.tensor(np.vstack((edges.col, edges.row)))
edges = acm['PvsL'].tocoo()
edges_pl = torch.tensor(np.vstack((edges.row, edges.col)))
edges_lp = torch.tensor(np.vstack((edges.col, edges.row)))
edges = acm['PvsV'].tocoo()
edges_pv = torch.tensor(np.vstack((edges.row, edges.col)))
edges_vp = torch.tensor(np.vstack((edges.col, edges.row)))
edges = acm['AvsF'].tocoo()
edges_af = torch.tensor(np.vstack((edges.row, edges.col)))
edges_fa = torch.tensor(np.vstack((edges.col, edges.row)))
print("edges format",edges_pa)
# node types
subjects = acm['L']
affiliations = acm['F']
conferences = acm['C']
papers = acm['P']
terms = acm['T']
proceedings = acm['V']

sub_full_to_dense = [0] * subjects.size
for i in range(subjects.size):
    letter = subjects[i][0].item()[0].lower()
    sub_full_to_dense[i] = int(ord(letter) - ord('a'))

print(f'Max subject index in converter: {max(sub_full_to_dense)}')
# paper label edges
paper_conf = acm['PvsC'].toarray()
paper_sub = acm['PvsL'].tocoo()
sub_array = paper_sub.col
for i in range(sub_array.size):
    sub_array[i] = sub_full_to_dense[sub_array[i]]
print(f'Max subject index in the dataset: {max(sub_array)}')
paper_sub = sp.sparse.coo_array((paper_sub.data, (paper_sub.row, sub_array)), shape=(paper_feat.shape[0], max(sub_full_to_dense) + 1))
paper_sub = paper_sub.toarray()
print(paper_sub.shape)
paper_author = edge_index_to_dict(edges_pa)
author_paper = edge_index_to_dict(edges_ap)
edge_index = [[], []]
conf_label_dist = torch.zeros(len(authors), paper_conf.shape[1])
sub_label_dist = torch.zeros(len(authors), max(sub_full_to_dense) + 1)
print(f'Subject Label Shape: {sub_label_dist.shape}')
features = torch.zeros(len(authors), paper_feat.shape[1])
for src_author, paper_list in author_paper.items():
    for paper in paper_list:
        # aggregating each paper's bag of words features into the author
        features[src_author] += paper_feat[paper]
        # populating label distributions
        conf_label_dist[src_author] += paper_conf[paper]
        sub_label_dist[src_author] += paper_sub[paper]
        # creating coauthor edges
        for end_author in paper_author[paper]:
            edge_index[0].append(src_author)
            edge_index[1].append(end_author)
print(features.shape)
print(conf_label_dist.shape)
print(sub_label_dist.shape)
conf_label_dist_norm = np.copy(conf_label_dist)
sub_label_dist_norm = np.copy(sub_label_dist)
for i in range(conf_label_dist_norm.shape[0]):
    conf_label_dist_norm[i] = conf_label_dist_norm[i] / conf_label_dist_norm[i].sum()
    sub_label_dist_norm[i] = sub_label_dist_norm[i] / sub_label_dist_norm[i].sum()
print(conf_label_dist_norm.shape)
print(sub_label_dist_norm.shape)
print(conf_label_dist_norm[0])
print(sub_label_dist_norm[0])
conf_label_dist_norm = torch.from_numpy(conf_label_dist_norm)
#np.savez("data/ACM/acm_conf",x = features,y=conf_label_dist_norm,edge_ap = edges_ap,edge_pa=edges_pa,edge_pc=edges_pc,edge_pl = edges_pl,edge_lp=edges_lp,edge_pv=edges_pv,edge_vp=edges_vp,edge_af=edges_af,edge_fa=edges_fa)
#np.savez("data/ACM/acm_subj",x = features,y=sub_label_dist_norm,edge_ap = edges_ap,edge_pa=edges_pa,edge_pc=edges_pc,edge_pl = edges_pl,edge_lp=edges_lp,edge_pv=edges_pv,edge_vp=edges_vp,edge_af=edges_af,edge_fa=edges_fa)
hetero_graph = geo.data.HeteroData()
hetero_graph['author'].x = features
hetero_graph['author'].y = conf_label_dist_norm
hetero_graph['author','to','paper'].edge_index = edges_ap
hetero_graph['paper','to','author'].edge_index = edges_pa
hetero_graph['paper','to','conference'].edge_index = edges_pc
hetero_graph['conference','to','paper'].edge_index = edges_cp
hetero_graph['paper','to','subjects'].edge_index = edges_pl
hetero_graph['subjects','to','paper'].edge_index = edges_lp
hetero_graph['paper','to','proceedings'].edge_index = edges_pv
hetero_graph['proceedings','to','paper'].edge_index = edges_vp
hetero_graph['author','to','affiliation'].edge_index = edges_af
hetero_graph['affiliation','to','author'].edge_index = edges_fa
#meta_path_adj = {'AP':edges_ap,'PA':edges_pa,'PC':edges_pc,'CP':edges_cp,'PS':edges_pl,'SP':edges_lp,'PR':edges_pv,'RP':edges_vp,'AF':edges_af,'FA':edges_fa}
#acm_data = [features,conf_label_dist_norm,meta_path_adj]
def return_acmgraph():
    return hetero_graph
#def return_acm():
#    return acm_data
#np.savez("data/acm/acm_conf", x=features, y=conf_label_dist, edge_index=edge_index)
#np.savez("data/acm/acm_sub", x=features, y=sub_label_dist, edge_index=edge_index)
#
# conf_file = np.load("data/acm/acm_conf.npz")
# conf_label_dist = np.load("data/acm/acm_conf.npz")['y']
# sub_label_dist = np.load("data/acm/acm_sub.npz")['y']
# print(sub_label_dist.shape)
# features = conf_file['x']
# edge_index = conf_file['edge_index']
#
# conf_label_dist_norm = np.copy(conf_label_dist)
# sub_label_dist_norm = np.copy(sub_label_dist)
#
# for i in range(conf_label_dist_norm.shape[0]):
#     conf_label_dist_norm[i] = conf_label_dist_norm[i] / conf_label_dist_norm[i].sum()
#     sub_label_dist_norm[i] = sub_label_dist_norm[i] / sub_label_dist_norm[i].sum()
#
# np.savez("data/acm/acm_conf", x=features, y=conf_label_dist_norm, edge_index=edge_index)
# np.savez("data/acm/acm_sub", x=features, y=sub_label_dist_norm, edge_index=edge_index)

