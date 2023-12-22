import numpy as np
import torch
from torch_geometric.utils import to_dense_adj,dense_to_sparse,get_laplacian
import time

device = 'cuda'
eps = 1e-7
def truncate(edge_indexF,mode,n):
    """
        mode:'label' or 'node'
        hetero_edge_index: edge_indexF
    """
    if mode =='l':
        f = edge_indexF[n:]
    else:
        f = edge_indexF[:n]
    return f

def updatehetero(edge_indexF,edge_indexl):
    """
    hetero_edge_index: edge_indexF
    updated label graph: edge_indexl
    goal: update hetero_edge_index with the updated label grpah
    """
    num_examples = edge_indexF.shape[0]-edge_indexl.shape[0]
    edge_indexF[num_examples:num_examples+edge_indexl.shape[0],num_examples:num_examples+edge_indexl.shape[1]] = edge_indexl
    return edge_indexF


def genhetero(edge_indexv,edge_indexl, Yv, train_mask):
    """
    node graph:edge_indexv 1711*1711 n*n matrix (adj)
    label graph:edge_indexl 4*4 m*m matrix (adj)
    node label: Yv
    hetero_edge_index = torch.zeros((n+m,n+m))

    """
    num_nodes = edge_indexv.size(dim=0)
    num_labels = edge_indexl.size(dim=1)
    hetero_edge_index = torch.zeros((num_nodes + num_labels, num_nodes + num_labels)).to(device)
    hetero_edge_index[:num_nodes, :num_nodes] = edge_indexv
    hetero_edge_index[num_nodes:, num_nodes:] = edge_indexl

    node_label_edge_index = torch.zeros((num_nodes, num_labels))
    for i in range(num_nodes):
        if train_mask[i]:
            val = torch.max(Yv[i])
            index = torch.argmax(Yv[i])
            node_label_edge_index[i][index] = val.item()
            # node_label_edge_index[i][index] = 1
    hetero_edge_index[:num_nodes, num_nodes:] = node_label_edge_index
    hetero_edge_index[num_nodes:, :num_nodes] = torch.t(node_label_edge_index)
    # for i in range(num_nodes):
    #     if train_mask[i]:
    #         val = torch.max(Yv[i])
    #         index = torch.argmax(Yv[i])
    #         hetero_edge_index[i][index+num_nodes] = val.item()
    #         hetero_edge_index[index+num_nodes][i] = val.item()
    return hetero_edge_index

def from_edge_index_to_adj(edge_index,edge_weight=None):
    adj = to_dense_adj(edge_index=edge_index,edge_attr=edge_weight)
    return adj[0]

def from_adj_to_edge_index(adj):
    edge_index = dense_to_sparse(adj)
    return edge_index[0],edge_index[1]
def genfeat(feature,y,train_mask):
    feature = feature[train_mask]
    y = y[train_mask]
    num_label = y.size(dim=1)
    num_example = y.size(dim=0)
    num_feature = feature.size(dim=1)
    # print(num_label)
    # print(num_example)
    # print(num_feature)
    feature_label = torch.zeros((num_label,num_feature)).to(device)
    feature_label = torch.add(feature_label,eps)
    count_label = np.zeros(num_label)
    for i in range(num_example):
        index = torch.argmax(y[i]).item()
        feature_label[index]+=feature[i]
        count_label[index]+=1
    for i in range(num_label):
        if count_label[i]!=0:
            feature_label[i]/=count_label[i]
        else:
            print("error on the dataset")
    # print(feature_label.shape)
    #print(feature_label[0])
    # print(count_label)
    return feature_label



def get_eigen(adj):
    eigenvalues = torch.linalg.eigvals(adj)
    # print(eigenvalues)
    eigen = eigenvalues.real
    sorted_eigen,_ = torch.sort(eigen)
    # print(sorted_eigen)
    return sorted_eigen
def clark(predict,actual):
    score = 0
    num = predict.shape[0]
    predict = predict.cpu().detach().numpy()
    actual = actual.cpu().detach().numpy()
    for i in range(num):
        score+=np.sqrt(np.sum(np.divide(np.square(actual[i]-predict[i]),np.square(actual[i]+predict[i]))))
    score = score/num
    return score
def intersection(predict,actual):
    score = 0
    num = predict.shape[0]
    predict = predict.cpu().detach().numpy()
    actual = actual.cpu().detach().numpy()
    for i in range(num):
        score+=np.sum(np.minimum(predict[i],actual[i]))
    score = score/num
    return score
def normalize(edge_index,edge_weight):
    lap = get_laplacian(edge_index,edge_weight=edge_weight,normalization='sym')
    # print(lap)
    return lap[0],lap[1]
def gcn_norm(adj):
    I = torch.eye(adj.shape[0]).to(device)
    adj = adj+I
    degrees = torch.sum(adj,1)
    degrees = torch.pow(degrees,-0.5)
    D = torch.diag(degrees)
    return torch.matmul(torch.matmul(D,adj),D)
def adj_norm(adj):
    D = torch.diag(torch.sum(adj,1))
    return D-adj


def k_hop(A,i,k):
    for hop in range(k):
        A[i]
def homophily(edge_list,labels):
    #print(edge_list)
    score = 0
    num_edges = len(edge_list[0])
    #print(num_edges)
    for i in range(num_edges):
        #print("labels",labels[edge_list[0][i]])
        #print(labels[edge_list[1][i]])
        if labels[edge_list[0][i]] == labels[edge_list[1][i]]:
            score+=1
    score = score/num_edges
    return score
def homophily_simple_sample(edge_list,labels,r,features):
    num_edges = len(edge_list[0])
    num_samples = int(num_edges*r)
    score = 0
    #score_feat = 0
    indices = torch.randperm(num_edges)[:num_samples]
    #cos = torch.nn.CosineSimilarity(dim=0)
    for i in indices:
        if labels[edge_list[0][i]] == labels[edge_list[1][i]]:
            score+=1
        #score_feat +=cos(features[edge_list[0][i]],features[edge_list[1][i]])
    score = score/num_samples
    #score_feat = score_feat/num_samples
    return score
'''def homophily_k_simple_sample(edge_list,labels,k,r):
    num_nodes = labels.shape[0]
    num_samples = int(num_nodes*r)
    score = torch.zeros((1,k)) # approximate scores for homophily up to hop k
    num_edges = 0
    indices = torch.randomperm(num_nodes)[:num_samples] # randomly sample a subset of nodes
    A = from_edge_index_to_adj(edge_list)
    for i in indices:
        for j in k:
'''
                

# compute all the h-score together with the converted graph
def homophily_k(edge_list,labels,k,features):
    A = from_edge_index_to_adj(edge_list)
    #A_list = []
    A_k = (torch.linalg.matrix_power(A,k)>0).long()
    for i in range(1,k):
        A_i = (torch.linalg.matrix_power(A,i)>0).long()
        #A_list.append((torch.linalg.matrix_power(A,i)>0).long())
        A_k = A_k-A_i
    A_k.fill_diagonal_(0)
    #torch.save(A_k,'order'+str(k)+'.pt')
    edge_listk,_ = from_adj_to_edge_index(A_k)
    #print("enter exact compute")
    #t0 = time.time()
    score = homophily(edge_listk,labels)
    r = 0.1
    #t1 = time.time()
    #print("exact time:",t1-t0)
    t0 = time.time()
    print("enter sample compute")
    #appro = homophily_simple_sample(edge_listk,labels,r,features)
    t1 = time.time()
    print("sample time:",t1-t0)
    #print("exact score:",score)
    #print("appro score:",appro)
    #print("approximation error:",abs(appro-score)/score)
    return score
                
def create_synthetic_graph(homo_score,num_nodes):
    pass

def fraction_matrix_power(A,ratio):
    A_ = A.type(torch.complex128)
    evals,evecs = torch.linalg.eig(A_)
    #print("eigenvalues",evals)
    #print("eigenvectors",evecs)
    evals = evals.real
    evecs = evecs.real
    #evecs = torch.round(evecs)
    print('eigenvalues',evals)
    print("eigenvectors",evecs)
    #evecs_ = torch.transpose(evecs,0,1)
    #print(torch.matmul(A.type(torch.double),evecs_[0]))
    #print(torch.matmul(A.type(torch.double),evecs_[1]))
    #print(torch.matmul(A.type(torch.double),evecs_[2]))
    #mchk = torch.matmul(evecs,torch.matmul(torch.diag(evals),torch.inverse(evecs)))
    #print("torch.inverse(evecs)",torch.inverse(evecs))
    #print("recover the original A4",mchk)
    evpow = evals**(ratio)
    print("powered eigenvalues",evpow)
    mpow = torch.matmul(evecs,torch.matmul(torch.diag(evpow),torch.linalg.inv(evecs)))
    print(torch.diag(evpow))
    print(torch.linalg.inv(evecs))
    print(evecs)
    #print("recover back/",torch.matmul(mpow,mpow))
    #print("return powered A4",mpow)
    return mpow
