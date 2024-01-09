import numpy as np
import torch
from scipy import io as sio, sparse
import torch_geometric as geo
import dgl
#from preprocess_DBLP import *
#from process_acm import return_acm
#from process_acm import return_acmgraph
import random
def load_data(name,seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if name == 'dblp':
        from preprocess_DBLP import return_graph
        data = return_graph()
        return load_dblp(data)
    elif name == 'acm':
        from process_acm import return_acmgraph
        data = return_acmgraph()
        return load_acm3(data)
    else:
        print("no data with the name provided exist")
        return    
    
def load_acm3(data):
    data_dict = {}
    for edge_type, store in data.edge_items():
        if store.get('edge_index') is not None:
            row, col = store.edge_index
        else:
            row, col, _ = store['adj_t'].t().coo()
        data_dict[edge_type] = (row, col)
    g = dgl.heterograph(data_dict)
    for node_type, store in data.node_items():
        for attr, value in store.items():
            if type(value)==int:
                value = torch.tensor([value]*20,dtype=torch.int8)
            g.nodes[node_type].data[attr] = value
        for edge_type, store in data.edge_items():
            for attr, value in store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_type].data[attr] = value
    print("dgl graph:",g)
    #data = return_acm()
    #features = data[0]
    #labels = data[1]
    #g = data[2]
    features = g.nodes['author'].data['x']
    labels = g.nodes['author'].data['y']
    num_classes = 14
    num_nodes = features.shape[0]
    #print(num_nodes)
    float_mask = np.random.permutation(np.linspace(0, 1, num_nodes))
    #train_idx = np.where(float_mask <= 0.01)[0]
    #val_idx = np.where((float_mask > 0.01) & (float_mask <= 0.02))[0]
    #test_idx = np.where(float_mask > 0.02)[0]
    train_idx = np.where(float_mask <= 0.5)[0]
    val_idx = np.where((float_mask > 0.5) & (float_mask <= 0.6))[0]
    test_idx = np.where(float_mask > 0.6)[0]
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return (
        g,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    )
def load_dblp(data):
    data_dict = {}
    for edge_type, store in data.edge_items():
        if store.get('edge_index') is not None:
            row, col = store.edge_index
        else:
            row, col, _ = store['adj_t'].t().coo()
        data_dict[edge_type] = (row, col)
    g = dgl.heterograph(data_dict)
    for node_type, store in data.node_items():
        for attr, value in store.items():
            if type(value)==int:
                value = torch.tensor([value]*20,dtype=torch.int8)
            #print(attr)
            #print(value.shape)
            #print(node_type)
            g.nodes[node_type].data[attr] = value
            #print("end?")
        for edge_type, store in data.edge_items():
            for attr, value in store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_type].data[attr] = value
    print("dgl graph:",g)
    features = g.nodes['author'].data['x']
    labels = g.nodes['author'].data['y']
    #labels = torch.from_numpy(np.load('./dblp_homo_list/APA.npz', allow_pickle=True)['y'])
    #labels = labels[0:features.shape[0],:] 
    num_classes = 4
    num_nodes = features.shape[0]
    #print(num_nodes)
    float_mask = np.random.permutation(np.linspace(0, 1, num_nodes))
    train_idx = np.where(float_mask <= 0.4)[0]
    val_idx = np.where((float_mask > 0.4) & (float_mask <= 0.5))[0]
    test_idx = np.where(float_mask > 0.5)[0]
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return (
        g,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    )

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte().to(torch.bool)
