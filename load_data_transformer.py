import numpy as np
import torch
from scipy import io as sio, sparse
import torch_geometric as geo
# from preprocess_DBLP import *
from process_acm import return_acmgraph

def load_acm3(remove_self_loop):
    g= return_acmgraph()
    features = g.['author'].x
    labels = g.['author'].y
    num_classes = 14
    num_nodes = features.shape[0]
    #print(num_nodes)
    float_mask = np.random.permutation(np.linspace(0, 1, num_nodes))
    train_idx = np.where(float_mask <= 0.01)[0]
    val_idx = np.where((float_mask > 0.01) & (float_mask <= 0.02))[0]
    test_idx = np.where(float_mask > 0.02)[0]
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