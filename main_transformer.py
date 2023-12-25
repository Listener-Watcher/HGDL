import torch
from sklearn.metrics import f1_score
#from utils import EarlyStopping, load_data
from scipy.spatial import distance
from utils_ import clark,intersection,from_edge_index_to_adj
# from utils_mugcn_pre import *
from load_data_transformer import *
from model import Gtransformerblock
eps = 1e-9
def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    _, indices2 = torch.max(labels,dim=1)
    prediction = indices.long().cpu().numpy()
    #labels = labels.cpu().numpy()
    labels_ = indices.long().cpu().numpy()

    accuracy = (prediction == labels_).sum() / len(prediction)
    micro_f1 = f1_score(labels_, prediction, average="micro")
    macro_f1 = f1_score(labels_, prediction, average="macro")
    score_cosin = 0
    score_can = 0
    score_cheb = 0
    for i in range(labels.shape[0]):
        score_cosin += distance.cosine(labels[i].cpu().detach().numpy(),logits[i].cpu().detach().numpy())
        score_can += distance.canberra(labels[i].cpu().detach().numpy(),logits[i].cpu().detach().numpy())
        score_cheb += distance.chebyshev(labels[i].cpu().detach().numpy(),logits[i].cpu().detach().numpy())
    score_cosin/= labels.shape[0]
    score_can/=labels.shape[0]
    score_cheb/=labels.shape[0]
    score_clark = clark(logits+eps,labels+eps)
    score_intersection = intersection(logits,labels)
    print(f'cosin distance:{score_cosin:.4f}')
    print(f'canberra distance:{score_can:.4f}')
    print(f'chebyshev distance:{score_cheb:.4f}')
    print(f'clark distance:{score_clark:.4f}')
    print(f'intersection distance:{score_intersection:.4f}')
    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func((logits[mask]+1e-9).log(), labels[mask]+1e-9)
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    (
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
    ) = load_acm3(remove_self_loop=False)
    #APCPA
    #meta_paths=[[("author","to","paper"),("paper","to","author")],[("author","to","paper"),("paper","to","conference"),("conference","to","paper"),("paper","to","author")]]
    meta_paths = [['AP','PA'],['AP','PC','CP','PA']]
    adj_list = []
    for paths in meta_paths:
        temp_adj = from_edge_index_to_adj(g[paths[0]])
        for i in range(1,len(paths)):
            temp_adj = torch.matmul(temp_adj,g[paths[i]])
        adj_list.append(temp_adj)
    # for meta_path in meta_paths:
    #     g1 = from_edge_index_to_adj(g[meta_path].edge_index)
    #     A1 = g1.adj()
    #     adj_list.append(A1)
    print("A1",adj_list[0].shape)
    print(features.shape)
    num_heads = len(adj_list)
    in_dim = features.shape[1]
    out_dim = num_classes
    dropout = args.dropout
    layer_norm = args.layer_norm
    model = Gtransformerblock(in_dim=in_dim, out_dim=out_dim, num_heads=num_heads, dropout=dropout, layer_norm=layer_norm, use_bias=use_bias)
    print("begin test")
    model.forward(adj_list,features)
    print("end test")
if __name__ == "__main__":
    import argparse
    #from utils import setup
    parser = argparse.ArgumentParser("HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    """parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )"""
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='gpu or cpu')
    parser.add_argument('--epochs', type=int, default=6000)
    parser.add_argument('--dataset',type=str,default='dblp')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--alpha',type=float,default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--layer_norm',type=bool,default=True)
    parser.add_argument('--residual',type=bool,default=True)
    parser.add_argument('--use_bias',type=bool,default=True)
    args = parser.parse_args()
    args = setup(args)
    main(args)
