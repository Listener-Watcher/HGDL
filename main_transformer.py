import torch
from sklearn.metrics import f1_score
from scipy.spatial import distance
from utils_ import clark,intersection,from_edge_index_to_adj
# from utils_mugcn_pre import *
from load_data_transformer import *
from model import Gtransformerblock
import dgl
import datetime
import errno
import os
import pickle
import random
#from utils_mugcn_pre import EarlyStopping
eps = 1e-9

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
        self.epochs = 0
        self.best_epochs = 0

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
            self.best_epochs = 0
        elif (loss > self.best_loss) and (acc <= self.best_acc):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
                self.best_epochs = self.epochs
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        self.epochs+=1
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        print("best epoch",self.best_epochs)
        model.load_state_dict(torch.load(self.filename))
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
    adj_list = []
    meta_paths=[[("author","to","paper"),("paper","to","author")],[("author","to","affiliation"),("affiliation","to","author")]]
    meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
    for meta_path in meta_paths:
        new_g = dgl.metapath_reachable_graph(g,meta_path)
        edge_index = torch.vstack((new_g.edges()[0].clone().detach(),new_g.edges()[1].clone().detach()))
        adj_list.append(from_edge_index_to_adj(edge_index.to(torch.long)))
    #print(adj_list[0].shape)
    #print("num_heads",len(adj_list))
    # meta_paths = [['AP','PA'],['AP','PC','CP','PA']]
    # adj_list = []
    # for paths in meta_paths:
    #     temp_adj = from_edge_index_to_adj(g[paths[0]])
    #     for i in range(1,len(paths)):
    #         temp_adj = torch.matmul(temp_adj,g[paths[i]])
    #     adj_list.append(temp_adj)

    #for meta_path in meta_paths:
    #     g1 = from_edge_index_to_adj(g[meta_path].edge_index)
    #     A1 = g1.adj()
    #     adj_list.append(A1)
    #print("A1",adj_list[0].shape)
    print(features.shape)
    #for i in range(3):
    #    adj_list.append(torch.ones(features.shape[0],features.shape[0]))
    num_heads = len(adj_list)
    in_dim = features.shape[1]
    out_dim = num_classes
    dropout = args.dropout
    layer_norm = args.layer_norm
    use_bias = args.use_bias
    model = Gtransformerblock(in_dim=in_dim,hid_dim=128,out_dim=out_dim, num_heads=num_heads, dropout=dropout, layer_norm=layer_norm, use_bias=use_bias)
    #print("begin test")
    #model.forward(adj_list,features)
    #print("end test")
    
    
    stopper = EarlyStopping(patience=args.patience)
    #loss_fcn = torch.nn.CrossEntropyLoss()
    loss_fcn = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0005, weight_decay=0
    )
    for epoch in range(1000):
        model.train()
        logits = model.forward(adj_list, features)
        loss = loss_fcn((logits[train_mask]+1e-9).log(), labels[train_mask]+1e-9)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[train_mask], labels[train_mask]
        )
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model, adj_list, features, labels, val_mask, loss_fcn
        )
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print(
            "Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
            "Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
                epoch + 1,
                loss.item(),
                train_micro_f1,
                train_macro_f1,
                val_loss.item(),
                val_micro_f1,
                val_macro_f1,
            )
        )

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model, adj_list, features, labels, test_mask, loss_fcn
    )
    print(
        "Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
            test_loss.item(), test_micro_f1, test_macro_f1
        )
    )

    
    
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
    parser.add_argument('--patience',type=int,default=1000)
    args = parser.parse_args()
    #args = setup(args)
    main(args)
