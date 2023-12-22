"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import torch_geometric as geo
from mugcn import MUGCN
import copy
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        #w_exp = torch.exp(w)/100
        #w_sum = torch.sum(w_exp)
        w_soft = w/100
        beta = torch.softmax(w_soft, dim=0)  # (M, 1)
        print("semantic attention")
        print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        #print("semantic attention",beta)
        #print("z shape",z.shape)
        #print((beta*z).shape)
        #print((beta*z).sum(1).shape)
        return (beta * z).sum(1)  # (N, D * K)


class MUGCNLayer(nn.Module):
    def __init__(self, g,meta_paths,out_size, device, train_mask,val_mask,test_mask,args):
        super(MUGCNLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.mugcn_layers = []
        self.semantic_attention = SemanticAttention(
            in_size=8920
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)
        for i,meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            #new_g = dgl.to_homogenous(new_g) // uncomment for asymmetric path
            print("homo dgl graph",new_g)
            homo = geo.data.Data()
            homo.x = new_g.ndata["x"].clone().detach()
            homo.y = new_g.ndata["y"].clone().detach()

            homo.edge_index = torch.vstack((new_g.edges()[0].clone().detach(),new_g.edges()[1].clone().detach()))
            #nx_g = dgl.to_networkx(new_g.cpu())
            #homo = geo.utils.from_networkx(nx_g)
            #hetero_g = geo.data.to_heterogenous(homo_g) //uncomment for asymmetric pat
            print("geometric graph",homo)
            freqv = [30,50,70,90]
            freql = [30,50,70,90]
            gcnhidden = [32,64,128,256,300]
            pgehidden = [32,64,128]
            homo = homo.to(device)
            best_loss = 100
            best_fv = None
            best_adjfv = None
            best_adjv = None
            best_gcnv = None
            best_config = {"freqv":freqv[0],"freql":freql[0],"gcnhidden":gcnhidden[0],"pgehidden":pgehidden[0]}
            '''for freqv_ in freqv:
                args.freqv = freqv_
                for freql_ in freql:
                    args.freql = freql_
                    for gcnhidden_ in gcnhidden:
                        args.gcnhiden = gcnhidden_
                        for pgehidden_ in pgehidden:
                            args.pgehidden = pgehidden_
                            mugcn = MUGCN(homo,train_mask,val_mask,test_mask,args,device)
                            best_val_loss,epoch,fv,adjfv,adjv,gcnv = mugcn.train()
                            if best_loss>best_val_loss:
                                best_loss = best_val_loss
                                best_fv = copy.deepcopy(fv)
                                best_adjfv = copy.deepcopy(adjfv)
                                best_adjv = copy.deepcopy(adjv)
                                best_gcnv = copy.deepcopy(gcnv)
                                best_config["freqv"] = freqv_
                                best_config["freql"] = freql_
                                best_config["gcnhidden"] = gcnhidden_
                                best_config["pgehidden"] = pgehidden_
                            print("current best val loss",best_loss)
                            print("current best config:",best_config)'''
            mugcn = MUGCN(homo,train_mask,val_mask,test_mask,args,device)
            best_val_loss,epoch,best_fv,best_adjfv,best_adjv,best_gcnv = mugcn.train()
            #best_fv = torch.load('./trained_model/best_fv.pt')
            #best_adjfv = torch.load("./trained_model/best_adjfv.pt")
            #best_adjv = torch.load("./trained_model/best_adjv.pt")
            #best_gcnv = torch.load("./trained_model/best_gcnv.pt")
            mugcn.load(best_fv,best_adjfv,best_adjv,best_gcnv)
            #torch.save(best_fv,"./trained_model/best_fv.pt")
            #torch.save(best_adjfv,"./trained_model/best_adjfv.pt")
            #torch.save(best_adjv,"./trained_model/best_adjv.pt")
            #torch.save(best_gcnv,"./trained_model/best_gcnv.pt")
            print("single mugcn test")
            mugcn.test()
            self.mugcn_layers.append(mugcn)



    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            semantic_embeddings.append(self.mugcn_layers[i].embed())
        #print(semantic_embeddings[0].shape)
        #print(semantic_embeddings[1].shape)
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)
        print(semantic_embeddings.shape)
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN_MUGCN(nn.Module):
    def __init__(
        self,g,meta_paths, hidden_size,out_size, device, train_mask,val_mask,test_mask,args):
        super(HAN_MUGCN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            MUGCNLayer(g,meta_paths, out_size, device, train_mask,val_mask,test_mask,args))
        self.predict = nn.Linear(8920, out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return F.softmax(self.predict(h),dim=1)



