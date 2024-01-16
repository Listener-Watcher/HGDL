#import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
#from dgl.nn.pytorch import GATConv
from layers import GraphConvolution
import torch_geometric as geo
import copy
from math import sqrt
from utils_ import gcn_norm,adj_norm
from torch.nn import KLDivLoss,CrossEntropyLoss
from sklearn.metrics import f1_score
from scipy.spatial import distance
from utils_ import clark,intersection,from_edge_index_to_adj
from torch.nn.parameter import Parameter
#eps = 1e-14

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
    return accuracy, micro_f1, macro_f1,score_cosin


def evaluate(model, g, g2,features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        #logits,_ = model(g, features,g2)
        logits = model(g,features)
    loss = loss_func((logits[mask]+1e-9).log(), labels[mask]+1e-9)
    accuracy, micro_f1, macro_f1,score_intersection = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1,score_intersection
def evaluate2(model, g, g2,features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits,_ = model(g, features,g2)
    loss = loss_func((logits[mask]+1e-9).log(), labels[mask]+1e-9)
    accuracy, micro_f1, macro_f1,score_intersection = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1,score_intersection

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=4):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        #print("semantic learning")
        #print(z.shape)
        #print(z[0].shape)
        #print(z[:,0].shape)
        #print(z[:,:,0].shape)
        w = self.project(z).mean(0)  # (M, 1)
        #w = self.project(z)
        #w_exp = torch.exp(w)/100
        #w_sum = torch.sum(w_exp)
        #w_soft = w
        #print(w.shape)
        beta = torch.softmax(w, dim=1)  # (M, 1)
        #print("semantic attention")
        #print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        #print("semantic attention",beta)
        #print("z shape",z.shape)
        #print((beta*z).shape)
        #print((beta*z).sum(1).shape)
        #result = (beta*z).sum(1)
        #print(z.sum(1).shape)
        #mean_dis = torch.mean(z[:,0],0)
        #mean_dis2 = torch.mean(z[:,1],0)
        #print(mean_dis.shape)
        #print(mean_dis2.shape)
        #mean_dis= torch.softmax(mean_dis,dim=0)
        #mean_dis2= torch.softmax(mean_dis2,dim=0)
        #print("mean dis 2 :",mean_dis2)
        #print("mean dis 1:",mean_dis)
        #mean_dis = mean_dis.expand((z.shape[0],)+mean_dis.shape)
        #mean_dis2 = mean_dis2.expand((z.shape[0],)+mean_dis2.shape)
        #print(mean_dis.shape)
        #print(mean_dis)
        #loss = KLDivLoss(reduction="none")
        #z1 = loss(z[:,0].log(),mean_dis).sum(dim=1)
        #z2 = loss(z[:,1].log(),mean_dis2).sum(dim=1)
        #print(z1)
        #z3 = torch.stack((z1,z2),dim=1)
        #z3 = torch.softmax(z3,dim=1)
        #z3 = (z3>0.5)
        #for i in range(z3.shape[0]):
        #    if z3[i][0]==0:
        #        print("bb")
        #z3 = torch.unsqueeze(z3,dim=2)
        #z4 = torch.mul(z3,beta)
        #print(z3)
        #print(z3.shape)
        #print(beta.shape)
        #beta_ = torch.squeeze(beta)
        #print(beta_.shape)
        #print("difference",torch.linalg.matrix_norm(z3-beta_))
        return (beta * z).sum(1)  # (N, D * K)
        #return (z4*z).sum(1)
class GCN_attention(nn.Module):
    def __init__(self, nfeat, nhid,nclass,dropout,num_layers):
        super(GCN_attention, self).__init__()
        self.Q = nn.Linear(nhid,nhid,bias=True)
        self.K = nn.Linear(nhid,nhid,bias=True)
        self.V = nn.Linear(nhid,nhid,bias=True)
        self.gcn1=GraphConvolution(nfeat,nhid)
        self.gcn2=GraphConvolution(nhid, nclass)
        self.layer_norm = nn.LayerNorm(nhid)
        self.dropout = dropout
    def forward(self,adj,x):
        x = F.relu(self.gcn1(x,adj))
        Q_h = self.Q(x)
        K_h = torch.transpose(self.K(x),0,1)
        V_h = self.V(x)
        #A_tilde = sparse_dense_mul(adj,torch.matmul(Q_h,K_h))
        A_tilde = torch.mul(adj.to_dense(),torch.matmul(Q_h,K_h))
        attention = F.softmax(A_tilde,dim=1)   
        #attention = torch.sparse.softmax(A_tilde,dim=1)     
        X_tilde = torch.matmul(gcn_norm(attention),V_h)
        X_tilde = F.relu(X_tilde)
        z = self.gcn2(X_tilde,adj)
        return F.softmax(z,dim=1)
    @torch.no_grad()
    def embed(self,adj,h):
        self.eval()
        x = F.relu(self.gcn1(h,adj))
        Q_h = self.Q(x)
        K_h = torch.transpose(self.K(x),0,1)
        V_h = self.V(x)
        A_tilde = torch.mul(adj.to_dense(),torch.matmul(Q_h,K_h))
        attention = F.softmax(A_tilde,dim=1)
        X_tilde = torch.matmul(gcn_norm(attention),V_h)
        X_tilde = F.relu(X_tilde)
        #z = self.gcn2(X_tilde,adj)
        return X_tilde
        #return F.softmax(z,dim=1)
        
        
class GCN_attention2(nn.Module):
    def __init__(self, nfeat, nhid,nclass,dropout,num_layers):
        super(GCN_attention2, self).__init__()
        self.Q = nn.Linear(nhid,nhid,bias=True)
        self.K = nn.Linear(nhid,nhid,bias=True)
        self.V = nn.Linear(nhid,nhid,bias=True)
        self.gcn1=GraphConvolution(nfeat,nhid)
        self.gcn2=GraphConvolution(nhid, nclass)
        self.gcn = GraphConvolution(nhid, nhid)
        self.layer_norm = nn.LayerNorm(nhid)
        self.dropout = dropout
    def forward(self,adj,x):
        x = F.relu(self.gcn1(x,adj))
        '''Q_h = self.Q(x)
        K_h = torch.transpose(self.K(x),0,1)
        V_h = self.V(x)
        #A_tilde = sparse_dense_mul(adj,torch.matmul(Q_h,K_h))
        A_tilde = torch.mul(adj.to_dense(),torch.matmul(Q_h,K_h))
        attention = F.softmax(A_tilde,dim=1)   
        #attention = torch.sparse.softmax(A_tilde,dim=1)     
        X_tilde = torch.matmul(gcn_norm(attention),V_h)
        X_tilde = F.relu(X_tilde)'''
        X_tilde = F.relu(self.gcn(x,adj))
        z = self.gcn2(X_tilde,adj)
        return F.softmax(z,dim=1)
        
class GCN_attention_v2(nn.Module):
    def __init__(self, nfeat, nhid,nclass,dropout,num_layers,adj_list,adj_list_origin):
        super(GCN_attention_v2, self).__init__()
        self.Q = nn.Linear(nhid,nhid,bias=True)
        self.K = nn.Linear(nhid,nhid,bias=True)
        self.V = nn.Linear(nhid,nhid,bias=True)
        self.gcn1=GraphConvolution(nfeat,nhid)
        self.gcn2=GraphConvolution(nhid, nclass)
        self.layer_norm = nn.LayerNorm(nhid)
        #self.weight = Parameter(torch.FloatTensor(adj_list[0].shape[0],3)).to("cuda:0")
        self.atten = nn.Linear(adj_list[0].shape[0],5)
        self.atten2 = nn.Linear(adj_list[0].shape[0],5)
        self.atten3 = nn.Linear(adj_list[0].shape[0],5)
        self.agg = nn.Linear(15,len(adj_list))
        self.dropout = dropout
        #self.memory_unit = torch.FloatTensor([[1/3,1/3,1/3,0,1,0]]*adj_list[0].shape[0]).to('cuda:0')
        self.memory_unit = torch.FloatTensor([[0,1]]*adj_list[0].shape[0]).to('cuda:0')
        #self.memory_adj = torch.FloatTensor(adj_list[0].shape).to('cuda:0')
        self.gcn = GraphConvolution(nhid,nhid)
    def forward(self,adj_list,x,adj_list_origin):
        #self.weight = torch.softmax(self.weight,dim=1)
        #print("weight",self.weight)
        Adj_kernel = []
        for i in range(len(adj_list)):
            #Adj_kernel.append(torch.stack((adj_list[i],adj_list_origin[i]),dim=1))
            Adj_kernel.append(adj_list[i])
        z1 = self.atten(Adj_kernel[0])
        z2 = self.atten2(Adj_kernel[1])
        z3 = self.atten3(Adj_kernel[2])
        #z4 = torch.stack((z1,z2,z3),dim=1).squeeze()
        #print(torch.cat((z1,z2),dim=1).shape)
        z4 = ((self.agg(torch.cat((z1,z2,z3),dim=1))))
        #z4 = self.agg(torch.cat((z1,z2),dim=1))
        #print("stack shape",z4.shape)
        nz = torch.softmax(z4,dim=1)
        uniform = torch.FloatTensor([[0,0,1]]*adj_list[0].shape[0]).to('cuda:0')
        #uniform = torch.FloatTensor([[0.5,0.5]]*adj_list[0].shape[0]).to('cuda:0')
        #print("nz away from uniform",torch.sum(torch.abs(nz-uniform)))
        #print("nz update",torch.sum(torch.abs(nz-self.memory_unit)))
        #print("nz away from 0,1",torch.sum(nz[:,1]>0.5)/nz.shape[0])
        print(nz)
        self.memory_unit = nz
        adj = nz[:,0]*adj_list[0]+nz[:,1]*adj_list[1]+nz[:,2]*adj_list[2]
        #adj = nz[:,0]*adj_list[0]+nz[:,1]*adj_list[1]
        #adj = gcn_norm(adj)
        #adj_sparse = adj.to_sparse()
        adj = adj.to_sparse()
        x = F.relu(self.gcn1(x,adj))
        
        Q_h = self.Q(x)
        K_h = torch.transpose(self.K(x),0,1)
        V_h = self.V(x)
        #A_tilde = sparse_dense_mul(adj,torch.matmul(Q_h,K_h))
        A_tilde = torch.mul(adj.to_dense(),torch.matmul(Q_h,K_h))
        attention = F.softmax(A_tilde,dim=1)   
        #attention = torch.sparse.softmax(A_tilde,dim=1)     
        X_tilde = torch.matmul(gcn_norm(attention),V_h)
        X_tilde = F.relu((X_tilde))
        #X_tilde = self.layer_norm(X_tilde)
        
        #random_mask = torch.rand(adj_list[0].shape,device="cuda:0")<0.9
        #X_tilde = F.relu(self.gcn(x,torch.mul(adj.to_dense(),random_mask).to_sparse()))
        #X_tilde = F.relu(self.gcn(x,adj))
        z = self.gcn2(X_tilde,adj)
        nj = torch.softmax(z4,dim=1)
        return F.softmax(z,dim=1),nj
  
        
class GCN_attention_v3(nn.Module):
    def __init__(self, nfeat, nhid,nclass,dropout,num_layers,adj_list,adj_list_origin):
        super(GCN_attention_v3, self).__init__()
        self.Q = nn.Linear(nhid,nhid,bias=True)
        self.K = nn.Linear(nhid,nhid,bias=True)
        self.V = nn.Linear(nhid,nhid,bias=True)
        self.gcn1=GraphConvolution(nfeat,nhid)
        self.gcn2=GraphConvolution(nhid, nclass)
        self.layer_norm = nn.LayerNorm(nhid)
        #self.weight = Parameter(torch.FloatTensor(adj_list[0].shape[0],3)).to("cuda:0")
        self.atten = nn.Linear(adj_list[0].shape[0],30)
        self.atten2 = nn.Linear(adj_list[0].shape[0],30)
        #self.atten3 = nn.Linear(adj_list[0].shape[0],20)
        self.agg = nn.Linear(60,len(adj_list))
        self.dropout = dropout
        #self.memory_unit = torch.FloatTensor([[1/3,1/3,1/3,0,1,0]]*adj_list[0].shape[0]).to('cuda:0')
        self.memory_unit = torch.FloatTensor([[0,1]]*adj_list[0].shape[0]).to('cuda:0')
        #self.memory_adj = torch.FloatTensor(adj_list[0].shape).to('cuda:0')
        self.gcn = GraphConvolution(nhid,nhid)
    def forward(self,adj_list,x,adj_list_origin):
        #self.weight = torch.softmax(self.weight,dim=1)
        #print("weight",self.weight)
        Adj_kernel = []
        for i in range(len(adj_list)):
            #Adj_kernel.append(torch.stack((adj_list[i],adj_list_origin[i]),dim=1))
            Adj_kernel.append(adj_list[i])
        z1 = self.atten(Adj_kernel[0])
        z2 = self.atten2(Adj_kernel[1])
        z3 = self.atten3(Adj_kernel[2])
        #z4 = torch.stack((z1,z2,z3),dim=1).squeeze()
        #print(torch.cat((z1,z2),dim=1).shape)
        z4 = ((self.agg(torch.cat((z1,z2,z3),dim=1))))
        #z4 = self.agg(torch.cat((z1,z2),dim=1))
        #print("stack shape",z4.shape)
        nz = torch.softmax(z4,dim=1)
        uniform = torch.FloatTensor([[0,1]]*adj_list[0].shape[0]).to('cuda:0')
        #uniform = torch.FloatTensor([[0.5,0.5]]*adj_list[0].shape[0]).to('cuda:0')
        #print("nz away from uniform",torch.sum(torch.abs(nz-uniform)))
        #print("nz update",torch.sum(torch.abs(nz-self.memory_unit)))
        #print("nz away from 0,1",torch.sum(nz[:,1]>0.5)/nz.shape[0])
        #print(nz)
        self.memory_unit = nz
        adj = nz[:,0]*adj_list[0]+nz[:,1]*adj_list[1]+nz[:,2]*adj_list[2]
        #adj = nz[:,0]*adj_list[0]+nz[:,1]*adj_list[1]
        #adj = gcn_norm(adj)
        #adj_sparse = adj.to_sparse()
        adj = adj.to_sparse()
        x = F.relu(self.gcn1(x,adj))
        """Q_h = self.Q(x)
        K_h = torch.transpose(self.K(x),0,1)
        V_h = self.V(x)
        #A_tilde = sparse_dense_mul(adj,torch.matmul(Q_h,K_h))
        A_tilde = torch.mul(adj.to_dense(),torch.matmul(Q_h,K_h))
        attention = F.softmax(A_tilde,dim=1)   
        #attention = torch.sparse.softmax(A_tilde,dim=1)     
        X_tilde = torch.matmul(gcn_norm(attention),V_h)
        X_tilde = F.relu((X_tilde))
        #X_tilde = self.layer_norm(X_tilde)
        #random_mask = torch.rand(adj_list[0].shape,device="cuda:0")<0.9
        """
        X_tilde = F.relu(self.gcn(x,adj))
        z = self.gcn2(X_tilde,adj)
        nj = torch.softmax(z4,dim=1)
        return F.softmax(z,dim=1),nj

        
        


class Gtransformerblock(nn.Module):
    def __init__(
        self,in_dim,hid_dim,out_dim,num_heads,adj_list,adj_list_origin,features,labels, train_mask,val_mask,test_mask,device,dropout=0.0, layer_norm=True, use_bias=False):
        super().__init__()
        self.in_channels = in_dim
        self.hid_channels = hid_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.adj_list = adj_list
        self.adj_list_origin = adj_list_origin
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask        
        self.device = device
        self.Q2 = nn.Linear(hid_dim,hid_dim,bias=True)
        self.K2 = nn.Linear(hid_dim,hid_dim,bias=True)
        self.V2 = nn.Linear(hid_dim,hid_dim,bias=True)
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.predict = nn.Linear(hid_dim*self.num_heads,out_dim)
        GAT = []
        loss = KLDivLoss(reduction="batchmean")
        for i in range(self.num_heads):
            gat = GCN_attention(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            gat.load_state_dict(torch.load("YELP_attention_single"+str(i)+".pth"))
            GAT.append(gat)
        self.mugcn_layers = GAT
        
    def forward(self, adj_list, h):
        #result_list = []
        #for i in range(self.num_heads):
        #   result_list.append(self.GCN(h,adj_list[i].to_sparse()))
        '''attn_out = self.attentionlayer.embed(adj_list,h)
        result_list = attn_out
        semantic_embeddings = torch.stack(
            result_list, dim=1
        )'''
        semantic_embeddings = []
        for i in range(self.num_heads):
            z = self.mugcn_layers[i].embed(adj_list[i],h)
            qh = self.Q2(z)
            kh = self.K2(z)
            vh = self.V2(z)
            semantic_embeddings.append((qh,kh,vh,z))
        result_embeddings = []
        Attention = []
        print(torch.mul(semantic_embeddings[0][0],semantic_embeddings[0][1]).shape)
        for i in range(self.num_heads):
            attention = []
            for j in range(self.num_heads):
                attention.append((torch.sum(torch.mul(semantic_embeddings[i][0],semantic_embeddings[j][1]),axis=1)))

            print(torch.stack(attention,dim=1).shape)
            attention = torch.softmax(torch.stack(attention,dim=1),axis=1)
            Attention.append(attention)
        result_embeddings = []
        for i in range(self.num_heads):
            result = torch.zeros(semantic_embeddings[j][3].shape).to("cuda:0")
            for j in range(self.num_heads):
                result = result+(self.beta*torch.mul(torch.unsqueeze(Attention[i][:,j],dim=1),semantic_embeddings[j][2]))
            result+=semantic_embeddings[j][3]
            result_embeddings.append(result)
        final_embedding = torch.cat((result_embeddings),dim=1)
        return F.softmax(self.predict(final_embedding),dim=1)
        #return F.softmax(p2,dim=1)
