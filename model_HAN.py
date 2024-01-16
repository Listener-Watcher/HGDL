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
    def __init__(self, in_size, hidden_size=32):
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
        GAT = []
        loss = KLDivLoss(reduction="batchmean")
        for i in range(self.num_heads):
            gat = GCN_attention(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            gat.load_state_dict(torch.load("ACM_attention_single"+str(i)+".pth"))
            GAT.append(gat)
        self.mugcn_layers = GAT
        print("Attention single case")
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[0], adj_list[0].to_sparse(), adj_list[1].to_sparse(),features, labels, test_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(),
        test_micro_f1, test_macro_f1))
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[1], adj_list[1].to_sparse(), adj_list[1].to_sparse(),features, labels, test_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(),
        test_micro_f1, test_macro_f1))
        #test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[2], adj_list[2].to_sparse(), adj_list[2].to_sparse(),features, labels, test_mask, loss)
        #print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(),
        #test_micro_f1, test_macro_f1))
        
        #self.GCN = GCN(nfeat=in_dim, nhid=hid_dim//2,dropout=0.3,num_layers=3)
        #self.Q2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        #self.K2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        #self.V2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        self.semantic_attention = SemanticAttention(
            in_size=hid_dim,
        )
        #self.beta = torch.nn.Parameter(torch.ones(1))
        #self.projection = nn.Linear(out_dim,out_dim)
        #if self.layer_norm:
        #    self.layer_norm1 = nn.LayerNorm(hid_dim)
        #self.FFN1 = nn.Linear(hid_dim,hid_dim//2)
        #self.FFN2 = nn.Linear(hid_dim//2,out_dim)
        #self.predict = nn.Linear(hid_dim, out_dim)
        self.predict = nn.Linear(hid_dim,out_dim)
        #self.reset_parameters()
        #if self.layer_norm:
            #self.layer_norm2 = nn.LayerNorm(out_dim)
        
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
           semantic_embeddings.append(self.mugcn_layers[i].embed(adj_list[i],h))
        #print(semantic_embeddings[0].shape)
        #print(semantic_embeddings[1].shape)
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)
        
        '''global_vectors = []
        for i in range(self.num_heads):
            global_vectors.append(torch.mean(attn_out[i]))
        semantics = torch.vstack(global_vectors)
        semantic_attention = torch.softmax(semantics,torch.transpose(semantics,0,1),dim=1)
        final_embedding = torch.zeros(attn_out[0].shape)
        for i in range(self.num_heads):
            final_embedding+=semantic_attention[i,i]*attn_out[i]'''
                
        '''Q2_list = []
        K2_list = []
        V2_list = []
        for i in range(self.num_heads):
            Q2_list.append(self.Q2(attn_out[i]))
            K2_list.append(self.K2(attn_out[i]))
            V2_list.append(self.V2(attn_out[i]))
        attention = []
        normalization_list = []
        for i in range(self.num_heads):
            normal = torch.zeros(attn_out[0].shape[0],attn_out[0].shape[0])
            for j in range(self.num_heads):
                #print("Q2 shape",Q2_list[i].shape)
                #print("K2_shape",K2_list[j].shape)
                normal+=torch.exp(torch.matmul(Q2_list[i],torch.transpose(K2_list[j],0,1)))
            normalization_list.append(normal)
        #print("normalization factor",normalization_list[0].shape)
        for i in range(self.num_heads):
            temp_list = []
            for j in range(self.num_heads):
                z = torch.exp(torch.matmul(Q2_list[i],torch.transpose(K2_list[j],0,1)))
                #print("z.shape",z.shape)
                temp_list.append(torch.divide(z,normalization_list[i]))
            attention.append(temp_list)
        #print("semantic attention:",attention)
        result_list = []
        for i in range(self.num_heads):
            temp_output = torch.zeros(attn_out[i].shape)
            for j in range(self.num_heads):
                temp_output+=torch.matmul(attention[i][j],V2_list[i])+attn_out[i]
            result_list.append(temp_output)'''
        print("semantic",semantic_embeddings.shape)
        p2 = self.semantic_attention(semantic_embeddings)  # (N, D * K)
        #final_embedding = torch.cat(result_list,axis=1)
        #print(final_embedding.shape)
        #p0 = self.layer_norm1(final_embedding)
        #p1 = F.relu(self.FFN1(p0))
        #p2 = self.layer_norm2(self.FFN2(p1)+p0)
        #p2 = self.FFN2(p1)
        #print(p2.shape)
        # for i in range(attn_out.shape[0]):
        #     self.Q2(attn_out[i])
        #     self.K2(attn_out[i])
        #     self.V2(attn_out[i])
        # h = attn_out.view(-1,self.out_channels)
        # h = F.dropout(h,self.dropout,training=self.training)
        # h = self.projection(h)
        # if self.residual:
        #     h = h_in1+h
        # if self.layer_norm:
        #     h = self.layer_norm1(h)
        # h_in2 = h
        # h = self.FFN1(h)
        # h = F.relu(h)
        # h = F.dropout(h,self.dropout,trianing=self.training)
        # h = self.FFn2(h)
        # if self.residual:
        #     h = h_in2+h
        # if self.layer_norm:
        #     h = self.layer_norm2(h)
        
        # return F.softmax(self.predict(h),dim=1)
        #print(p2.shape)
        return F.softmax(self.predict(p2),dim=1)
        #return F.softmax(p2,dim=1)
