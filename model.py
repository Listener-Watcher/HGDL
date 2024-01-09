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
        logits = model(g, features,g2)
    loss = loss_func((logits[mask]+1e-9).log(), labels[mask]+1e-9)
    accuracy, micro_f1, macro_f1,score_intersection = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1,score_intersection


def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size())
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
        X_tilde = self.layer_norm(X_tilde)
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
        X_tilde = self.layer_norm(X_tilde)
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
        self.atten = nn.Linear(adj_list[0].shape[0],1)
        self.dropout = dropout
        #self.memory_unit = torch.FloatTensor([[1/3,1/3,1/3,0,1,0]]*adj_list[0].shape[0]).to('cuda:0')
        self.memory_unit = torch.FloatTensor([[1/2,1/2]]*adj_list[0].shape[0]).to('cuda:0')
        #self.memory_adj = torch.FloatTensor(adj_list[0].shape).to('cuda:0')
        self.gcn=GraphConvolution(nhid,nhid)
    def forward(self,adj_list,x,adj_list_origin):
        #self.weight = torch.softmax(self.weight,dim=1)
        #print("weight",self.weight)
        Adj_kernel = []
        for i in range(len(adj_list)):
            #Adj_kernel.append(torch.stack((adj_list[i],adj_list_origin[i]),dim=1))
            Adj_kernel.append(adj_list[i])
        z1 = self.atten(Adj_kernel[0])
        z2 = self.atten(Adj_kernel[1])
        #z3 = self.atten(Adj_kernel[2])
        #z4 = torch.stack((z1,z2,z3),dim=1).squeeze()
        z4 = torch.stack((z1,z2),dim=1).squeeze()
        #print("stack shape",z4.shape)
        nz = torch.softmax(z4,dim=1)
        #uniform = torch.FloatTensor([[1/3,1/3,1/3,0,1,0]]*adj_list[0].shape[0]).to('cuda:0')
        uniform = torch.FloatTensor([[1/2,1/2]]*adj_list[0].shape[0]).to('cuda:0')
        print("nz away from uniform",torch.sum(torch.abs(nz-uniform)))
        print("nz update",torch.sum(torch.abs(nz-self.memory_unit)))
        print(nz)
        self.memory_unit = nz
        #adj = nz[:,0]*adj_list[0]+nz[:,1]*adj_list[1]+nz[:,2]*adj_list[2]
        adj = nz[:,0]*adj_list[0]+nz[:,1]*adj_list[1]
        #adj = gcn_norm(adj)
        adj = adj.to_sparse()
        x = F.relu(self.gcn1(x,adj))
        x = F.relu(self.gcn(x,adj))
        '''Q_h = self.Q(x)
        K_h = torch.transpose(self.K(x),0,1)
        V_h = self.V(x)
        #A_tilde = sparse_dense_mul(adj,torch.matmul(Q_h,K_h))
        A_tilde = torch.mul(adj.to_dense(),torch.matmul(Q_h,K_h))
        attention = F.softmax(A_tilde,dim=1)   
        #attention = torch.sparse.softmax(A_tilde,dim=1)     
        X_tilde = torch.matmul(gcn_norm(attention),V_h)
        X_tilde = self.layer_norm(X_tilde)'''
        X_tilde = x
        z = self.gcn2(X_tilde,adj)
        return F.softmax(z,dim=1) 
        
        
        
      
        
        
class GCN(nn.Module):
    def __init__(self, nfeat, nhid,nclass,dropout,num_layers):
        super(GCN, self).__init__()
        self.layers = []
        self.layers.append(GraphConvolution(nfeat,nhid))
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers.append(GraphConvolution(nhid,nhid))

        self.layers.append(GraphConvolution(nhid, nclass))
        self.dropout = dropout
        self.layers = torch.nn.ModuleList(self.layers)
    def forward(self,adj,x):
        for i,conv in enumerate(self.layers):
            if i != len(self.layers)-1:
                x = F.relu(conv(x,adj))
                x = F.dropout(x,self.dropout,training=self.training)
            else:
                x = conv(x,adj)
        return F.softmax(x,dim=1)
    @torch.no_grad()
    def embed(self,adj,x):
        self.eval()
        for i,conv in enumerate(self.layers):
            if i != len(self.layers)-1:
                x = F.relu(conv(x,adj))
                #x = F.dropout(x,self.dropout,training=self.training)
            else:
                x = conv(x,adj)
        print("GAT",x.shape)
        #print("GAT2",h.flatten(1).shape)
        return x
    @torch.no_grad()
    def embed2(self,adj,x):
        self.eval()
        for i,conv in enumerate(self.layers):
            if i != len(self.layers)-1:
                x = F.relu(conv(x,adj))
                #x = F.dropout(x,self.dropout,training=self.training)
        print("GAT",x.shape)
        #print("GAT2",h.flatten(1).shape)
        return x





class MultiAttentionLayer(nn.Module):
    def __init__(self, in_dim,hid_dim,out_dim,num_heads,use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.multi_heads = nn.ModuleList()
        for i in range(num_heads):
            if use_bias:
                Q = nn.Linear(in_dim,hid_dim,bias=True)
                K = nn.Linear(in_dim,hid_dim,bias=True)
                V = nn.Linear(in_dim,hid_dim,bias=True)
            else:
                Q = nn.Linear(in_dim,hid_dim,bias=False)
                K = nn.Linear(in_dim,hid_dim,bias=False)
                V = nn.Linear(in_dim,hid_dim,bias=False)
            self.multi_heads.append(nn.ModuleDict({"Q":Q,"K":K,"V":V}))
            self.O = nn.Linear(hid_dim*2,in_dim,bias=True)
            #self.FF1 = nn.Linear(in_dim,in_dim//2,bias=True)
            #self.FF2 = nn.Linear(in_dim//2,in_dim,bias=True)
            self.layer_norm1 = nn.LayerNorm(in_dim)
            #self.layer_norm2 = nn.LayerNorm(in_dim)
            self.predict = nn.Linear(in_dim,out_dim)
    '''def reset_parameters(self):
        for i in range(num_heads):
            nn.init.xavier_uniform(self.multi_heads[i]["Q"].weight)
            nn.init.xavier_uniform(self.multi_heads[i]["K"].weight)
            nn.init.xavier_uniform(self.multi_heads[i]["V"].weight)
            #self.FFN1.weight.bias.data.fill_(0.01)'''
    @torch.no_grad()
    def embed(self,adj_list,h):
        X_result = []
        for i in range(self.num_heads):
            Q_h = self.multi_heads[i]["Q"](h)
            K_h = self.multi_heads[i]["K"](h)
            V_h = self.multi_heads[i]["V"](h)
            #print("multi-attention layer:Qh",Q_h.shape)
            #print("multi-attention layer:Kh",K_h.shape)
            #print("h shape",h.shape[1])
            A_tilde = torch.mul(torch.torch.matmul(Q_h,torch.transpose(K_h,0,1)),adj_list[i])
            #A_tilde = torch.matmul(Q_h,torch.transpose(K_h,0,1))
            #print("A_tilde",A_tilde.shape)
            attention = torch.nn.functional.softmax(A_tilde,dim=1)
            #attention = adj_norm(attention)
            #nn.functional.dropout(attention,0.5,training=True)
            #print("attention",attention)
            #print("attention check",torch.sum(attention[0]))
            #print("attention check",torch.sum(attention[1]))
            #print("V_h",V_h.shape)
            X_tilde = torch.matmul(attention,V_h)
            #print("X_tilde",X_tilde.shape)
            X_result.append(X_tilde)
        return X_result
    def forward(self,adj_list,h):
        X_result = []
        for i in range(self.num_heads):
            Q_h = self.multi_heads[i]["Q"](h)
            K_h = self.multi_heads[i]["K"](h)
            V_h = self.multi_heads[i]["V"](h)
            #print("multi-attention layer:Qh",Q_h.shape)
            #print("multi-attention layer:Kh",K_h.shape)
            #print("h shape",h.shape[1])
            A_tilde = torch.mul(torch.torch.matmul(Q_h,torch.transpose(K_h,0,1)),adj_list[i])
            #A_tilde = torch.matmul(Q_h,torch.transpose(K_h,0,1))
            #print("A_tilde",A_tilde.shape)
            attention = torch.nn.functional.softmax(A_tilde,dim=1)
            #attention = adj_norm(attention)
            #nn.functional.dropout(attention,0.5,training=True)
            #print("attention",attention)
            #print("attention check",torch.sum(attention[0]))
            #print("attention check",torch.sum(attention[1]))
            #print("V_h",V_h.shape)
            X_tilde = torch.matmul(attention,V_h)
            #print("X_tilde",X_tilde.shape)
            X_result.append(X_tilde)
        X_final = torch.cat(X_result,axis=1)
        M = self.layer_norm1(self.O(X_final))
        #K = self.FF2(F.relu(self.FF1(M)))
        #Z = self.layer_norm2(M+K)
        return F.softmax(self.predict(M),dim=1)
        #return X_result
        #X_final = torch.cat(X_result,axis=1)
        #print(X_final.shape)
        #return X_final

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
        '''self.attentionlayer = MultiAttentionLayer(in_dim,hid_dim,out_dim,num_heads,use_bias).to('self.device')
        loss_fcn = torch.nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.attentionlayer.parameters(), lr=0.001, weight_decay=0)
        best_loss = 100
        patience = 0
        for epoch in range(300):
            self.attentionlayer.train()
            logits = self.attentionlayer.forward(adj_list, features)
            loss = loss_fcn((logits[train_mask]+1e-9).log(), labels[train_mask]+1e-9)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val_loss = loss_fcn((logits[val_mask]+1e-9).log(), labels[val_mask]+1e-9)
            if val_loss<best_loss:
                best_loss = val_loss
            else:
                patience+=1
            if patience>=10:
                print("early stop")
                break
            print("val loss",val_loss.item())'''
            
            
            
        """GAT = []
        #Optimizer = []
        parameters = []
        for i in range(self.num_heads):
            gat = GCN(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            gat.train()
            #optimizer = torch.optim.Adam(list(gat.parameters()),lr=0.005,weight_decay=0)
            GAT.append(gat)
            parameters+=list(GAT[i].parameters())
        optimizer = torch.optim.Adam(parameters,lr=0.005,weight_decay = 0)
        loss = KLDivLoss(reduction="batchmean")
        #optimizer = torch.optim.Adam(list(gat.parameters()),lr=0.005,weight_decay=0)
        best_gat = [None,None]
        patience = 0
        best_epoch = 0
        #Label_corr = []
        #Kl_loss = []
        best_val = 100
        weight = torch.tensor([0.9986,0.0014]).to(self.device)
        #weight = torch.tensor([0.5,0.5]).to(self.device)
        for epoch in range(2000):
            optimizer.zero_grad()
            Label_corr = []
            Kl_loss = []
            #Output = []
            Kl_loss_val = []
            for i in range(self.num_heads):
                output = GAT[i].forward(adj_list[i].to_sparse(),features)
                #Output.append(GAT[i].embed(adj_list[i].to_sparse(),features))
                kl_loss = loss((output[train_mask]+eps).log(),(labels[train_mask]+eps))
                kl_loss_val = loss((output[val_mask]+eps).log(),(labels[val_mask]+eps))
                label_corr = torch.matmul(torch.t(output),output)
                Label_corr.append(label_corr)
                Kl_loss.append(kl_loss)
                Kl_loss_val.append(kl_loss_val)
            total_loss = sum(Kl_loss)
            estimated_anchor = torch.zeros(Label_corr[0].shape).to(self.device)
            print("estimated_anchor shape",estimated_anchor.shape)
            for i in range(self.num_heads):
                estimated_anchor+= weight[i]*Label_corr[i]
            consistency_loss = torch.zeros(1).to(self.device)
            #for i in range(self.num_heads):
            #    consistency_loss+=torch.linalg.matrix_norm(torch.abs(Label_corr[i]-estimated_anchor),ord='fro')
            for i in range(self.num_heads):
                consistency_loss+=torch.linalg.matrix_norm(torch.abs(Label_corr[i]),ord='fro')
            print("consistency_loss",consistency_loss.item())
            #(Kl_loss[0]+0.00001*consistency_loss-0.001*divergence_loss).backward(retain_graph=True)
            #(Kl_loss[1]+0.00001*consistency_loss-0.001*divergence_loss).backward()
            #(Kl_loss[0]-0.001*divergence_loss).backward(retain_graph=True)
            #(Kl_loss[1]-0.001*divergence_loss).backward()           
            #(Kl_loss[0]+0.00001*consistency_loss).backward(retain_graph=True)
            #(Kl_loss[1]+0.00001*consistency_loss).backward()           
            #print("consistency_loss",consistency_loss)
            #print("divergence_loss",divergence_loss)
            '''estimated_anchor = torch.zeros(label_corr[0].shape).to(self.device)
            for i in range(self.num_heads):
                estimated_anchor+= weight[i]*label_corr[i]
            consistency_loss = torch.zeros(1).to(self.device)
            for i in range(self.num_heads):
                consistency_loss+=torch.linalg.matrix_norm(torch.abs(Label_corr[i]-estimated_anchor),ord='fro')
            print("consistency_loss",consistency_loss.item())'''
            #print(0.00001*torch.linalg.matrix_norm(torch.abs(Label_corr[ic]-Label_corr[jc]),ord='fro').item())
            #(Kl_loss[0]+0.00001*consistency_loss-0.001*divergence_loss).backward(retain_graph=True)
            #(Kl_loss[1]+0.00001*consistency_loss-0.001*divergence_loss).backward()
            #(Kl_loss[0]-0.001*divergence_loss).backward(retain_graph=True)
            #(Kl_loss[1]-0.001*divergence_loss).backward()           
            #(Kl_loss[0]+0.00001*consistency_loss).backward(retain_graph=True)
            #(Kl_loss[1]+0.00001*consistency_loss).backward()           
            #print("consistency_loss",consistency_loss)
            #print("divergence_loss",divergence_loss)
            (Kl_loss[0]+0.0001*consistency_loss).backward(retain_graph=True)
            (Kl_loss[1]+0.0001*consistency_loss).backward()
            #Kl_loss[0].backward()
            #Kl_loss[1].backward()
            optimizer.step()
            #val_loss = evaluate(new_g,features,labels,val_mask,gat)
            val_loss = sum(Kl_loss_val)
            #print("without reg:",val_loss)
            #val_loss = val_loss+0.0001*consistency_loss-0.0001*divergence_loss
            if best_val>val_loss:
                best_val = val_loss
                best_gat[0] = copy.deepcopy(GAT[0].state_dict())
                best_gat[1] = copy.deepcopy(GAT[1].state_dict())
                best_epoch = epoch
                patience = 0
            else:
            	patience+=1
            if patience>=100:
            	print("best_val",best_val)
            	print("early stop")
            	break
            print("best val loss:",best_val)
            print("epoch",epoch)
            print("training loss",total_loss.item())
        GAT = []
        for i in range(self.num_heads):
            gat = GCN(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            gat.load_state_dict(best_gat[i])
            GAT.append(gat)
        self.mugcn_layers = GAT"""
        

        gat = GCN_attention_v2(in_dim,hid_dim,out_dim,dropout,1,adj_list,adj_list_origin).to(self.device)
        gat.train()
        optimizer = torch.optim.Adam(list(gat.parameters()),lr=0.005,weight_decay=0)
        loss = KLDivLoss(reduction="batchmean")
        best_loss = 100
        patience = 0
        best_model = None
        #memory = torch.FloatTensor([[1/3,1/3,1/3]]*adj_list[0].shape[0]).to('cuda:0')
        #memory_loss_val = 100
        for epoch in range(1000):
            optimizer.zero_grad()
            output = gat.forward(adj_list,features,adj_list_origin)
            kl_loss = loss((output[train_mask]+eps).log(),(labels[train_mask]+eps))
            kl_loss_val = loss((output[val_mask]+eps).log(),(labels[val_mask]+eps))
            #kl_loss.backward(retain_graph=True)
            #if kl_loss_val<memory_loss_val:
            #    (kl_loss+torch.sum(torch.abs(memory-nz))).backward()
            #else:
            #    kl_loss.backward()
            kl_loss.backward()
            #memory=nz
            optimizer.step()
            if kl_loss_val<best_loss:
                best_loss = kl_loss_val
                patience = 0
                best_model= copy.deepcopy(gat.state_dict())
            else:
                patience+=1
                print("patience:",patience)
            if patience>50:
                break
            print("val loss",kl_loss_val.item())
        gat.load_state_dict(best_model)
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(gat, adj_list, adj_list_origin,features, labels, val_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(), test_micro_f1, test_macro_f1))
        '''GAT = []
        parameters = []
        #Optimizer = []
        for i in range(self.num_heads):
            gat = GCN_attention(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            gat.train()
            #optimizer = torch.optim.Adam(list(gat.parameters()),lr=0.005,weight_decay=0)
            #Optimizer.append(optimizer)
            GAT.append(gat)
            parameters+=list(GAT[i].parameters())
        optimizer = torch.optim.Adam(parameters,lr=0.005,weight_decay = 0)
        loss = KLDivLoss(reduction="batchmean")
        #optimizer = torch.optim.Adam(list(gat.parameters()),lr=0.005,weight_decay=0)
        best_gat = [None]*self.num_heads
        patience = 0
        best_epoch = 0
        #Label_corr = []
        #Kl_loss = []
        best_val = 100
        weight = torch.tensor([1,0]).to(self.device)
        #weight = torch.tensor([0.33333,0.33333,0.33333]).to(self.device)
        for epoch in range(1000):
            Label_corr = []
            #Label_corr2 = []
            Kl_loss = []
            Output = []
            Kl_loss_val = []
            Consistency_loss = []
            Indices = []
            #random_mask = torch.rand(self.features.shape[0], device="cuda") < 0.9
            optimizer.zero_grad()
            for i in range(self.num_heads):
                #Optimizer[i].zero_grad()
                output = GAT[i].forward(adj_list[i].to_sparse(),features)
                #Output.append(GAT[i].embed(adj_list[i].to_sparse(),features))
                Output.append(output)
                print("max index",torch.argmax(output,dim=1))
                mask = (torch.argmax(output[train_mask],dim=1)==torch.argmax(labels[train_mask],dim=1))
                Indices.append(mask)
                print(torch.sum(mask).item()/mask.shape[0])
                kl_loss = loss((output[train_mask]+eps).log(),(labels[train_mask]+eps))
                kl_loss_val = loss((output[val_mask]+eps).log(),(labels[val_mask]+eps))
                #label_corr = F.softmax(torch.matmul(torch.t(output[train_mask][mask]),output[train_mask][mask]),dim=1)
                #label_corr = output
                #Label_corr.append(label_corr)
                #Label_corr2.append(torch.matmul(torch.t(torch.matmul(adj_list[i],output)),torch.matmul(adj_list[i],output)))
                Kl_loss.append(kl_loss)
                Kl_loss_val.append(kl_loss_val)
            anchor_mask = torch.logical_and(Indices[0],Indices[1])
            print("total correct for model 2",torch.sum(Indices[1]))
            print("overlapped",torch.sum(anchor_mask))
            nonoverlap_mask = []
            for i in range(self.num_heads):
                mask = torch.logical_xor(Indices[i],anchor_mask)
                nonoverlap_mask.append(mask)
                Label_corr.append(F.softmax(torch.matmul(torch.t(Output[i][train_mask][mask]),Output[i][train_mask][mask]),dim=1))
            #Label_corr.append(torch.matmul(torch.t(Output[0]),Output[1]))
            total_loss = sum(Kl_loss)
            #print("Label corr matrix:",Label_corr)
            #estimated_anchor = torch.zeros(Label_corr[0].shape).to(self.device)
            #print("estimated_anchor shape",estimated_anchor.shape)
            #for i in range(len(Label_corr)):
            #    estimated_anchor+= weight[i]*Label_corr[i]
            #consistency_loss = torch.zeros(1).to(self.device)
            consistency_loss = torch.linalg.matrix_norm(torch.abs(F.softmax(torch.matmul(torch.t(Output[0][train_mask][nonoverlap_mask[1]]),Output[0][train_mask][nonoverlap_mask[1]]),dim=1)-Label_corr[1]),ord='fro')
            Consistency_loss.append(consistency_loss)
            consistency_loss = torch.linalg.matrix_norm(torch.abs(F.softmax(torch.matmul(torch.t(Output[1][train_mask][nonoverlap_mask[0]]),Output[1][train_mask][nonoverlap_mask[0]]),dim=1)-Label_corr[0]),ord='fro')
            Consistency_loss.append(consistency_loss)
            #Consistency_loss = []
            #for i in range(self.num_heads):
            #    consistency_loss=torch.linalg.matrix_norm(torch.abs(Label_corr[i]-Label_corr2[i]),ord='fro')
            #    Consistency_loss.append(consistency_loss)
            #print("consistency_loss",Consistency_loss[0])
            #print("consistency_loss",Consistency_loss[0].item())
            #print("consistency_loss",Consistency_loss[1].item())
            #print(0.00001*torch.linalg.matrix_norm(torch.abs(Label_corr[ic]-Label_corr[jc]),ord='fro').item())
            #(Kl_loss[0]+0.00001*consistency_loss-0.001*divergence_loss).backward(retain_graph=True)
            #(Kl_loss[1]+0.00001*consistency_loss-0.001*divergence_loss).backward()
            #(Kl_loss[0]-0.001*divergence_loss).backward(retain_graph=True)
            #(Kl_loss[1]-0.001*divergence_loss).backward()           
            #(Kl_loss[0]+0.00001*consistency_loss).backward(retain_graph=True)
            #(Kl_loss[1]+0.00001*consistency_loss).backward()           
            #print("consistency_loss",consistency_loss)
            #print("divergence_loss",divergence_loss)
            #(Kl_loss[0]+0.5*Consistency_loss[0]).backward(retain_graph=True)
            #(Kl_loss[1]+0.5*Consistency_loss[1]).backward()
            (Kl_loss[0]+0.1*Consistency_loss[0]).backward(retain_graph=True)
            (Kl_loss[1]+0.1*Consistency_loss[1]).backward()
            #(Kl_loss[0]).backward()
            #(Kl_loss[1]).backward()
            optimizer.step()
            #Optimizer[0].step()
            #Optimizer[1].step()
            #val_loss = evaluate(new_g,features,labels,val_mask,gat)
            val_loss = sum(Kl_loss_val)
            #print("without reg:",val_loss)
            #val_loss = val_loss+0.0001*consistency_loss-0.0001*divergence_loss
            if best_val>val_loss:
                best_val = val_loss
                best_gat[0] = copy.deepcopy(GAT[0].state_dict())
                best_gat[1] = copy.deepcopy(GAT[1].state_dict())
                best_epoch = epoch
                patience = 0
            else:
            	patience+=1
            if patience>=100:
            	print("best_val",best_val)
            	print("early stop")
            	break
            print("best val loss:",best_val)
            print("val loss 1:",Kl_loss_val[0])
            print("val loss 2:",Kl_loss_val[1])
            print("epoch",epoch)
            #print("training loss",total_loss.item())
        '''
        GAT = []
        for i in range(self.num_heads):
            gat = GCN_attention(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            #gat.load_state_dict(best_gat[i])
            GAT.append(gat)
        #torch.save(GAT[0].state_dict(),"GAT0_APA_1000_con.pth")
        #torch.save(GAT[1].state_dict(),"GAT1_APA_1000_con.pth")
        #GAT[0].load_state_dict(torch.load("GAT0_APA_1000.pth"))
        #GAT[1].load_state_dict(torch.load("GAT1_APA_1000.pth"))
        self.mugcn_layers = GAT
        
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[1], adj_list[1].to_sparse(), features, labels, val_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(), test_micro_f1, test_macro_f1))
        #self.GCN = GCN(nfeat=in_dim, nhid=hid_dim//2,dropout=0.3,num_layers=3)
        #self.Q2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        #self.K2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        #self.V2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        self.semantic_attention = SemanticAttention(
            in_size=out_dim,
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



    '''def train_model(mode,weight):
        if mode == 'GCN':
            GAT = []
            parameters = []
            for i in range(self.num_heads):
                gat = GCN(in_dim, hid_dim,out_dim,dropout,1).to('cuda:0')
                gat.train()
                #optimizer = torch.optim.Adam(list(gat.parameters()),lr=0.005,weight_decay=0)
                GAT.append(gat)
                parameters+=list(GAT[i].parameters())
            optimizer = torch.optim.Adam(parameters,lr=0.005,weight_decay = 0)
            loss = KLDivLoss(reduction="batchmean")
            #optimizer = torch.optim.Adam(list(gat.parameters()),lr=0.005,weight_decay=0)
            best_gat = [None]*self.num_heads
            patience = 0
            best_epoch = 0
            best_val = 100
            for epoch in range(2000):
                optimizer.zero_grad()
                Label_corr = []
                Kl_loss = []
                Output = []
                Kl_loss_val = []
                for i in range(self.num_heads):
                    output = GAT[i].forward(adj_list[i].to_sparse(),features)
                    Output.append(GAT[i].embed(adj_list[i].to_sparse(),features))
                    kl_loss = loss((output[train_mask]+eps).log(),(labels[train_mask]+eps))
                    kl_loss_val = loss((output[val_mask]+eps).log(),(labels[val_mask]+eps))
                    label_corr = torch.matmul(torch.t(output),output)
                    Label_corr.append(label_corr)
                    Kl_loss.append(kl_loss)
                    Kl_loss_val.append(kl_loss_val)
                total_loss = sum(Kl_loss)
                #for ic in range(len(Label_corr)):
                #    for jc in range(ic+1,len(Label_corr)):
                        #print("label corr match")
                 #       consistency_loss = torch.linalg.matrix_norm(torch.abs(Label_corr[ic]-Label_corr[jc]),ord='fro')
                 #       divergence_loss = (torch.linalg.matrix_norm(torch.abs(Output[0]-Output[1]),ord='fro'))
                        #print(0.00001*torch.linalg.matrix_norm(torch.abs(Label_corr[ic]-Label_corr[jc]),ord='fro').item())
                #(Kl_loss[0]+0.00001*consistency_loss-0.001*divergence_loss).backward(retain_graph=True)
                #(Kl_loss[1]+0.00001*consistency_loss-0.001*divergence_loss).backward()
                #(Kl_loss[0]-0.001*divergence_loss).backward(retain_graph=True)
                #(Kl_loss[1]-0.001*divergence_loss).backward()           
                #(Kl_loss[0]+0.00001*consistency_loss).backward(retain_graph=True)
                #(Kl_loss[1]+0.00001*consistency_loss).backward()           
                #print("consistency_loss",consistency_loss)
                #print("divergence_loss",divergence_loss)
                Kl_loss[0].backward(retain_graph=True)
                Kl_loss[1].backward()
                optimizer.step()
                #val_loss = evaluate(new_g,features,labels,val_mask,gat)
                val_loss = sum(Kl_loss_val)
                #print("without reg:",val_loss)
                #val_loss = val_loss+0.0001*consistency_loss-0.0001*divergence_loss
                if best_val>val_loss:
                    best_val = val_loss
                    best_gat[0] = copy.deepcopy(GAT[0].state_dict())
                    best_gat[1] = copy.deepcopy(GAT[1].state_dict())
                    best_epoch = epoch
                    patience = 0
                else:
                    patience+=1
                if patience>=100:
                    print("best_val",best_val)
                    print("early stop")
                    break
                print("best val loss:",best_val)
                print("epoch",epoch)
                print("training loss",total_loss.item())
            GAT = []
            for i in range(self.num_heads):
                gat = GCN(in_dim, hid_dim,out_dim,dropout,1).to('cuda:0')
                gat.load_state_dict(best_gat[i])
                GAT.append(gat)
            self.mugcn_layers = GAT'''
        
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
        #return F.softmax(self.predict(p2),dim=1)
        return F.softmax(p2,dim=1)
