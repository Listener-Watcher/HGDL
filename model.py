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


def evaluate(model, g,features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits,_ = model(g, features)
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
    def __init__(self, nfeat, nhid,nclass,dropout,num_layers,adj_list,attention_dim):
        super(GCN_attention_v2, self).__init__()
        self.Q = nn.Linear(nhid,nhid,bias=True)
        self.K = nn.Linear(nhid,nhid,bias=True)
        self.V = nn.Linear(nhid,nhid,bias=True)
        self.gcn1=GraphConvolution(nfeat,nhid)
        self.gcn2=GraphConvolution(nhid, nclass)
        self.layer_norm = nn.LayerNorm(nhid)
        #self.weight = Parameter(torch.FloatTensor(adj_list[0].shape[0],3)).to("cuda:0")
        self.atten = nn.Linear(adj_list[0].shape[0],attention_dim)
        self.atten2 = nn.Linear(adj_list[0].shape[0],attention_dim)
        self.atten3 = nn.Linear(adj_list[0].shape[0],attention_dim)
        self.agg = nn.Linear(attention_dim*len(adj_list),len(adj_list))
        self.dropout = dropout
        #self.memory_unit = torch.FloatTensor([[1/3,1/3,1/3,0,1,0]]*adj_list[0].shape[0]).to('cuda:0')
        #self.memory_unit = torch.FloatTensor([[0,1]]*adj_list[0].shape[0]).to('cuda:0')
        #self.memory_adj = torch.FloatTensor(adj_list[0].shape).to('cuda:0')
        self.gcn = GraphConvolution(nhid,nhid)
    def forward(self,adj_list,x):
        #self.weight = torch.softmax(self.weight,dim=1)
        #print("weight",self.weight)
        Adj_kernel = []
        for i in range(len(adj_list)):
            Adj_kernel.append(adj_list[i])
        z1 = self.atten(Adj_kernel[0])
        z2 = self.atten2(Adj_kernel[1])
        z3 = self.atten3(Adj_kernel[2])
        z4 = ((self.agg(torch.cat((z1,z2,z3),dim=1))))
        #z4 = self.agg(torch.cat((z1,z2),dim=1))
        #print("stack shape",z4.shape)
        nz = torch.softmax(z4,dim=1)
        #uniform = torch.FloatTensor([[0,1]]*adj_list[0].shape[0]).to('cuda:0')
        #uniform = torch.FloatTensor([[0.5,0.5]]*adj_list[0].shape[0]).to('cuda:0')
        #print("nz away from uniform",torch.sum(torch.abs(nz-uniform)))
        #print("nz update",torch.sum(torch.abs(nz-self.memory_unit)))
        #print("nz away from 0,1",torch.sum(nz[:,1]>0.5)/nz.shape[0])
        #print(nz)
        #self.memory_unit = nz
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
        #random_mask = torch.rand(adj_list[0].shape,device="cuda:0")<0.3
        #X_tilde = F.relu(self.gcn(x,torch.mul(adj.to_dense(),random_mask).to_sparse()))
        #X_tilde = F.relu(self.gcn(x,adj))
        z = self.gcn2(X_tilde,adj)
        nj = torch.softmax(z4,dim=1)
        return F.softmax(z,dim=1),nj
        
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

class Gtransformerblock(nn.Module):
    def __init__(
        self,in_dim,hid_dim,out_dim,attention_dim,num_heads,adj_list,features,labels, train_mask,val_mask,test_mask,device,dropout=0.0, layer_norm=True, use_bias=False,gamma=0.0001):
        super().__init__()
        self.in_channels = in_dim
        self.hid_channels = hid_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.adj_list = adj_list
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask        
        self.device = device
        gat = GCN_attention_v2(in_dim,hid_dim,out_dim,dropout,1,adj_list,attention_dim).to(self.device)
        gat.train()
        optimizer = torch.optim.Adam(list(gat.parameters()),lr=0.005,weight_decay=0)
        loss = KLDivLoss(reduction="batchmean")
        best_loss = 100
        patience = 0
        best_model = None
        uniform = torch.FloatTensor([[1/len(adj_list)]*len(adj_list)]*adj_list[0].shape[0]).to('cuda:0')
        for epoch in range(2000):
            optimizer.zero_grad()
            output,nj = gat.forward(adj_list,features)
            kl_loss = loss((output[train_mask]+eps).log(),(labels[train_mask]+eps))
            kl_loss_val = loss((output[val_mask]+eps).log(),(labels[val_mask]+eps))
            #uniform = torch.FloatTensor([[1/3,1/3,1/3]]*adj_list[0].shape[0]).to('cuda:0')
            #uniform = torch.FloatTensor([[0.5,0.5]]*adj_list[0].shape[0]).to('cuda:0')
            con_loss = torch.sum(torch.abs(nj-uniform))
            (kl_loss-gamma*con_loss).backward()
            #kl_loss.backward()
            optimizer.step()
            if kl_loss_val<best_loss:
                best_loss = kl_loss_val
                patience = 0
                best_model= copy.deepcopy(gat.state_dict())
            else:
                patience+=1
                print("patience:",patience)
            if patience>150:
                break
            print("val loss",kl_loss_val.item())
        print("best val loss",best_loss.item())
        #torch.save(best_model,"DBLP_sample_our_method.pth")
        gat.load_state_dict(best_model)
        print("HGDL")
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate2(gat, adj_list,features, labels, test_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(), test_micro_f1, test_macro_f1))
        
        
        
        """
        GAT = []
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
        best_gat = [None for i in range(self.num_heads)]
        patience = 0
        best_epoch = 0
        #Label_corr = []
        #Kl_loss = []
        best_val = 100
        #weight = torch.tensor([1,0]).to(self.device)
        #weight = torch.tensor([0.33333,0.33333,0.33333]).to(self.device)
        for epoch in range(2000):
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

                kl_loss = loss((output[train_mask]+eps).log(),(labels[train_mask]+eps))
                kl_loss_val = loss((output[val_mask]+eps).log(),(labels[val_mask]+eps))

                Kl_loss.append(kl_loss)
                Kl_loss_val.append(kl_loss_val)

            total_loss = sum(Kl_loss)
            for i in range(self.num_heads):
                (Kl_loss[i]).backward()
            #(Kl_loss[1]).backward()
            #(Kl_loss[2]).backward()
            optimizer.step()
            val_loss = sum(Kl_loss_val)
            if best_val>val_loss:
                best_val = val_loss
                for i in range(self.num_heads):
                    best_gat[i] = copy.deepcopy(GAT[i].state_dict())
                best_epoch = epoch
                patience = 0
            else:
            	patience+=1
            if patience>=100:
            	print("best_val",best_val)
            	print("early stop")
            	break
            print("best val loss:",best_val)
            for i in range(self.num_heads):
               print("val loss:",Kl_loss_val[i])
            print(patience)
            #print("epoch",epoch)
            #print("training loss",total_loss.item())
        
        GAT = []
        for i in range(self.num_heads):
            gat = GCN_attention(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            gat.load_state_dict(best_gat[i])
            GAT.append(gat)
            #torch.save(GAT[i].state_dict(),"ACM_attention_single"+str(i)+".pth")
        '''GAT = []
        for i in range(self.num_heads):
            gat = GCN_attention(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            gat.load_state_dict(torch.load("GAT"+str(i)+"_drug_811_32_.pth"))
            GAT.append(gat)'''
        self.mugcn_layers = GAT
        print("Attention single case")
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[0], adj_list[0].to_sparse(), adj_list[1].to_sparse(),features, labels, test_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(),
        test_micro_f1, test_macro_f1))
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[1], adj_list[1].to_sparse(), adj_list[1].to_sparse(),features, labels, test_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(),
        test_micro_f1, test_macro_f1))
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[2], adj_list[2].to_sparse(), adj_list[2].to_sparse(),features, labels, test_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(),
        test_micro_f1, test_macro_f1))
        """
       
        
        """
        GAT = []
        parameters = []
        #Optimizer = []
        for i in range(self.num_heads):
            gat = GCN_attention2(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
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
        for epoch in range(2000):
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
                output = GAT[i].forward(adj_list[i].to_sparse(),features)

                kl_loss = loss((output[train_mask]+eps).log(),(labels[train_mask]+eps))
                kl_loss_val = loss((output[val_mask]+eps).log(),(labels[val_mask]+eps))

                Kl_loss.append(kl_loss)
                Kl_loss_val.append(kl_loss_val)

            total_loss = sum(Kl_loss)

            (Kl_loss[0]).backward()
            (Kl_loss[1]).backward()
            optimizer.step()
            val_loss = sum(Kl_loss_val)
            if best_val>val_loss:
                best_val = val_loss
                for i in range(self.num_heads):
                    best_gat[i] = copy.deepcopy(GAT[i].state_dict())
                best_epoch = epoch
                patience = 0
            else:
            	patience+=1
            if patience>=200:
            	print("best_val",best_val)
            	print("early stop")
            	break
            #print("best val loss:",best_val)
            #print("val loss 1:",Kl_loss_val[0])
            #print("val loss 2:",Kl_loss_val[1])
            #print("epoch",epoch)
            #print("training loss",total_loss.item())
        
        GAT = []
        for i in range(self.num_heads):
            gat = GCN_attention2(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            gat.load_state_dict(best_gat[i])
            GAT.append(gat)
            #torch.save(GAT[i].state_dict(),"GAT"+str(i)+"_drug_811_32_.pth")
        '''GAT = []
        for i in range(self.num_heads):
            gat = GCN_attention(in_dim, hid_dim,out_dim,dropout,1).to(self.device)
            gat.load_state_dict(torch.load("GAT"+str(i)+"_drug_811_32_.pth"))
            GAT.append(gat)'''
        self.mugcn_layers = GAT
        print("GCN attention single case")
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[0], adj_list[0].to_sparse(), adj_list[1].to_sparse(),features, labels, test_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(),
        test_micro_f1, test_macro_f1))
        test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[1], adj_list[1].to_sparse(), adj_list[1].to_sparse(),features, labels, test_mask, loss)
        print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(),
        test_micro_f1, test_macro_f1))
        """
        
        
        #test_loss, test_acc, test_micro_f1, test_macro_f1,_ = evaluate(self.mugcn_layers[2], adj_list[2].to_sparse(),,features, labels, test_mask, loss)
        #print("Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_loss.item(),
        #test_micro_f1, test_macro_f1))
        
        #self.GCN = GCN(nfeat=in_dim, nhid=hid_dim//2,dropout=0.3,num_layers=3)
        #self.Q2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        #self.K2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        #self.V2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        #self.semantic_attention = SemanticAttention(
        #    in_size=out_dim,
        #)
        #self.beta = torch.nn.Parameter(torch.ones(1))
        #self.projection = nn.Linear(out_dim,out_dim)
        #if self.layer_norm:
        #    self.layer_norm1 = nn.LayerNorm(hid_dim)
        #self.FFN1 = nn.Linear(hid_dim,hid_dim//2)
        #self.FFN2 = nn.Linear(hid_dim//2,out_dim)
        #self.predict = nn.Linear(hid_dim, out_dim)
        #self.predict = nn.Linear(hid_dim,out_dim)
        #self.reset_parameters()
        #if self.layer_norm:
            #self.layer_norm2 = nn.LayerNorm(out_dim)
        
    def forward(self, adj_list, h):
        #result_list = []
        #for i in range(self.num_heads):
        #   result_list.append(self.GCN(h,adj_list[i].to_sparse()))
        semantic_embeddings = []
        for i in range(self.num_heads):
           semantic_embeddings.append(self.mugcn_layers[i].embed(adj_list[i],h))

        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)
        print("semantic",semantic_embeddings.shape)
        p2 = self.semantic_attention(semantic_embeddings)  # (N, D * K)
        return F.softmax(p2,dim=1)
