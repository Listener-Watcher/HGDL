#import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
#from dgl.nn.pytorch import GATConv
import torch_geometric as geo
import copy
from math import sqrt

class MultiAttentionLayer(nn.Module):
    def __init__(self, in_dim,out_dim,num_heads,use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.multi_heads = nn.ModuleList()
        for i in range(num_heads):
            if use_bias:
                Q = nn.Linear(in_dim,out_dim,bias=True)
                K = nn.Linear(in_dim,out_dim,bias=True)
                V = nn.Linear(in_dim,out_dim,bias=True)
            else:
                Q = nn.Linear(in_dim,out_dim,bias=False)
                K = nn.Linear(in_dim,out_dim,bias=False)
                V = nn.Linear(in_dim,out_dim,bias=False)
            self.multi_heads.append(nn.ModuleDict({"Q":Q,"K":K,"V":V}))
    def forward(self,adj_list,h):
        X_result = []
        for i in range(self.num_heads):
            Q_h = self.multi_heads[i]["Q"](h)
            K_h = self.multi_heads[i]["K"](h)
            V_h = self.multi_heads[i]["V"](h)
            #print("multi-attention layer:Qh",Q_h.shape)
            #print("multi-attention layer:Kh",K_h.shape)
            A_tilde = torch.mul((1/sqrt(h.shape[1]))*torch.torch.matmul(Q_h,torch.transpose(K_h,0,1)),adj_list[i])
            #print("A_tilde",A_tilde.shape)
            attention = torch.nn.functional.softmax(A_tilde,dim=1)
            #print("attention",attention)
            #print("attention check",torch.sum(attention[0]))
            #print("attention check",torch.sum(attention[1]))
            #print("V_h",V_h.shape)
            X_tilde = torch.matmul(attention,V_h)
            #print("X_tilde",X_tilde.shape)
            X_result.append(X_tilde)
        return X_result
        #X_final = torch.cat(X_result,axis=1)
        #print(X_final.shape)
        #return X_final

class Gtransformerblock(nn.Module):
    def __init__(
        self,in_dim,hid_dim,out_dim,num_heads, dropout=0.0, layer_norm=True, use_bias=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm        
        self.attentionlayer = MultiAttentionLayer(in_dim,hid_dim//num_heads,num_heads,use_bias)
        self.Q2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        self.K2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        self.V2 = nn.Linear(hid_dim//num_heads,hid_dim//num_heads,bias=True)
        #self.beta = torch.nn.Parameter(torch.ones(1))
        #self.projection = nn.Linear(out_dim,out_dim)
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(hid_dim)
        self.FFN1 = nn.Linear(hid_dim,out_dim*2)
        self.FFN2 = nn.Linear(out_dim*2,out_dim)
        #if self.layer_norm:
            #self.layer_norm2 = nn.LayerNorm(out_dim)

    def forward(self, adj_list, h):
        attn_out = self.attentionlayer(adj_list,h)
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
        result_list = attn_out
        final_embedding = torch.cat(result_list,axis=1)
        #print(final_embedding.shape)
        p0 = self.layer_norm1(final_embedding)
        p1 = F.relu(self.FFN1(final_embedding))
        #p2 = self.layer_norm2(self.FFN2(p1)+p0)
        p2 = self.FFN2(p1)
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
        return F.softmax(p2,dim=1)
