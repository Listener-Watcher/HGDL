#import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
#from dgl.nn.pytorch import GATConv
import torch_geometric as geo
import copy


class MultiAttentionLayer(nn.Module):
    def __init__(self, in_dim,out_dim,num_heads,use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim,out_dim*num_heads,bias=True)
            self.K = nn.Linear(in_dim,out_dim*num_heads,bias=True)
            self.V = nn.Linear(in_dim,out_dim*num_heads,bias=True)
        else:
            self.Q = nn.Linear(in_dim,out_dim*num_heads,bias=False)
            self.K = nn.Linear(in_dim,out_dim*num_heads,bias=False)
            self.V = nn.Linear(in_dim,out_dim*num_heads,bias=False)
    def forward(self,adj_list,h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        print(Q_h.shape)
        print(K_h.shape)
        A_tilde = torch.mul((1/torch.sqrt(h.shape[1]))*torch.torch.matmul(Q_h,torch.transpose(K_h,0,1)),adj_list[0])
        print("A_tilde",A_tilde.shape)
        attention = torch.nn.functional.softmax(A_tilde,dim=1)
        print("attention",attention.shape)
        print("V_h",V_h.shape)
        X_tilde = torch.matmul(attention,V_h)
        print("X_tilde",X_tilde.shape)
        return X_tilde

class Gtransformerblock(nn.Module):
    def __init__(
        self,in_dim, out_dim, num_heads, dropout=0.0, layer_norm=True, use_bias=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm        
        self.attentionlayer = MultiAttentionLayer(in_dim,out_dim//num_heads,num_heads,use_bias)
        self.Q2 = nn.Linear(out_dim,out_dim,bias=True)
        self.K2 = nn.Linear(out_dim,out_dim,bias=True)
        self.V2 = nn.Linear(out_dim,out_dim,bias=True)
        self.projection = nn.Linear(out_dim,out_dim)
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
        self.FFN1 = nn.Linear(out_dim,out_dim*2)
        self.FFN2 = nn.Linear(out_dim*2,out_dim)
        if self.layer_norm:
            self.layer_norm2 = nn.Layer_nrom(out_dim)

    def forward(self, adj_list, h):
        h_in1 = h
        attn_out = self.attentionlayer(adj_list,h)
        print("multi-head",attn_out.shape)
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
        return attn_out