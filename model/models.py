import torch.nn as nn
import torch
import copy
import torch.nn.functional as F

from model.layers import *

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class CMuST(nn.Module):
    """
    applies attention mechanisms both temporally and spatially within a dataset. It is particularly designed for scenarios
    where data has inherent spatial and temporal attributes,
    multi-dimensional time series forecasting.
    
    part ref https://github.com/zezhishao/STID/blob/master/stid/stid_arch/stid_arch.py
           & https://github.com/joshsohn/STAEformer-final-project/blob/main/model/Spacetimeformer.py
    """
    def __init__(self, num_nodes, input_len=12, output_len=12, tod_size=48,
        obser_dim=3, output_dim=1, obser_embed_dim=24, tod_embed_dim=24, dow_embed_dim=24, timestamp_embed_dim=12,
        spatial_embed_dim=12, temporal_embed_dim=48, prompt_dim=72,
        self_atten_dim=24, cross_atten_dim=24, feed_forward_dim=256, num_heads=4, dropout=0.1,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.tod_size = tod_size
        
        self.obser_dim = obser_dim
        self.output_dim = output_dim
        self.obser_embed_dim = obser_embed_dim
        self.tod_embed_dim = tod_embed_dim
        self.dow_embed_dim = dow_embed_dim
        self.spatial_embed_dim = spatial_embed_dim
        self.temporal_embed_dim = temporal_embed_dim
        self.prompt_dim = prompt_dim
        self.self_atten_dim=self_atten_dim
        self.cross_atten_dim=cross_atten_dim
        
        self.model_dim = obser_embed_dim + spatial_embed_dim + temporal_embed_dim + prompt_dim
        self.num_heads = num_heads
        
        self.input_start = 0
        self.spatial_start = obser_embed_dim
        self.temporal_start = obser_embed_dim + spatial_embed_dim
        
        self.obser_mlp = nn.Linear(obser_dim, obser_embed_dim)
        self.timestamp_mlp = nn.Linear(6, timestamp_embed_dim)
        self.spatial_mlp = nn.Linear(2, spatial_embed_dim)
        self.temporal_mlp = nn.Linear(timestamp_embed_dim+dow_embed_dim+tod_embed_dim, temporal_embed_dim)
        self.tod_embedding = nn.Embedding(tod_size, tod_embed_dim)
        self.dow_embedding = nn.Embedding(7, dow_embed_dim)
        self.prompt = nn.Parameter(torch.empty(input_len, num_nodes, prompt_dim))
        nn.init.xavier_uniform_(self.prompt)
        
        # fusion & regression
        # self.conv_o = nn.Conv2d(obser_embed_dim, obser_embed_dim, kernel_size=1)
        # self.conv_s = nn.Conv2d(spatial_embed_dim, spatial_embed_dim, kernel_size=1)
        # self.conv_t = nn.Conv2d(temporal_embed_dim, temporal_embed_dim, kernel_size=1)
        # self.W_z = nn.Linear(obser_embed_dim+spatial_embed_dim+temporal_embed_dim, self.model_dim)
        # self.W_p = nn.Linear(prompt_dim, self.model_dim)
        # self.FC_y = nn.Linear(input_len * self.model_dim, output_len * output_dim)
        self.conv_o = nn.Conv2d(obser_embed_dim, 256, kernel_size=1)
        self.conv_s = nn.Conv2d(spatial_embed_dim, 256, kernel_size=1)
        self.conv_t = nn.Conv2d(temporal_embed_dim, 256, kernel_size=1)
        self.W_z = nn.Linear(256, self.model_dim)
        self.W_p = nn.Linear(prompt_dim, self.model_dim)
        self.FC_y = nn.Linear(input_len * self.model_dim, output_len * output_dim)
        
        # output layer
        # self.output_mlp = nn.Linear(input_len * self.model_dim, output_len * output_dim)
        self.layers = clones(MSTI(self.input_start, self.spatial_start, self.temporal_start, 
                                          obser_embed_dim, spatial_embed_dim, temporal_embed_dim, self.model_dim,
                                          self_atten_dim, cross_atten_dim, feed_forward_dim, num_heads, dropout), N=1)

    def forward(self, x):
        
        # extract features
        batch_size = x.shape[0]
        tod = x[..., 1]
        dow = x[..., 2]
        coor = x[..., 3:5]
        timestamp = x[..., 5:11]
        
        obser = x[..., : self.obser_dim]
        obser_emb = self.obser_mlp(obser)  # (batch_size, input_len, num_nodes, obser_embed_dim)
        features = [obser_emb]

        spatial_emb = self.spatial_mlp(coor) # (batch_size, input_len, num_nodes, spatial_embed_dim)
        features.append(spatial_emb)
        
        tod_emb = self.tod_embedding((tod * self.tod_size).long())  # (batch_size, input_len, num_nodes, tod_embed_dim)
        dow_emb = self.dow_embedding(dow.long())  # (batch_size, input_len, num_nodes, dow_embed_dim)
        timestamp_emb = self.timestamp_mlp(timestamp)   # (batch_size, input_len, num_nodes, timestamp_embed_dim)
        temporal_emb = self.temporal_mlp(torch.cat((timestamp_emb, tod_emb, dow_emb), dim=-1))   # (batch_size, input_len, num_nodes, temporal_embed_dim)
        features.append(temporal_emb)
        
        expanded_prompt = self.prompt.expand(size=(batch_size, *self.prompt.shape))
        features.append(expanded_prompt)
        
        x = torch.cat(features, dim=-1)  # (batch_size, input_len, num_nodes, model_dim)
        
        # fusion & regression
        H_o = x.transpose(1, -1)[:, :self.spatial_start, :, :]  # (batch_size, obser_embedding_dim, num_nodes, input_len)
        H_s = x.transpose(1, -1)[:, self.spatial_start:self.temporal_start, :, :]  # (batch_size, spatial_embedding_dim, num_nodes, input_len)
        H_t = x.transpose(1, -1)[:, self.temporal_start:self.prompt_start, :, :]  # (batch_size, temporal_embedding_dim, num_nodes, input_len)
        
        # Z_o = self.conv_o(H_o)  # (batch_size, obser_embedding_dim, num_nodes, input_len)
        # Z_s = self.conv_s(H_s)  # (batch_size, spatial_embedding_dim, num_nodes, input_len)
        # Z_t = self.conv_t(H_t)  # (batch_size, temporal_embedding_dim, num_nodes, input_len)

        # Z = torch.cat((Z_o, Z_s, Z_t), dim=1)  # (batch_size, obser_embedding_dim+spatial_embedding_dim+temporal_embedding_dim, num_nodes, input_len)
        # Z = Z.transpose(1, -1) # (batch_size, input_len, num_nodes, obser_embedding_dim+spatial_embedding_dim+temporal_embedding_dim)

        Z_o = self.conv_o(H_o)  # (batch_size, obser_embedding_dim, num_nodes, input_len)
        Z_s = self.conv_s(H_s)  # (batch_size, spatial_embedding_dim, num_nodes, input_len)
        Z_t = self.conv_t(H_t)  # (batch_size, temporal_embedding_dim, num_nodes, input_len)

        Z = F.relu(Z_o) + F.relu(Z_s) + F.relu(Z_t)  # (batch_size, obser_embedding_dim+spatial_embedding_dim+temporal_embedding_dim, num_nodes, input_len)
        Z = Z.transpose(1, -1) # (batch_size, input_len, num_nodes, obser_embedding_dim+spatial_embedding_dim+temporal_embedding_dim)

        H_p = F.relu(x[:, :, :, self.prompt_start:])  # (batch_size, input_len, num_nodes, prompt_dim)

        # regression
        out_ = F.relu(self.W_z(Z)) + F.relu(self.W_p(H_p))
        out_ = out_.transpose(1, 2).reshape(batch_size, self.num_nodes, self.input_len * self.model_dim)
        out = self.FC_y(out_).view(batch_size, self.num_nodes, self.output_len, self.output_dim)
        out = out.transpose(1, 2)  # (batch_size, output_len, num_nodes, output_dim)

        return out
    
        # transform output and reshape for prediction
        # out = x.transpose(1, 2).reshape(batch_size, self.num_nodes, self.input_len * self.model_dim)
        # out = self.output_mlp(out).view(batch_size, self.num_nodes, self.output_len, self.output_dim)
        # out = out.transpose(1, 2)  # (batch_size, output_len, num_nodes, output_dim)

        # return out

        return out
