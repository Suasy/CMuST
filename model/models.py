import torch.nn as nn
import torch
import copy
from torchinfo import summary

from model.layers import *

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class CMuST(nn.Module):
    """
    applies attention mechanisms both temporally and spatially within a dataset. It is particularly designed for scenarios
    where data has inherent spatial and temporal attributes,
    multi-dimensional time series forecasting.
    """
    def __init__(self, num_nodes, in_steps=12, out_steps=12, steps_per_day=144,
        obser_dim=3, output_dim=1, obser_embedding_dim=24, tod_embedding_dim=24, dow_embedding_dim=24, timestamp_embedding_dim=12,
        spatial_embedding_dim=12, temporal_embedding_dim=48, prompt_dim=72,
        self_atten_dim=24, cross_atten_dim=24, feed_forward_dim=256, num_heads=4,
        num_layers=1, dropout=0.1,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        
        self.obser_dim = obser_dim
        self.output_dim = output_dim
        self.obser_embedding_dim = obser_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.temporal_embedding_dim = temporal_embedding_dim
        self.prompt_dim = prompt_dim
        self.self_atten_dim=self_atten_dim
        self.cross_atten_dim=cross_atten_dim
        
        self.model_dim = obser_embedding_dim + spatial_embedding_dim + temporal_embedding_dim + prompt_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.input_start = 0
        self.spatial_start = obser_embedding_dim
        self.temporal_start = obser_embedding_dim + spatial_embedding_dim
        
        self.obser_mlp = nn.Linear(obser_dim, obser_embedding_dim)
        self.timestamp_mlp = nn.Linear(6, timestamp_embedding_dim)
        self.spatial_mlp = nn.Linear(2, spatial_embedding_dim)
        self.temporal_mlp = nn.Linear(timestamp_embedding_dim+dow_embedding_dim+tod_embedding_dim, temporal_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.prompt = nn.init.xavier_uniform_(nn.Parameter(torch.empty(in_steps, num_nodes, prompt_dim)))
        self.output_mlp = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        # self.output_mlp = nn.Linear(in_steps * self.obser_embedding_dim, out_steps * output_dim)
        self.layers = clones(MSTI(self.input_start, self.spatial_start, self.temporal_start, 
                                          obser_embedding_dim, spatial_embedding_dim, temporal_embedding_dim, self.model_dim,
                                          self_atten_dim, cross_atten_dim, feed_forward_dim, num_heads, dropout), N=num_layers)

    def forward(self, x):
        
        # extract features
        batch_size = x.shape[0]
        tod = x[..., 1]
        dow = x[..., 2]
        coor = x[..., 3:5]
        timestamp = x[..., 5:11]
        
        obser = x[..., : self.obser_dim]
        obser_emb = self.obser_mlp(obser)  # (batch_size, in_steps, num_nodes, obser_embedding_dim)
        features = [obser_emb]

        spatial_emb = self.spatial_mlp(coor) # (batch_size, in_steps, num_nodes, spatial_embedding_dim)
        features.append(spatial_emb)
        
        tod_emb = self.tod_embedding((tod * self.steps_per_day).long())  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
        dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
        timestamp_emb = self.timestamp_mlp(timestamp)   # (batch_size, in_steps, num_nodes, timestamp_embedding_dim)
        temporal_emb = self.temporal_mlp(torch.cat((timestamp_emb, tod_emb, dow_emb), dim=-1))   # (batch_size, in_steps, num_nodes, temporal_embedding_dim)
        features.append(temporal_emb)
        
        expanded_prompt = self.prompt.expand(size=(batch_size, *self.prompt.shape))
        features.append(expanded_prompt)
        
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        
        # MSTI
        for layer in self.layers:
            x = layer(x, batch_size, self.num_nodes, self.in_steps)
        
        # output
        out = x.transpose(1, 2).reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        out = self.output_mlp(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        
        # _x = x[..., self.input_start:self.input_start+self.obser_embedding_dim]
        # out = _x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, obser_embedding_dim)
        # out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.obser_embedding_dim)
        # out = self.output_mlp(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        # out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = CMuST(300, 12, 12)
    # DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    # model = model.to(DEVICE)
    summary(model, [16, 12, 300, 11])
    
    # from torch.autograd import profiler
    # with profiler.profile(use_cuda=True) as prof:
        # model(x)
        # summary(model, [16, 12, 207, 3])

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
