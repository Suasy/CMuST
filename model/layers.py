import torch.nn as nn
import torch
import math

class GaussianDistribution:

    def __init__(self, parameters: torch.Tensor):
        # Split mean and log of variance
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        # Clamp the log of variances
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self):
        # Sample from the distribution
        return self.mean + self.std * torch.randn_like(self.std)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Autoencoder(nn.Module):

    def __init__(self, encoder, decoder, emb_channels: int, z_channels: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, img: torch.Tensor) :
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(img)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        """
        Decode from latent representation
        """
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        return self.decoder(z)
    
    
class AttentionLayer(nn.Module):
    """
    implementation of a multi-head attention layer for neural networks.
    
    part ref https://github.com/XDZhelheim/STAEformer/blob/main/model/STAEformer.py#L6
    """

    def __init__(self, q_dim, k_dim, v_dim, atten_dim, num_heads=8, mask=False):
        super().__init__()

        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = atten_dim // num_heads

        self.FC_Q = nn.Linear(q_dim, atten_dim)
        self.FC_K = nn.Linear(k_dim, atten_dim)
        self.FC_V = nn.Linear(v_dim, atten_dim)

        self.out_proj = nn.Linear(atten_dim, v_dim)

    def forward(self, query, key, value):
        """
        output tensor after applying multi-head attention and the final linear transformation.
        """
        # dimensions
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        # transform the input
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # split the transformed inputs into multiple heads and reshape for multi-headed attention
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        # calculate attention scores
        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim**0.5

        # apply a triangular mask to ignore future tokens
        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        # weighted sum
        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value

        # concatenate outputs from different heads and final linear transformation
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    """
    implements self-attention followed by a feed-forward network, including normalization and optional dropout.
    """
    
    def __init__(
        self, x_dim, atten_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.x_dim = x_dim

        # attention and FFN
        self.attn = AttentionLayer(x_dim, x_dim, x_dim, atten_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(x_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, x_dim),
        )

        # normalization and dropout
        self.ln1 = nn.LayerNorm(x_dim)
        self.ln2 = nn.LayerNorm(x_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, x_start, dim=-2):
        """
        output tensor, with self-attention and feed-forward transformations applied to a slice of the input tensor.
        """
        _x = x # original input
        x = x.transpose(dim, -2) # transpose input for processing
        residual = x[..., x_start:x_start+self.x_dim] # residual for later addition
        x_qkv = x[..., x_start:x_start+self.x_dim] # query, key, and value from input
        
        _out = self.attn(x_qkv, x_qkv, x_qkv) # self-attention mechanism
        _out = self.dropout1(_out) # dropout
        _out = self.ln1(residual + _out) # layer normalization, and add residual
        
        residual = _out # update residual

        _out = self.feed_forward(_out) # feed-forward layer
        _out = self.dropout2(_out) # dropout
        _out = self.ln2(residual + _out) # layer normalization, and add residual

        _out = _out.transpose(dim, -2) # transpose output back to original shape
        
        # concatenate output with unchanged parts of input
        out = torch.cat((_x[..., :x_start], _out, _x[..., x_start+self.x_dim:]), dim=-1)
        
        return out


class CrossAttentionLayer(nn.Module):
    """
    Implements a cross-attention layer followed by a feed-forward network, including normalization and optional dropout. This layer
    allows for attention across different representations (queries, keys, and values from potentially different sources).
    """
    
    def __init__(
        self, q_dim, k_dim, v_dim, atten_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        # q k v dimensions
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        # attention and FFN
        self.attn = AttentionLayer(q_dim, k_dim, v_dim, atten_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(v_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, v_dim),
        )

        # normalization and dropout
        self.ln1 = nn.LayerNorm(v_dim)
        self.ln2 = nn.LayerNorm(v_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, q_start, k_start, v_start, dim=-2):
        """
        output tensor, where the cross-attention and feed-forward transformations have been applied to parts of the input tensor.
        """
        _x = x # original input
        x = x.transpose(dim, -2) # transpose input for processing

        # residual value
        residual = x[..., v_start:v_start+self.v_dim]

        # query, key, and value tensors
        query = x[..., q_start:q_start+self.q_dim]
        key = x[..., k_start:k_start+self.k_dim]
        value = x[..., v_start:v_start+self.v_dim]

        # cross-attention mechanism
        _out = self.attn(query, key, value)

        # dropout and layer normalization
        _out = self.dropout1(_out)
        _out = self.ln1(residual + _out)

        residual = _out # update residual

        _out = self.feed_forward(_out) # feed-forward layer
        _out = self.dropout2(_out) # dropout
        _out = self.ln2(residual + _out) # layer normalization, and add residual

        # Transpose output tensor and concatenate with original input
        _out = _out.transpose(dim, -2)
        out = torch.cat((_x[..., :v_start], _out, _x[..., v_start+self.v_dim:]), dim=-1)

        return out


class MSTI(nn.Module):
    """
    combines self-attention and cross-attention mechanisms to process data that includes spatial and temporal components.
    utilizes self-attention for temporal and spatial features independently and cross-attention to integrate observations with
    spatial and temporal embeddings.
    """
    def __init__(
        self, 
        input_start, spatial_start, temporal_start, 
        obser_embedding_dim, spatial_embedding_dim, temporal_embedding_dim, model_dim, 
        self_atten_dim, cross_atten_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()
        
        # start indices for different features
        self.input_start = input_start
        self.spatial_start = spatial_start
        self.temporal_start = temporal_start
        
        # Cross-attention layers for integrating observations with temporal and spatial embeddings
        self.cross_attn_x_t = CrossAttentionLayer(temporal_embedding_dim, obser_embedding_dim, obser_embedding_dim, 
                                                  cross_atten_dim, feed_forward_dim, num_heads, dropout)
        self.cross_attn_t_x = CrossAttentionLayer(obser_embedding_dim, temporal_embedding_dim, temporal_embedding_dim, 
                                                  cross_atten_dim, feed_forward_dim, num_heads, dropout)
        self.cross_attn_x_s = CrossAttentionLayer(spatial_embedding_dim, obser_embedding_dim, obser_embedding_dim, 
                                                  cross_atten_dim, feed_forward_dim, num_heads, dropout)
        self.cross_attn_s_x = CrossAttentionLayer(obser_embedding_dim, spatial_embedding_dim, spatial_embedding_dim, 
                                                  cross_atten_dim, feed_forward_dim, num_heads, dropout)
        
        # Self-attention layers for temporal and spatial processing
        self.self_attn_t_t = SelfAttentionLayer(model_dim, self_atten_dim, feed_forward_dim, num_heads, dropout)
        self.self_attn_t_t1 = SelfAttentionLayer(model_dim, self_atten_dim, feed_forward_dim, num_heads, dropout)
        self.self_attn_s_s = SelfAttentionLayer(model_dim, self_atten_dim, feed_forward_dim, num_heads, dropout)
        self.self_attn_s_s1 = SelfAttentionLayer(model_dim, self_atten_dim, feed_forward_dim, num_heads, dropout)
    
    def forward(self, x, batch_size, num_nodes, in_steps):
        """
        output tensor after processing through multiple attention layers.
        """
        # cross-attention to integrate different feature dimensions
        x = self.cross_attn_x_t(x, q_start=self.temporal_start, k_start=self.input_start, v_start=self.input_start, dim=1)
        x = self.cross_attn_t_x(x, q_start=self.input_start, k_start=self.temporal_start, v_start=self.temporal_start, dim=1)
        x = self.cross_attn_x_s(x, q_start=self.spatial_start, k_start=self.input_start, v_start=self.input_start, dim=2)
        x = self.cross_attn_s_x(x, q_start=self.input_start, k_start=self.spatial_start, v_start=self.spatial_start, dim=2)
        
        # self-attention for temporal and spatial dimensions
        x = self.self_attn_t_t(x, x_start=self.input_start, dim=1)
        x = self.self_attn_t_t1(x, x_start=self.input_start, dim=1)
        x = self.self_attn_s_s(x, x_start=self.input_start, dim=2)
        x = self.self_attn_s_s1(x, x_start=self.input_start, dim=2)
    
        return x
        