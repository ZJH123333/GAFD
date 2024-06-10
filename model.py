import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F
import numpy as np

# GIN Layer
class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())


    def forward(self, v, a):
        b, n = v.shape[0], v.shape[1]
        v_aggregate = torch.matmul(a, v)
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        v_aggregate = rearrange(v_aggregate, 'b n c -> (b n) c')
        v_combine = self.mlp(v_aggregate)
        v_combine = rearrange(v_combine, '(b n) c -> b n c', b=b, n=n)
        return v_combine

# QKV dot product
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn # context is the final results, and attn is the attention score

# Multi-head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        # d_model: The dimension of the features
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        # The first output is the results after attention, and the second results is the attention score
        return self.layer_norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # self-attention
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

# Transformer Layer
class LayerTransformer(nn.Module):
    def __init__(self, n_layers, d_model, d_k, d_v, d_ff, n_heads):
        super(LayerTransformer, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs

# Readout
class LayerReadout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# transformer+GIN block
class T_G_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, d_k, d_v, d_ff, transformer_layer=2, num_heads=1):
        super(T_G_Block, self).__init__()
        self.transformer = LayerTransformer(n_layers=transformer_layer, d_model=input_dim, d_k=d_k, d_v=d_v, d_ff=d_ff, n_heads=num_heads)
        self.gin = LayerGIN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, v, a):
        # v: batch * node * feature, a: batch * node * node
        out = self.transformer(v)
        out = self.gin(out, a)
        return out


# Feature encoder
class Extractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, d_k, d_v, d_ff, fc_dim, fc_hidden_dim, transformer_layer=2, num_heads=1):
        # input_dim: The input dimension
        # hidden_dim: The hidden layer dimension of MLP in GIN layer
        # output_dim: The output dimension of GIN layer
        super(Extractor, self).__init__()
        self.t_g_layer = T_G_Block(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                   d_k=d_k, d_v=d_v, d_ff=d_ff,
                                   transformer_layer=transformer_layer, num_heads=num_heads)
        self.fc_mlp = nn.Linear(fc_dim, fc_hidden_dim)
        self.readout = LayerReadout()
        self.t_g_mlp = nn.Linear(input_dim * output_dim, fc_hidden_dim)
        self.gamma = nn.Parameter(torch.Tensor([0.0]))
        self.project = nn.Sequential(nn.Linear(fc_hidden_dim, fc_hidden_dim//2), nn.Linear(fc_hidden_dim//2, fc_hidden_dim))

    def forward(self, v, a, fc):
        gamma = F.sigmoid(self.gamma)
        t_g_feature = self.t_g_layer(v, a)
        t_g_out = F.dropout(F.relu(self.t_g_mlp(F.dropout(self.readout(t_g_feature)))))
        fc_out = F.dropout(F.relu(self.fc_mlp(F.dropout(fc))))
        feature = gamma * t_g_out + (1 - gamma) * fc_out
        feature_project = self.project(feature)
        return feature, feature_project

# Classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, num_class):
        super(Classifier, self).__init__()
        self.mlp = nn.Linear(input_dim, num_class)

    def forward(self, x):
        return self.mlp(x)
