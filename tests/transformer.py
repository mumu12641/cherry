import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Attention 参数
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Feed Forward 参数
        self.W_ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.W_ff2 = nn.Linear(d_ff, d_model, bias=False)

        # LayerNorm 参数
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [Batch, SeqLen, d_model]

        # --- 1. Self-Attention ---
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Score = Q @ K.T / sqrt(d_k)
        # K.transpose(-2, -1) 交换最后两维
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(Q.size(-1))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Add & Norm
        x = self.norm1(x + attn_output)

        # --- 2. Feed Forward ---
        # Linear -> Relu -> Linear
        ff_out = self.W_ff2(F.relu(self.W_ff1(x)))

        # Add & Norm
        x = self.norm2(x + ff_out)

        return x


# 参数设置
B, Seq, Dim = 1, 4, 8
model = SimpleTransformerBlock(d_model=Dim, d_ff=32)
input_tensor = torch.randn(B, Seq, Dim)
output = model(input_tensor)
print(output)
print(output.shape)  # torch.Size([1, 4, 8])

import torch

A = torch.randn(1, 2, 3, 4, 5)
B = torch.randn(1, 2, 5, 6)

C = torch.matmul(A, B)  # 或者使用 A @ B
print(C.shape)
