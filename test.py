import numpy as np

# ==========================================
# 0. 配置与常量 (Configuration)
# ==========================================
N_LAYERS = 12
DIM = 768
FFN_DIM = 2048
HEAD_NUM = 12
HEAD_DIM = 64
VOCAB_SIZE = 32000
MAX_SEQ = 1024
EPS = 1e-5
SCALE_VAL = 0.125  # 1 / sqrt(64)

# ==========================================
# 1. 基础算子实现 (Basic Operators)
# ==========================================

def rms_norm(x, weight, eps=EPS):
    # x: [..., dim], weight: [dim]
    mean_square = np.mean(x**2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(mean_square + eps)
    return x * rsqrt * weight

def silu(x):
    # x * sigmoid(x)
    return x * (1.0 / (1.0 + np.exp(-x)))

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def apply_rope(x, pos):
    # x: [1, head_num, head_dim]
    # 简单的 RoPE 实现，对应 MLIR 中 implied 的逻辑
    # 假设 head_dim 是偶数
    batch, heads, d = x.shape
    half_d = d // 2
    
    # 生成 theta
    # theta_i = 10000 ^ (-2*i / d)
    indices = np.arange(half_d, dtype=np.float32)
    theta = 10000.0 ** (-2 * indices / d)
    
    # 生成 cos/sin 表
    # angle = pos * theta
    angles = pos * theta
    cos_val = np.cos(angles)
    sin_val = np.sin(angles)
    
    # 执行旋转 [x_even, x_odd]
    # x_out_even = x_even * cos - x_odd * sin
    # x_out_odd  = x_even * sin + x_odd * cos
    x_even = x[:, :, :half_d]
    x_odd  = x[:, :, half_d:]
    
    out_even = x_even * cos_val - x_odd * sin_val
    out_odd  = x_even * sin_val + x_odd * cos_val
    
    return np.concatenate([out_even, out_odd], axis=-1)

def generate_mask(pos, max_len):
    # MLIR: cherry.generate_mask
    # 创建一个 [1, max_len] 的掩码
    # 0.0 表示可见，-inf 表示遮蔽
    mask = np.full((1, max_len), -1e9, dtype=np.float32)
    mask[0, :pos+1] = 0.0
    return mask

# ==========================================
# 2. 模型前向传播 (Llama Forward)
# ==========================================

def llama_forward(token_id, pos, k_cache, v_cache, mask, 
                  embedding_table, 
                  rms_att_weight, wq, wk, wv, wo, 
                  rms_ffn_weight, w1, w2, w3, 
                  rms_final_weight, wcls):
    
    # 1. Embedding Lookup
    # x = embedding_table[token_id] -> [1, 768]
    x = embedding_table[token_id].reshape(1, DIM)
    
    # 2. Transformer Layers Loop
    for i in range(N_LAYERS):
        # 保存残差连接的输入
        curr_x = x 
        
        # ================= Part A: Attention Block =================
        
        # A.1 RMS Norm
        # weight slice: rms_att_weight[i]
        xb = rms_norm(curr_x, rms_att_weight[i])
        
        # A.2 Q, K, V Projections
        # MatMul: [1, 768] @ [768, 768]
        q_raw = xb @ wq[i]
        k_raw = xb @ wk[i]
        v_raw = xb @ wv[i]
        
        # A.3 RoPE
        # Reshape to heads: [1, 12, 64]
        q_heads = q_raw.reshape(1, HEAD_NUM, HEAD_DIM)
        k_heads = k_raw.reshape(1, HEAD_NUM, HEAD_DIM)
        v_heads = v_raw.reshape(1, HEAD_NUM, HEAD_DIM)
        
        q_rope = apply_rope(q_heads, pos)
        k_rope = apply_rope(k_heads, pos)
        # v 不需要 RoPE
        
        # Reshape back to flat for consistency with MLIR flow (though we split again later)
        q = q_rope.reshape(1, DIM)
        k = k_rope.reshape(1, DIM)
        v = v_heads.reshape(1, DIM)
        
        # A.4 Update KV Cache
        # Cache shape: [12, 1024, 768]
        # Update layer i, position pos
        k_cache[i, pos, :] = k.flatten()
        v_cache[i, pos, :] = v.flatten()
        
        # A.5 Multi-Head Attention
        # 获取当前层完整的 KV (直到 max_len)
        k_layer_full = k_cache[i] # [1024, 768]
        v_layer_full = v_cache[i] # [1024, 768]
        
        att_out_list = []
        
        # 模拟 scf.for %h = 0 to 12 (Heads Loop)
        for h in range(HEAD_NUM):
            offset = h * HEAD_DIM
            
            # Slice Head Q: [1, 64]
            q_head = q_rope[:, h, :] 
            
            # Slice Head K: [1024, 64] -> Transpose -> [64, 1024]
            # 注意：这里取的是整个 buffer，但实际上只有 0..pos 是有效的
            # MLIR 代码是取了整个 [1024, 64]，然后靠 Mask 屏蔽掉无效区域
            k_head_full = k_layer_full[:, offset:offset+HEAD_DIM]
            k_head_T = k_head_full.T 
            
            # Score: [1, 64] @ [64, 1024] -> [1, 1024]
            score = q_head @ k_head_T
            
            # Add Mask
            score = score + mask
            
            # Scale
            score_scaled = score * SCALE_VAL
            
            # Softmax
            probs = softmax(score_scaled, axis=1)
            
            # Context: [1, 1024] @ [1024, 64] -> [1, 64]
            v_head_full = v_layer_full[:, offset:offset+HEAD_DIM]
            context_head = probs @ v_head_full
            
            att_out_list.append(context_head)
            
        # Merge Heads: [1, 12, 64] -> [1, 768]
        att_out_final = np.concatenate(att_out_list, axis=-1)
        
        # A.6 Output Projection
        xb2 = att_out_final @ wo[i]
        
        # A.7 Residual Connection
        x_resid_1 = curr_x + xb2
        
        # ================= Part B: FFN Block =================
        
        # B.1 RMS Norm
        xb_ffn = rms_norm(x_resid_1, rms_ffn_weight[i])
        
        # B.2 Projections (Gate w1, Up w3)
        # w1: [768, 2048], w3: [768, 2048]
        hb = xb_ffn @ w1[i]  # Gate
        hb2 = xb_ffn @ w3[i] # Up
        
        # B.3 Activation (SiLU)
        hb_silu = silu(hb)
        
        # B.4 Element-wise Mul
        hb_mul = hb_silu * hb2
        
        # B.5 Down Projection (w2)
        # w2: [2048, 768]
        ffn_out = hb_mul @ w2[i]
        
        # B.6 Residual Connection
        x = x_resid_1 + ffn_out # Update x for next layer
        
    # 3. Final Block
    x_norm = rms_norm(x, rms_final_weight)
    
    # Logits: [1, 768] @ [768, 32000]
    # wcls in MLIR is [32000, 768], needs transpose to be [768, 32000]
    logits = x_norm @ wcls.T
    
    return logits, k_cache, v_cache

# ==========================================
# 3. Host 程序 (Main Execution)
# ==========================================

def host():
    print("Initializing Weights...")
    # 初始化权重 (使用 MLIR 中的 dense 值)
    # Embedding
    embedding = np.full((VOCAB_SIZE, DIM), 2.0, dtype=np.float32)
    
    # Attention Weights [12, 768, ...]
    rms_att = np.full((N_LAYERS, DIM), 3.0, dtype=np.float32)
    wq = np.full((N_LAYERS, DIM, DIM), 4.0, dtype=np.float32)
    wk = np.full((N_LAYERS, DIM, DIM), 5.0, dtype=np.float32)
    wv = np.full((N_LAYERS, DIM, DIM), 6.0, dtype=np.float32)
    wo = np.full((N_LAYERS, DIM, DIM), 7.0, dtype=np.float32)
    
    # FFN Weights
    rms_ffn = np.full((N_LAYERS, DIM), 8.0, dtype=np.float32)
    w1 = np.full((N_LAYERS, DIM, FFN_DIM), 9.0, dtype=np.float32) # Gate
    w2 = np.full((N_LAYERS, FFN_DIM, DIM), 10.0, dtype=np.float32) # Down
    w3 = np.full((N_LAYERS, DIM, FFN_DIM), 11.0, dtype=np.float32) # Up
    
    # Final Weights
    rms_final = np.full((DIM,), 12.0, dtype=np.float32)
    wcls = np.full((VOCAB_SIZE, DIM), 13.0, dtype=np.float32)
    
    # KV Cache Init (Zeros)
    k_cache = np.zeros((N_LAYERS, MAX_SEQ, DIM), dtype=np.float32)
    v_cache = np.zeros((N_LAYERS, MAX_SEQ, DIM), dtype=np.float32)
    
    # Generation Loop
    curr_token = 1 # Start Token
    curr_pos = 0
    max_len = 10
    
    print("Starting Generation Loop...")
    
    while curr_pos < max_len:
        # Generate Mask
        mask = generate_mask(curr_pos, MAX_SEQ)
        
        # Call Forward
        logits, k_cache, v_cache = llama_forward(
            curr_token, curr_pos, k_cache, v_cache, mask,
            embedding,
            rms_att, wq, wk, wv, wo,
            rms_ffn, w1, w2, w3,
            rms_final, wcls
        )
        
        # Argmax
        next_token = np.argmax(logits, axis=1)[0]
        print(logits)
        # Print Logits (Sum to check validity without printing massive array)
        print(f"Step {curr_pos}: Token={curr_token}, Next={next_token}, Logits Sum={np.sum(logits):.4f}")
        
        # Update Loop Vars
        curr_token = next_token
        curr_pos += 1
        
    print("Generation Finished.")

if __name__ == "__main__":
    host()
