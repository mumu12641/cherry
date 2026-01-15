import torch
import torch.nn.functional as F
import math
import time

# 检查是否有 GPU，如果有则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ==========================================
# 1. 数据准备
# ==========================================
BATCH_SIZE = 1
SEQ_LEN_KV = 1024
HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64

# 创建数据并移动到对应设备 (CPU 或 GPU)
q_input = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device)       # [1, 768]
k_input = torch.randn(SEQ_LEN_KV, HIDDEN_SIZE, device=device)       # [1024, 768]
v_input = torch.randn(SEQ_LEN_KV, HIDDEN_SIZE, device=device)       # [1024, 768]
scale_val = 1.0 / math.sqrt(HEAD_DIM)

# ==========================================
# 2. 函数定义
# ==========================================

def mha_loop(q, k, v, scale):
    head_outputs = []
    for i in range(NUM_HEADS):
        start = i * HEAD_DIM
        end = (i + 1) * HEAD_DIM
        
        # Slice (切片)
        q_head = q[:, start:end]          # [1, 64]
        k_head = k[:, start:end]          # [1024, 64]
        v_head = v[:, start:end]          # [1024, 64]
        
        # Attention
        score = torch.matmul(q_head, k_head.transpose(0, 1)) * scale
        probs = F.softmax(score, dim=-1)
        context = torch.matmul(probs, v_head)
        
        head_outputs.append(context)
        
    return torch.cat(head_outputs, dim=-1)

def mha_vectorized(q, k, v, scale):
    # Reshape Q: [1, 768] -> [1, 12, 64] -> [12, 1, 64]
    q_heads = q.view(1, NUM_HEADS, HEAD_DIM).transpose(0, 1)
    
    # Reshape K: [1024, 768] -> [1024, 12, 64] -> [12, 64, 1024]
    k_heads = k.view(-1, NUM_HEADS, HEAD_DIM).permute(1, 2, 0)
    
    # Reshape V: [1024, 768] -> [1024, 12, 64] -> [12, 1024, 64]
    v_heads = v.view(-1, NUM_HEADS, HEAD_DIM).transpose(0, 1)
    
    # Batch Matmul
    scores = torch.matmul(q_heads, k_heads) * scale
    probs = F.softmax(scores, dim=-1)
    context_heads = torch.matmul(probs, v_heads)
    
    # Restore shape
    return context_heads.transpose(0, 1).contiguous().view(1, HIDDEN_SIZE)

# ==========================================
# 3. 性能测试函数
# ==========================================

def benchmark(func, args, name, n_repeat=1000):
    # 1. 预热 (Warmup) - 运行 10 次不计时
    for _ in range(10):
        func(*args)
    
    # 确保 GPU 完成预热任务
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # 2. 开始计时
    start_time = time.time()
    
    for _ in range(n_repeat):
        func(*args)
        
    # 确保 GPU 完成所有任务
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_repeat * 1000  # 转换为毫秒
    print(f"[{name}] Average time: {avg_time:.4f} ms")
    return avg_time

# ==========================================
# 4. 运行对比
# ==========================================

print(f"\nStarting Benchmark ({1000} iterations)...")

# 测试循环版本
time_loop = benchmark(mha_loop, (q_input, k_input, v_input, scale_val), "Loop Version")

# 测试向量化版本
time_vec = benchmark(mha_vectorized, (q_input, k_input, v_input, scale_val), "Vectorized Version")

# 计算加速比
speedup = time_loop / time_vec
print(f"\nSpeedup: {speedup:.2f}x")

# 验证正确性
out_loop = mha_loop(q_input, k_input, v_input, scale_val)
out_vec = mha_vectorized(q_input, k_input, v_input, scale_val)
if torch.allclose(out_loop, out_vec, atol=1e-5):
    print("Correctness Check: PASS")
else:
    print("Correctness Check: FAIL")
