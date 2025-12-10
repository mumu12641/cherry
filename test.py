import numpy as np


def softmax(x):
    # 数值稳定的 Softmax
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# --- 模拟环境设置 ---
SEQ_LEN = 10  # 假设最大序列长度 (相当于 1024)
DIM = 4  # 维度 (相当于 768)
POS = 2  # 当前生成到第 2 个词 (即 cache 中 0, 1, 2 是有效的)

# 1. 初始化全量 Cache (模拟显存中预分配好的大数组)
# 后面未使用的部分全是 0
k_cache = np.zeros((SEQ_LEN, DIM))
v_cache = np.zeros((SEQ_LEN, DIM))

# 2. 填入一些模拟数据 (模拟之前的推理过程)
# 只有前 POS+1 个位置有真实数据
valid_data_k = np.random.randn(POS + 1, DIM)
valid_data_v = np.random.randn(POS + 1, DIM)
k_cache[: POS + 1] = valid_data_k
v_cache[: POS + 1] = valid_data_v

# 3. 当前的 Query (当前 token)
q = np.random.randn(1, DIM)

print(f"当前 POS: {POS}")
print(f"Cache 形状: {k_cache.shape} (全量)")

# ==========================================
# 方案二：全量计算 + Mask (不切片)
# ==========================================

# 步骤 1: 全量矩阵乘法
# q: [1, DIM], k_cache.T: [DIM, SEQ_LEN]
# 结果 scores: [1, SEQ_LEN] -> 包含有效分数和无效分数(0)
raw_scores = np.matmul(q, k_cache.T)

print(f"\n1. 原始分数 (Raw Scores): \n{raw_scores}")
# 注意：你会看到后面全是 0.0，因为 k_cache 后面全是 0

# 步骤 2: 创建 Mask
# 规则：<= POS 的位置是 0 (保留)，> POS 的位置是 -inf (屏蔽)
mask = np.full((1, SEQ_LEN), float("-inf"))
mask[0, : POS + 1] = 0.0
mask[0, :POS+1] = 0.0

print(f"\n2. Mask: \n{mask}")

# 步骤 3: 应用 Mask
masked_scores = raw_scores + mask

print(f"\n3. Mask 后分数: \n{masked_scores}")
# 注意：后面的 0.0 变成了 -inf

# 步骤 4: Softmax
attn_weights = softmax(masked_scores)

print(f"\n4. 注意力权重 (Softmax后): \n{attn_weights}")
# 关键点：
# 有效位置 (0, 1, 2): 有正常的概率值 (和为 1)
# 无效位置 (3 到 9): 全是 0.0 (因为 e^-inf = 0)

# 步骤 5: 全量加权求和
# weights: [1, SEQ_LEN], v_cache: [SEQ_LEN, DIM]
# 虽然 v_cache 后面全是 0，且 weights 后面也全是 0，计算结果是安全的
output = np.matmul(attn_weights, v_cache)

print(f"\n5. 最终输出 (Output): \n{output}")
