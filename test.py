import torch
import numpy as np

def solve_llama_shape_mismatch():
    dim = 768
    
    # 1. 模拟数据
    # x: 模拟输入向量 (1, 768)
    x = torch.randn(1, dim)
    
    # wq: 模拟权重矩阵 (768, 768)
    # 假设这是从 .bin 文件里读出来的原始顺序
    wq_raw = torch.randn(dim, dim)
    
    # ==========================================
    # 模拟 llama2.c 的实现 (C语言逻辑)
    # ==========================================
    # llama2.c 把 x 当作列向量，遍历 W 的每一行与 x 做点积
    # out[i] = sum(W[i][j] * x[j])
    print("正在计算 llama2.c 风格的结果...")
    llama_out = torch.zeros(1, dim)
    
    # 为了演示原理，这里用矩阵运算模拟 C 的行处理逻辑:
    # (768, 768) @ (768, 1) -> (768, 1) -> reshape (1, 768)
    llama_out = torch.matmul(wq_raw, x.T).reshape(1, dim)
    
    # ==========================================
    # 你的实现: matmul(x, wq)
    # ==========================================
    print("正在计算你的原始结果 (x @ wq)...")
    # 错误的方式：直接乘
    my_out_wrong = torch.matmul(x, wq_raw)
    
    # 正确的方式：必须对 W 进行转置！
    # 这样 W 的“列”就变成了原来的“行”
    print("正在计算修正后的结果 (x @ wq.T)...")
    my_out_correct = torch.matmul(x, wq_raw.T)
    
    # ==========================================
    # 验证
    # ==========================================
    diff_wrong = (llama_out - my_out_wrong).abs().max().item()
    diff_correct = (llama_out - my_out_correct).abs().max().item()
    
    print("-" * 40)
    print(f"原始 WQ 直接相乘误差: {diff_wrong:.6f} (结果完全不同)")
    print(f"转置 WQ 后相乘误差:   {diff_correct:.6f} (结果一致)")
    print("-" * 40)
    
    if diff_correct < 1e-5:
        print("✅ 解决方案验证成功：")
        print("   在你的代码中，请使用: result = matmul(x, wq.T)")
        print("   或者在加载权重时，直接把 wq 转置保存。")

if __name__ == "__main__":
    solve_llama_shape_mismatch()
