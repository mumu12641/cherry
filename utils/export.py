# /**
#  * @file export.py
#  * @brief Utility to extract individual tensor weights from llama2.c binary model files.
#  *
#  * Target Data Source:
#  *   $ wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
#  *    python export.py stories110M.bin stories110M
#  */

import sys
import struct
import os
import numpy as np

def create_directory_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, mode=0o755)
            print(f"📂 Creating output directory: {dir_path}")
        except OSError as _:
            print(f"❌ Failed to create directory: {dir_path}")
            sys.exit(1)
    else:
        print(f"📂 Using existing directory: {dir_path}")

def save_tensor(output_dir, name, data):
    """
    保存 Tensor 为二进制文件，格式与 C 代码一致：
    [ndim (int), shape (int...), data (float...)]
    """
    filepath = os.path.join(output_dir, f"{name}.bin")
    
    shape = data.shape
    ndim = len(shape)
    
    with open(filepath, 'wb') as f:
        # 1. 写入 ndim
        f.write(struct.pack('i', ndim))
        # 2. 写入 shape
        f.write(struct.pack(f'{ndim}i', *shape))
        # 3. 写入数据 (确保是 float32)
        data.astype(np.float32).tofile(f)

    # 打印日志
    shape_str = ", ".join(map(str, shape))
    print(f"💾 Saved: {name:<25} Shape: [{shape_str}]")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <model_file> [output_dir]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else "weights"

    print(f"📖 Reading Model: {model_path}")

    if not os.path.isfile(model_path):
        print(f"❌ Cannot open model file: {model_path}")
        sys.exit(1)

    # 读取 Config (7个 int)
    # C struct: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
    with open(model_path, 'rb') as f:
        config_bytes = f.read(28) # 7 * 4 bytes
        if len(config_bytes) != 28:
            print("❌ Failed to read Config.")
            sys.exit(1)
        
        config = struct.unpack('iiiiiii', config_bytes)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = config

        # 处理 Shared Weights 标志
        shared_weights = vocab_size > 0
        vocab_size = abs(vocab_size)

        print("⚙️  Model Configuration:")
        print(f"   • dim:        {dim}")
        print(f"   • hidden_dim: {hidden_dim}")
        print(f"   • n_layers:   {n_layers}")
        print(f"   • n_heads:    {n_heads}")
        print(f"   • n_kv_heads: {n_kv_heads}")
        print(f"   • vocab_size: {vocab_size} {'(Shared)' if shared_weights else '(Unshared)'}")
        print(f"   • seq_len:    {seq_len}")
        print("----------------------------------------")

        # 读取剩余的所有权重数据
        print("🚀 Loading weights into memory...")
        # 从当前位置读取剩余所有数据作为 float32 数组
        weights_data = np.fromfile(f, dtype=np.float32)

    if weights_data.size == 0:
        print("❌ File contains no weight data.")
        sys.exit(1)

    create_directory_if_not_exists(output_dir)
    print("🚀 Starting extraction...")

    # 指针偏移量
    offset = 0
    head_size = dim // n_heads

    # 定义需要转置的权重名称集合
    transpose_targets = {
        "layers_wq", "layers_wk", "layers_wv", "layers_wo",
        "layers_w1", "layers_w2", "layers_w3", "output_wcls"
    }

    # 辅助函数：提取并处理权重
    token_embeddings_ref = None # 用于 shared weights

    def extract_and_save(name, shape):
        nonlocal offset, token_embeddings_ref
        
        # 计算元素总数
        size = np.prod(shape)
        
        # 切片
        data = weights_data[offset : offset + size]
        offset += size
        
        # Reshape
        data = data.reshape(shape)

        # 保存 token_embeddings 的引用，以防 output_wcls 需要共享
        if name == "token_embeddings":
            token_embeddings_ref = data

        # 检查是否需要转置 (最后两个维度)
        if name in transpose_targets:
            # 交换最后两个维度 (-1 和 -2)
            data = np.swapaxes(data, -1, -2)
            # 注意：转置后为了保证内存连续性以便正确写入文件，通常需要 contiguous()
            # 但 numpy 的 tofile 会自动处理，或者我们可以显式调用
            data = np.ascontiguousarray(data)

        save_tensor(output_dir, name, data)

    # --- 按顺序提取权重 ---

    # 1. token_embeddings
    extract_and_save("token_embeddings", (vocab_size, dim))

    # 2. layers_rms_att_weight
    extract_and_save("layers_rms_att_weight", (n_layers, dim))

    # 3. layers_wq
    extract_and_save("layers_wq", (n_layers, dim, n_heads * head_size))

    # 4. layers_wk
    extract_and_save("layers_wk", (n_layers, dim, n_kv_heads * head_size))

    # 5. layers_wv
    extract_and_save("layers_wv", (n_layers, dim, n_kv_heads * head_size))

    # 6. layers_wo
    extract_and_save("layers_wo", (n_layers, n_heads * head_size, dim))

    # 7. layers_rms_ffn_weight
    extract_and_save("layers_rms_ffn_weight", (n_layers, dim))

    # 8. layers_w1
    extract_and_save("layers_w1", (n_layers, hidden_dim, dim))

    # 9. layers_w2
    extract_and_save("layers_w2", (n_layers, dim, hidden_dim))

    # 10. layers_w3
    extract_and_save("layers_w3", (n_layers, hidden_dim, dim))

    # 11. final_rms_norm
    extract_and_save("final_rms_norm", (dim,))

    # 12. output_wcls
    if shared_weights:
        print("ℹ️  Shared Weights detected: Copying token_embeddings to output_wcls...")
        # 复制 token_embeddings
        data = token_embeddings_ref.copy()
        
        # 即使是共享的，按照你的要求，output_wcls 也需要转置
        # token_embeddings 是 [vocab, dim]，转置后变成 [dim, vocab]
        if "output_wcls" in transpose_targets:
            data = np.swapaxes(data, -1, -2)
            data = np.ascontiguousarray(data)
            
        save_tensor(output_dir, "output_wcls", data)
    else:
        extract_and_save("output_wcls", (vocab_size, dim))

    print(f"\n✨ Done! All weights have been extracted to '{output_dir}'.")

if __name__ == "__main__":
    main()