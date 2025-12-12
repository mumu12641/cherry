import struct
import numpy as np
import os

def load_tensor(filepath):
    """
    读取由 C 语言 save_tensor 函数生成的二进制文件。
    格式: [ndim (int)] + [shape (int array)] + [data (float array)]
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found: {filepath}")
        return None

    with open(filepath, 'rb') as f:
        # 1. 读取 ndim (int, 4 bytes)
        # 'i' 代表 C 语言的 int
        ndim_bytes = f.read(4)
        if not ndim_bytes:
            return None
        ndim = struct.unpack('i', ndim_bytes)[0]

        # 2. 读取 shape (int array, ndim * 4 bytes)
        shape_bytes = f.read(4 * ndim)
        # '{ndim}i' 表示读取 ndim 个整数
        shape = struct.unpack(f'{ndim}i', shape_bytes)

        # 3. 读取 data (float array)
        # 使用 numpy 直接从文件流剩余部分读取 float32 (对应 C 的 float)
        # 这样比 struct.unpack 快得多，特别是对于大数组
        data = np.frombuffer(f.read(), dtype=np.float32)

    # 4. 根据 shape 重塑数组
    try:
        tensor = data.reshape(shape)
        print(f"✅ Loaded: {filepath}")
        print(f"   Shape: {shape}")
        print(f"   Dtype: {tensor.dtype}")
        return tensor
    except ValueError as e:
        print(f"❌ Error reshaping data: {e}")
        print(f"   Expected elements: {np.prod(shape)}")
        print(f"   Actual elements: {data.size}")
        return None

# =================使用示例=================
if __name__ == "__main__":
    # 假设你的文件名为 output.bin
    file_path = "/home/nx/ycy/pb/cherry/utils/stories110M/output_wcls.bin" 
    
    # 如果你没有文件，先生成一个模拟文件测试
    # (仅用于演示，实际使用时请直接读取你的文件)
    # import struct
    # with open(file_path, 'wb') as f:
    #     ndim = 2
    #     shape = [2, 5]
    #     data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0], dtype=np.float32)
    #     f.write(struct.pack('i', ndim))
    #     f.write(struct.pack(f'{ndim}i', *shape))
    #     f.write(data.tobytes())

    # 读取文件
    tensor = load_tensor(file_path)

    # if tensor is not None:
    #     print("\n前 10 个数据预览:")
    #     # flatten() 展平以便打印，就像你在问题里贴的那样
    #     print(tensor.flatten()[:10]) 
    print(tensor.shape)
