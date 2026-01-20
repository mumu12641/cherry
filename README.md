<p align="center"> 
	<img src="asset/icon.png" width=160 height=160  >
</p>
<div align="center">
    <img alt="License" src="https://img.shields.io/github/license/mumu12641/cherry?color=red&style=flat-square">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/mumu12641/cherry?color=red&style=flat-square">
<h1 align="center">
    Cherry
</h1>
<p align="center">
    A simple AI compiler based on MLIR, developed purely for educational purposes.
</p>
</div>

## 🔍 Demo

<a href="https://asciinema.org/a/769592?autoplay=1"><img src="https://asciinema.org/a/769592.svg" alt="cherry demo" width="100%" /></a>

## ✨ Features

*   **🛠️ Custom IR**: Implemented custom Dialects and Types.
*   **🔄 Compilation Pipeline**: Implemented Conversion Passes from custom Dialect to `Linalg` and `Tensor` Dialects, with a complete pipeline lowering to LLVM IR.
*   **🐍 Python Interface**: Exposes compiled operators to Python via `pybind11` for easy invocation.
*   **⚡ Lightweight Runtime**: Developed a lightweight C++ Runtime library handling model weight loading and Tokenizer processing, enabling end-to-end model inference.

## 🚀 Build & Run

### 1. 🏗️ Build Project

Ensure `CMake`, and `Ninja` are installed in your environment.

```bash
git clone https://github.com/mumu12641/cherry
cd cherry
git clone --depth 1 -b llvmorg-19.1.7 https://github.com/llvm/llvm-project.git third_party/llvm-project

mkdir build && cd build
# Generate build files
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .. -G Ninja
# Build core library and Python extensions
ninja cherry
ninja cherry_py
# (Optional) Build debug tools
ninja cherry_opt

cd ..
```

### 2. 📦 Prepare Model Weights

Download the `stories110M` model and convert it to the format supported by Cherry.

```bash
cd utils

# Download original weights
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin

# Export weights (parameter slicing)
python export.py stories110M.bin stories110M

# Download Tokenizer
wget https://github.com/karpathy/llama2.c/raw/refs/heads/master/tokenizer.bin -O stories110M/tokenizer.bin

cd ..
```

### 3. 🏃 Run Inference

Use the Python script to load the compiled operators and weights for inference.

```bash
cd python
chmod +x ./test.sh
# Here uses $BUILD_DIR/third_party/llvm-project/llvm/bin/clang++, you can perhaps replace it with your own clang++.
./test.sh
```
