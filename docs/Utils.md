# Utils

这里主要是根据 llama2.c 里面的模型文件进行 export，把每个权重都 export 为一个单独的文件。
然后值得注意的是，对于以下这些权重
"layers_wq", "layers_wk", "layers_wv", "layers_wo",
"layers_w1", "layers_w2", "layers_w3", "output_wcls"
在 llama2.c 中，以 W1 为例子他是用 W1 (2048,768) @ x (768,) -> xout (2048,) 进行计算的，而我是用 x(1,768) @ W1 (768,2048) -> xout (1, 2048) 进行计算的，因此需要把以上这些权重的最后两个维度进行转置再存入文件。layers_w1(12, 2048, 768) => layers_w1(12, 768, 2048)。