# Runtime

在 runtime 文件夹中，使用 runtime_call 来调用 runtime 中的函数，这里比较特殊的是，如果是普通的参数那么就直接传递，如果是字符串就要通过 attr 进行传递。

然后对于 load weight ，这个必须在做前端的时候把这些函数再生成好。

对于以上两个功能需要非常注意的是：里面的 string 都是使用的 i8 的 tensor 保存，因此在 llvm ir 生成的函数签名中是这样的：
declare void @build_tokenizer(i64, ptr, ptr, i64, i64, i64)
这里的第一个 i64 是 vocab size，后面的全部都是 tokenizer.bin 这个 string 对应的tensor，分别为 alloc ptr, align ptr, offset, size, stride，其中只需要 align ptr 即可。