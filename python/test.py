import cherry.core as core

ir = core.IrGenerator()


def build_llama_forward():
    N_LAYERS = 12
    MAX_SEQ = 1024
    DIM = 768
    VOCAB = 32000
    HIDDEN = 2048

    i64 = ir.create_type("i64")
    f32 = ir.create_type("f32")

    # KV Cache: [12, 1024, 768]
    t_cache = ir.create_tensor_type([N_LAYERS, MAX_SEQ, DIM], f32)

    # Embedding Table: [32000, 768]
    t_embed = ir.create_tensor_type([VOCAB, DIM], f32)

    # Layer Norm Weights: [12, 768] (用于 rms_att_weight, rms_ffn_weight)
    t_layer_norm = ir.create_tensor_type([N_LAYERS, DIM], f32)

    # Attention Weights: [12, 768, 768] (用于 wq, wk, wv, wo)
    t_attn_weight = ir.create_tensor_type([N_LAYERS, DIM, DIM], f32)

    # FFN Weights
    # Gate/Up: [12, 768, 2048]
    t_ffn_up = ir.create_tensor_type([N_LAYERS, DIM, HIDDEN], f32)
    # Down: [12, 2048, 768]
    t_ffn_down = ir.create_tensor_type([N_LAYERS, HIDDEN, DIM], f32)

    # Final Norm: [768]
    t_final_norm = ir.create_tensor_type([DIM], f32)

    # Classifier (Output Head): [768, 32000]
    t_classifier = ir.create_tensor_type([DIM, VOCAB], f32)

    # Output Logits: [1, 32000]
    t_logits = ir.create_tensor_type([1, VOCAB], f32)

    args = [
        i64,
        i64,
        t_cache,
        t_cache,
        t_embed,
        t_layer_norm,
        t_attn_weight,
        t_attn_weight,
        t_attn_weight,
        t_attn_weight,
        t_layer_norm,
        t_ffn_up,
        t_ffn_down,
        t_ffn_up,
        t_final_norm,
        t_classifier,
    ]

    rets = [t_logits, t_cache, t_cache]

    ir.create_function("llama_forward", args, rets, True)
    ir.ret()


build_llama_forward()
ir.create_function("host", [], [], False)
vocab_size = ir.constant(32000)
ir.runtime_call(
    "builder_tokenizer",
    vocab_size,
    str_args=["/home/nx/ycy/pb/cherry/tests/llama/tokenizer.bin"],
)
base_dir = "/home/nx/ycy/pb/cherry/utils/stories110M/"

embedding = ir.load_weight(base_dir + "token_embeddings.bin", [32000, 768], "f32")
rms_att = ir.load_weight(base_dir + "layers_rms_att_weight.bin", [12, 768], "f32")
wq = ir.load_weight(base_dir + "layers_wq.bin", [12, 768, 768], "f32")
wk = ir.load_weight(base_dir + "layers_wk.bin", [12, 768, 768], "f32")
wv = ir.load_weight(base_dir + "layers_wv.bin", [12, 768, 768], "f32")
wo = ir.load_weight(base_dir + "layers_wo.bin", [12, 768, 768], "f32")
rms_ffn = ir.load_weight(base_dir + "layers_rms_ffn_weight.bin", [12, 768], "f32")
w1 = ir.load_weight(base_dir + "layers_w1.bin", [12, 768, 2048], "f32")
w2 = ir.load_weight(base_dir + "layers_w2.bin", [12, 2048, 768], "f32")
w3 = ir.load_weight(base_dir + "layers_w3.bin", [12, 768, 2048], "f32")
rms_final = ir.load_weight(base_dir + "final_rms_norm.bin", [768], "f32")
wcls = ir.load_weight(base_dir + "output_wcls.bin", [768, 32000], "f32")

k_cache_init = ir.create_tensor([0.0], [12, 1024, 768])
v_cache_init = ir.create_tensor([0.0], [12, 1024, 768])

start_token = ir.constant(1)
start_pos = ir.constant(0)
max_len = ir.constant(30)


def cond_fn(curr_token, curr_pos, curr_k, curr_v):
    is_lt = ir.cmpi(curr_pos, max_len, "slt")
    return (is_lt, [curr_token, curr_pos, curr_k, curr_v])


def body_fn(curr_token, curr_pos, curr_k, curr_v):
    one = ir.constant(1)
    next_pos = ir.scalar_add(curr_pos, one)
    next_token = curr_token

    return [next_token, next_pos, curr_k, curr_v]


initial_args = [start_token, start_pos, k_cache_init, v_cache_init]

results = ir.create_loop(initial_args, cond_fn, body_fn)

final_token, final_pos, final_k, final_v = results
ir.ret()
print(ir.dump())
