import cherry.core as core
from tools import from_json_config

ir = core.IrGenerator()


def build_model():
    @from_json_config(ir, "llama110M.json")
    def llama_forward(
        token_id,
        pos,
        k_cache,
        v_cache,
        embed_table,
        rms_att_w,
        wq,
        wk,
        wv,
        wo,
        rms_ffn_w,
        w1,
        w2,
        w3,
        rms_final_w,
        wcls,
    ):
        zero = ir.constant(0)
        one = ir.constant(1)
        dim = ir.constant(768)
        ffn_dim = ir.constant(2048)
        head_num = ir.constant(12)
        head_dim = ir.constant(64)
        scale_val = ir.constant(0.125)
        max_len = ir.constant(1024)
        vocab_size = ir.constant(32000)
        valid_len = ir.scalar_add(pos, one)

        x = ir.tensor_slice(embed_table, [token_id, zero], [1, 768])

        def body(iv, curr_x, curr_k_cache, curr_v_cache):
            return curr_x, curr_k_cache, curr_v_cache

        x_final, k_cache_final, v_cache_final = ir.create_for_loop(
            0, 12, 1, [x, k_cache, v_cache], body
        )

        logits = ir.create_tensor([0.0], [1, 32000])

        return logits, k_cache_final, v_cache_final


build_model()

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
    res = ir.call(
        "llama_forward",
        [
            curr_token,
            curr_pos,
            curr_k,
            curr_v,
            embedding,
            rms_att,
            wq,
            wk,
            wv,
            wo,
            rms_ffn,
            w1,
            w2,
            w3,
            rms_final,
            wcls,
        ],
    )

    return [next_token, next_pos, curr_k, curr_v]


initial_args = [start_token, start_pos, k_cache_init, v_cache_init]

results = ir.create_while_loop(initial_args, cond_fn, body_fn)

final_token, final_pos, final_k, final_v = results
ir.ret([])
print(ir.dump())
