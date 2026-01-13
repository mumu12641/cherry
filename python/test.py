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

        def layers_body(index, curr_x, curr_k_cache, curr_v_cache):
            layer = ir.index_cast(index)

            rms_att_w_layer_1 = ir.tensor_slice(rms_att_w, [layer, zero], [1, 768])
            rms_att_w_layer = ir.reshape(rms_att_w_layer_1, [768])
            xb = ir.rmsnorm(curr_x, rms_att_w_layer, 1e-5)

            wq_layer_1 = ir.tensor_slice(wq, [layer, zero, zero], [1, 768, 768])
            wq_layer = ir.reshape(wq_layer_1, [768, 768])

            wk_layer_1 = ir.tensor_slice(wk, [layer, zero, zero], [1, 768, 768])
            wk_layer = ir.reshape(wk_layer_1, [768, 768])

            wv_layer_1 = ir.tensor_slice(wv, [layer, zero, zero], [1, 768, 768])
            wv_layer = ir.reshape(wv_layer_1, [768, 768])

            q_raw = ir.matmul(xb, wq_layer)
            k_raw = ir.matmul(xb, wk_layer)
            v = ir.matmul(xb, wv_layer)

            q_raw_heads = ir.reshape(q_raw, [1, 12, 64])
            q_rope = ir.rope(q_raw_heads, pos)
            q = ir.reshape(q_rope, [1, 768])

            k_raw_heads = ir.reshape(k_raw, [1, 12, 64])
            k_rope = ir.rope(k_raw_heads, pos)
            k = ir.reshape(k_rope, [1, 768])

            k_1 = ir.reshape(k, [1, 1, 768])
            next_k_cache = ir.tensor_set_slice(curr_k_cache, k_1, [layer, pos])
            v_1 = ir.reshape(v, [1, 1, 768])
            next_v_cache = ir.tensor_set_slice(curr_v_cache, v_1, [layer, pos])

            k_layer_full_1 = ir.tensor_slice(
                next_k_cache, [layer, zero, zero], [1, 1024, 768]
            )
            k_layer_full = ir.reshape(k_layer_full_1, [1024, 768])
            v_layer_full_1 = ir.tensor_slice(
                next_v_cache, [layer, zero, zero], [1, 1024, 768]
            )
            v_layer_full = ir.reshape(v_layer_full_1, [1024, 768])

            att_out_init = ir.create_tensor([0.0], [1, 12, 64])

            def heads_body(index, curr_att_out):
                head = ir.index_cast(index)
                offset = ir.scalar_mul(head, head_dim)

                q_head = ir.tensor_slice(q, [zero, offset], [1, 64])
                k_head = ir.tensor_slice(k_layer_full, [zero, offset], [1024, 64])
                k_head_T = ir.transpose(k_head, [1, 0])
                score = ir.masked_matmul(q_head, k_head_T, valid_len)

                score_scaled = ir.tensor_mul_scalar(score, scale_val)

                probs = ir.softmax(score_scaled, 1)

                v_head = ir.tensor_slice(v_layer_full, [zero, offset], [1024, 64])

                context_head = ir.matmul(probs, v_head)
                context_head_reshape = ir.reshape(context_head, [1, 1, 64])
                next_att_out = ir.tensor_set_slice(
                    curr_att_out, context_head_reshape, [zero, head]
                )
                return [next_att_out]

            [att_out_final_heads] = ir.create_for_loop(
                0, 12, 1, [att_out_init], heads_body
            )

            att_out_final = ir.reshape(att_out_final_heads, [1, 768])

            wo_layer_1 = ir.tensor_slice(wo, [layer, zero, zero], [1, 768, 768])
            wo_layer = ir.reshape(wo_layer_1, [768, 768])
            xb2 = ir.matmul(att_out_final, wo_layer)

            x_resid_1 = ir.add(curr_x, xb2)

            rms_ffn_w_layer_1 = ir.tensor_slice(rms_ffn_w, [layer, zero], [1, 768])
            rms_ffn_w_layer = ir.reshape(rms_ffn_w_layer_1, [768])
            xb_ffn = ir.rmsnorm(x_resid_1, rms_ffn_w_layer, 1e-5)

            w1_layer_1 = ir.tensor_slice(w1, [layer, zero, zero], [1, 768, 2048])
            w1_layer = ir.reshape(w1_layer_1, [768, 2048])
            w3_layer_1 = ir.tensor_slice(w3, [layer, zero, zero], [1, 768, 2048])
            w3_layer = ir.reshape(w3_layer_1, [768, 2048])

            hb = ir.matmul(xb_ffn, w1_layer)
            hb2 = ir.matmul(xb_ffn, w3_layer)
            hb_silu = ir.silu(hb)
            hb_mul = ir.mul(hb_silu, hb2)

            w2_layer_1 = ir.tensor_slice(w2, [layer, zero, zero], [1, 2048, 768])
            w2_layer = ir.reshape(w2_layer_1, [2048, 768])
            ffn_out = ir.matmul(hb_mul, w2_layer)
            x_next = ir.add(x_resid_1, ffn_out)

            return [x_next, next_k_cache, next_v_cache]

        [x_final, k_cache_final, v_cache_final] = ir.create_for_loop(
            0, 12, 1, [x, k_cache, v_cache], layers_body
        )
        x_norm = ir.rmsnorm(x_final, rms_final_w, 1e-5)
        logits = ir.matmul(x_norm, wcls)
        return [logits, k_cache_final, v_cache_final]


build_model()

ir.create_function("host", [], [], False)
vocab_size = ir.constant(32000)
ir.runtime_call(
    "build_tokenizer",
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
    [logits, next_k_cache, next_v_cache] = ir.call(
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
    arg_max = ir.argmax(logits, 1)
    zero = ir.constant(0)
    next_token = ir.tensor_get(arg_max, [zero])
    ir.runtime_call("decode", curr_token, next_token)

    return [next_token, next_pos, next_k_cache, next_v_cache]


initial_args = [start_token, start_pos, k_cache_init, v_cache_init]

results = ir.create_while_loop(initial_args, cond_fn, body_fn)

final_token, final_pos, final_k, final_v = results
ir.runtime_call("end", max_len)
ir.runtime_call("free_tokenizer")
ir.ret([])
print(ir.dump())
ir.dump_to_file("test.mlir")
