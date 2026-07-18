// The Metal kernels are vendored from antirez/ds4 at
// 80ebbc396aee40eedc1d829222f3362d10fa4c6c under the MIT license in
// metal/deepseek/LICENSE. pb compiles them into the existing FlashMoe Metal
// execution facade only when the pinned DeepSeek V4 Flash graph is selected.

pub(crate) const DEEPSEEK_V4_METAL_SHADERS: &str = concat!(
    r#"
#include <metal_stdlib>
#ifdef DS4_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;
#ifdef DS4_METAL_HAS_TENSOR
using namespace mpp::tensor_ops;
#endif

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define M_PI_F 3.14159265358979323846f

enum ds4_sort_order {
    DS4_SORT_ORDER_ASC,
    DS4_SORT_ORDER_DESC,
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

"#,
    include_str!("metal/deepseek/flash_attn.metal"),
    include_str!("metal/deepseek/dense.metal"),
    include_str!("metal/deepseek/moe.metal"),
    include_str!("metal/deepseek/dsv4_hc.metal"),
    include_str!("metal/deepseek/unary.metal"),
    include_str!("metal/deepseek/dsv4_kv.metal"),
    include_str!("metal/deepseek/dsv4_rope.metal"),
    include_str!("metal/deepseek/dsv4_misc.metal"),
    include_str!("metal/deepseek/argsort.metal"),
    include_str!("metal/deepseek/cpy.metal"),
    include_str!("metal/deepseek/concat.metal"),
    include_str!("metal/deepseek/get_rows.metal"),
    include_str!("metal/deepseek/sum_rows.metal"),
    include_str!("metal/deepseek/softmax.metal"),
    include_str!("metal/deepseek/repeat.metal"),
    include_str!("metal/deepseek/glu.metal"),
    include_str!("metal/deepseek/norm.metal"),
    include_str!("metal/deepseek/bin.metal"),
    include_str!("metal/deepseek/set_rows.metal"),
    include_str!("metal/deepseek/pb_decode.metal"),
);

pub(crate) const DEEPSEEK_V4_REQUIRED_METAL_KERNELS: &[&str] = &[
    "kernel_mul_mv_q8_0_f32",
    "kernel_mul_mv_f16_f32",
    "kernel_rms_norm_f32_4",
    "kernel_rms_norm_mul_f32_4",
    "kernel_dsv4_shared_gate_up_swiglu_q8_0",
    "kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32",
    "kernel_mul_mv_slots6_q2_K_sum6_f32",
    "kernel_dsv4_hc_split_sinkhorn",
    "kernel_dsv4_hc_split_weighted_sum_norm4",
    "kernel_dsv4_hc_expand4",
    "kernel_dsv4_shared_down_hc_expand4_q8_0",
    "kernel_dsv4_q8_hc_expand4_q8_0",
    "kernel_dsv4_hc_weighted_sum",
    "kernel_dsv4_qkv_rms_norm_f32_4",
    "kernel_dsv4_fp8_kv_quantize_f32",
    "kernel_dsv4_indexer_hadamard_fp4_f32",
    "kernel_dsv4_kv_fp8_store_f32",
    "kernel_dsv4_ratio4_shift_f32",
    "kernel_dsv4_compressor_store_one",
    "kernel_dsv4_rope_tail_f32",
    "kernel_dsv4_directional_steering_project_f32",
    "kernel_dsv4_indexer_score_one_direct",
    "kernel_dsv4_router_weights_one",
    "kernel_dsv4_router_finalize_one",
    "kernel_dsv4_indexed_mixed_attention_heads8",
    "kernel_dsv4_indexer_scores_tiled_f32",
    "kernel_dsv4_indexer_weighted_sum",
    "kernel_dsv4_softmax_pool",
    "kernel_swiglu_f32",
    "kernel_dsv4_softplus_sqrt_f32_4",
    "kernel_soft_max_f32",
    "kernel_soft_max_f32_4",
    "kernel_argsort_f32_i32_desc",
    "kernel_argsort_merge_f32_i32_desc",
    "kernel_get_rows_f32",
    "kernel_get_rows_i32",
    "kernel_set_rows_f32_i32",
    "kernel_sum_rows_f32_f32",
    "kernel_pb_dsv4_embedding_hc4",
    "kernel_pb_dsv4_rms_norm_f32",
    "kernel_pb_dsv4_compressor_step",
    "kernel_pb_dsv4_decode_attention_h512",
    "kernel_pb_dsv4_output_collapse_norm4",
];
