// pb-owned glue kernels for the DeepSeek V4 Flash graph.  The numerical
// primitives and quantized projections remain the pinned DS4 kernels; these
// kernels fuse graph operations whose host-side DS4 implementation otherwise
// depends on its standalone runtime state.

struct pb_dsv4_embedding_args {
    uint token;
    uint hidden;
    uint hc;
};

kernel void kernel_pb_dsv4_embedding_hc4(
        constant pb_dsv4_embedding_args &args,
        device const half *embedding,
        device float *hc,
        uint gid [[thread_position_in_grid]]) {
    const uint count = args.hidden * args.hc;
    if (gid >= count) return;
    hc[gid] = float(embedding[(ulong)args.token * args.hidden + gid % args.hidden]);
}

struct pb_dsv4_rms_norm_args {
    uint width;
    uint rows;
    uint weighted;
    float eps;
};

kernel void kernel_pb_dsv4_rms_norm_f32(
        constant pb_dsv4_rms_norm_args &args,
        device const float *input,
        device const float *weight,
        device float *output,
        threadgroup float *sums [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint tiisg [[thread_index_in_simdgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;
    device const float *src = input + (ulong)row * args.width;
    device float *dst = output + (ulong)row * args.width;
    float local = 0.0f;
    for (uint d = tid; d < args.width; d += ntg) {
        const float v = src[d];
        local += v * v;
    }
    local = simd_sum(local);
    if (tiisg == 0u) sums[sgitg] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint groups = (ntg + 31u) / 32u;
        for (uint i = 0; i < groups; ++i) total += sums[i];
        sums[0] = rsqrt(total / float(args.width) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sums[0];
    for (uint d = tid; d < args.width; d += ntg) {
        dst[d] = src[d] * inv_rms * (args.weighted != 0u ? weight[d] : 1.0f);
    }
}

struct pb_dsv4_compressor_args {
    uint width;
    uint head_dim;
    uint ratio;
    uint position;
    uint emit_row;
};

// Updates the fixed compressor frontier and, at emission boundaries, performs
// the exact dimension-wise softmax pool used by DS4. RMSNorm, RoPE, and FP8
// simulation remain separate pinned kernels so their reduction/codegen shape
// stays identical to the reference implementation.
kernel void kernel_pb_dsv4_compressor_step(
        constant pb_dsv4_compressor_args &args,
        device const float *kv,
        device const float *score,
        device const half *ape,
        device float *state_kv,
        device float *state_score,
        device float *compressed,
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    const uint coff = args.ratio == 4u ? 2u : 1u;
    const uint state_rows = coff * args.ratio;
    const uint pos_mod = args.position % args.ratio;
    const uint dst_row = args.ratio == 4u ? args.ratio + pos_mod : pos_mod;
    for (uint d = tid; d < args.width; d += ntg) {
        const uint dst = dst_row * args.width + d;
        state_kv[dst] = kv[d];
        state_score[dst] = score[d] + float(ape[pos_mod * args.width + d]);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if ((args.position + 1u) % args.ratio != 0u) return;

    device float *out = compressed + (ulong)args.emit_row * args.head_dim;
    for (uint d = tid; d < args.head_dim; d += ntg) {
        float max_score = -INFINITY;
        if (args.ratio == 4u) {
            for (uint row = 0; row < 4u; ++row) {
                max_score = max(max_score, state_score[row * args.width + d]);
                max_score = max(max_score, state_score[(row + 4u) * args.width + args.head_dim + d]);
            }
            float sum = 0.0f;
            float value = 0.0f;
            for (uint row = 0; row < 4u; ++row) {
                const uint first = row * args.width + d;
                const uint second = (row + 4u) * args.width + args.head_dim + d;
                const float w0 = exp(state_score[first] - max_score);
                const float w1 = exp(state_score[second] - max_score);
                sum += w0 + w1;
                value += state_kv[first] * w0 + state_kv[second] * w1;
            }
            out[d] = value / sum;
        } else {
            for (uint row = 0; row < state_rows; ++row) {
                max_score = max(max_score, state_score[row * args.width + d]);
            }
            float sum = 0.0f;
            float value = 0.0f;
            for (uint row = 0; row < state_rows; ++row) {
                const uint index = row * args.width + d;
                const float weight = exp(state_score[index] - max_score);
                sum += weight;
                value += state_kv[index] * weight;
            }
            out[d] = value / sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (args.ratio == 4u) {
        const uint half_rows = 4u * args.width;
        for (uint d = tid; d < half_rows; d += ntg) {
            state_kv[d] = state_kv[half_rows + d];
            state_score[d] = state_score[half_rows + d];
        }
    }
}

struct pb_dsv4_attention_args {
    uint n_head;
    uint head_dim;
    uint n_raw;
    uint raw_cap;
    uint raw_start;
    uint n_comp;
    uint top_k;
    uint use_top_k;
    uint position;
    uint window;
    uint ratio;
    float scale;
};

// Single-token fused decode attention. Each thread owns one float4, matching
// the DS4 indexed-attention reduction shape, and performs online softmax over
// the shared raw ring plus the selected compressed memory rows.
kernel void kernel_pb_dsv4_decode_attention_h512(
        constant pb_dsv4_attention_args &args,
        device const float *q,
        device const float *raw_kv,
        device const float *compressed_kv,
        device const int *selected,
        device const float *sinks,
        device float *output,
        threadgroup float *shared [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint tiisg [[thread_index_in_simdgroup]]) {
    if (head >= args.n_head || args.head_dim != 512u || tid >= 128u) return;
    device const float4 *qh = (device const float4 *)(q + (ulong)head * 512u);
    const half4 qv = (half4)qh[tid];
    float4 acc = 0.0f;
    if (tid == 0u) {
        shared[8] = -FLT_MAX/2.0f;
        shared[9] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total = args.n_raw + (args.use_top_k != 0u ? args.top_k : args.n_comp);
    for (uint key_index = 0; key_index < total; ++key_index) {
        device const float *key = raw_kv;
        bool include = true;
        if (key_index < args.n_raw) {
            const uint ring_row = (args.raw_start + key_index) % args.raw_cap;
            key = raw_kv + (ulong)ring_row * 512u;
        } else {
            const uint comp_slot = key_index - args.n_raw;
            int comp_row = args.use_top_k != 0u ? selected[comp_slot] : int(comp_slot);
            include = comp_row >= 0 && uint(comp_row) < args.n_comp;
            if (include) {
                key = compressed_kv + (ulong)uint(comp_row) * 512u;
            }
        }
        float partial = 0.0f;
        half4 kv = 0.0h;
        if (include) {
            // DS4's attention cache is float-addressable but its attention
            // dot/value path rounds Q, K, and V through half first.
            kv = (half4)((device const float4 *)key)[tid];
            partial = dot((float4)qv, (float4)kv);
        }
        partial = simd_sum(partial);
        if (tiisg == 0u) shared[sgitg] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float dot = 0.0f;
            for (uint group = 0; group < 4u; ++group) dot += shared[group];
            const float score = include ? dot * args.scale : -INFINITY;
            const float old_max = shared[8];
            const float next_max = max(old_max, score);
            shared[10] = exp(old_max - next_max);
            shared[11] = include ? exp(score - next_max) : 0.0f;
            shared[8] = next_max;
            shared[9] = shared[9] * shared[10] + shared[11];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = acc * shared[10] + (include ? (float4)kv * shared[11] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        const float old_max = shared[8];
        const float next_max = max(old_max, sinks[head]);
        shared[10] = exp(old_max - next_max);
        shared[8] = next_max;
        shared[9] = shared[9] * shared[10] + exp(sinks[head] - next_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc *= shared[10];
    const float inverse_sum = 1.0f / shared[9];
    device float4 *dst = (device float4 *)(output + (ulong)head * 512u);
    dst[tid] = acc * inverse_sum;
}

struct pb_dsv4_output_collapse_args {
    uint hidden;
    float eps;
    float hc_eps;
};

// Final HC collapse plus weighted output RMSNorm. The four output HC weights
// are sigmoid(pre * scale + base) + eps, exactly matching DS4's output-head
// sequence. One threadgroup keeps the collapsed row live for the norm
// reduction and avoids a separate materialized 4096-wide intermediate.
kernel void kernel_pb_dsv4_output_collapse_norm4(
        constant pb_dsv4_output_collapse_args &args,
        device const float *hc,
        device const float *pre,
        device const float *scale,
        device const float *base,
        device const float *norm_weight,
        device float *output,
        threadgroup float *sums [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint tiisg [[thread_index_in_simdgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    const float w0 = 1.0f / (1.0f + exp(-(pre[0] * scale[0] + base[0]))) + args.hc_eps;
    const float w1 = 1.0f / (1.0f + exp(-(pre[1] * scale[0] + base[1]))) + args.hc_eps;
    const float w2 = 1.0f / (1.0f + exp(-(pre[2] * scale[0] + base[2]))) + args.hc_eps;
    const float w3 = 1.0f / (1.0f + exp(-(pre[3] * scale[0] + base[3]))) + args.hc_eps;
    float local = 0.0f;
    for (uint d = tid; d < args.hidden; d += ntg) {
        const float v = hc[d] * w0 + hc[args.hidden + d] * w1 +
                        hc[2u * args.hidden + d] * w2 + hc[3u * args.hidden + d] * w3;
        output[d] = v;
        local += v * v;
    }
    local = simd_sum(local);
    if (tiisg == 0u) sums[sgitg] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint groups = (ntg + 31u) / 32u;
        for (uint i = 0; i < groups; ++i) total += sums[i];
        sums[0] = rsqrt(total / float(args.hidden) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = sums[0];
    for (uint d = tid; d < args.hidden; d += ntg) {
        output[d] = output[d] * inv_rms * norm_weight[d];
    }
}
