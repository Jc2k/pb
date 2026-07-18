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

struct pb_dsv4_embedding_batch_args {
    uint tokens;
    uint hidden;
    uint hc;
};

// Long prompts enter the fixed DeepSeek graph as one [token, HC, hidden]
// tensor. Token ids remain request-scoped shared storage; model embeddings are
// the same resident mmap-backed F16 tensor used by decode.
kernel void kernel_pb_dsv4_embedding_hc4_batch(
        constant pb_dsv4_embedding_batch_args &args,
        device const uint *tokens,
        device const half *embedding,
        device float *hc,
        uint gid [[thread_position_in_grid]]) {
    const ulong row = (ulong)args.hidden * args.hc;
    const ulong count = (ulong)args.tokens * row;
    if ((ulong)gid >= count) return;
    const uint token_row = uint((ulong)gid / row);
    const uint hidden_col = uint((ulong)gid % args.hidden);
    hc[gid] = float(embedding[(ulong)tokens[token_row] * args.hidden + hidden_col]);
}

struct pb_dsv4_copy_args {
    uint elements;
};

kernel void kernel_pb_dsv4_f32_to_f16(
        constant pb_dsv4_copy_args &args,
        device const float *src,
        device half *dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid < args.elements) dst[gid] = half(src[gid]);
}

struct pb_dsv4_swiglu_args {
    uint elements;
    float clamp;
};

kernel void kernel_pb_dsv4_swiglu_batch(
        constant pb_dsv4_swiglu_args &args,
        device const float *gate,
        device const float *up,
        device float *mid,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.elements) return;
    const float g = clamp(gate[gid], -args.clamp, args.clamp);
    const float u = clamp(up[gid], -args.clamp, args.clamp);
    mid[gid] = (g / (1.0f + exp(-g))) * u;
}

struct pb_dsv4_raw_store_batch_args {
    uint tokens;
    uint raw_cap;
    uint head_dim;
    uint pos0;
};

// Only the final logical SWA window is material after a zero-prefix batch.
// Restricting the copy to those rows also avoids write races when tokens exceed
// the ring capacity. DS4 rounds raw-cache values through F16 on store.
kernel void kernel_pb_dsv4_raw_store_batch(
        constant pb_dsv4_raw_store_batch_args &args,
        device const float *kv,
        device float *raw,
        uint gid [[thread_position_in_grid]]) {
    const uint kept = min(args.tokens, args.raw_cap);
    const ulong count = (ulong)kept * args.head_dim;
    if ((ulong)gid >= count) return;
    const uint local_row = gid / args.head_dim;
    const uint col = gid % args.head_dim;
    const uint token = args.tokens - kept + local_row;
    const uint raw_row = (args.pos0 + token) % args.raw_cap;
    raw[(ulong)raw_row * args.head_dim + col] = float(half(kv[(ulong)token * args.head_dim + col]));
}

struct pb_dsv4_raw_context_batch_args {
    uint tokens;
    uint prefix_raw;
    uint raw_cap;
    uint head_dim;
    uint pos0;
};

// Materialize the restored raw ring followed by the new suffix as one logical
// sequence. Batch attention can then apply its causal window to every suffix
// query without changing the scheduler-owned resident ring.
kernel void kernel_pb_dsv4_raw_context_batch(
        constant pb_dsv4_raw_context_batch_args &args,
        device const float *raw,
        device const float *kv,
        device float *context,
        uint gid [[thread_position_in_grid]]) {
    const uint rows = args.prefix_raw + args.tokens;
    const ulong count = (ulong)rows * args.head_dim;
    if ((ulong)gid >= count) return;
    const uint row = gid / args.head_dim;
    const uint col = gid % args.head_dim;
    if (row < args.prefix_raw) {
        const uint first_pos = args.pos0 - args.prefix_raw;
        const uint raw_row = (first_pos + row) % args.raw_cap;
        context[gid] = raw[(ulong)raw_row * args.head_dim + col];
    } else {
        const uint token = row - args.prefix_raw;
        context[gid] = kv[(ulong)token * args.head_dim + col];
    }
}

struct pb_dsv4_group_copy_args {
    uint tokens;
    uint groups;
    uint group;
    uint group_width;
    uint rank;
};

kernel void kernel_pb_dsv4_gather_attention_group(
        constant pb_dsv4_group_copy_args &args,
        device const float *heads,
        device float *group_rows,
        uint gid [[thread_position_in_grid]]) {
    const ulong count = (ulong)args.tokens * args.group_width;
    if ((ulong)gid >= count) return;
    const uint token = gid / args.group_width;
    const uint col = gid % args.group_width;
    const ulong source_stride = (ulong)args.groups * args.group_width;
    heads += (ulong)token * source_stride + (ulong)args.group * args.group_width;
    group_rows[gid] = heads[col];
}

kernel void kernel_pb_dsv4_scatter_attention_rank(
        constant pb_dsv4_group_copy_args &args,
        device const float *rank_rows,
        device float *all_ranks,
        uint gid [[thread_position_in_grid]]) {
    const ulong count = (ulong)args.tokens * args.rank;
    if ((ulong)gid >= count) return;
    const uint token = gid / args.rank;
    const uint col = gid % args.rank;
    all_ranks[(ulong)token * args.groups * args.rank + (ulong)args.group * args.rank + col] = rank_rows[gid];
}

struct pb_dsv4_compressor_prefill_args {
    uint tokens;
    uint width;
    uint head_dim;
    uint ratio;
    uint pos0;
};

// The compressor is recurrent across prompt rows. One threadgroup advances the
// exact decode frontier, synchronizing the full projected row before a ratio-4
// pool reads its second half. This removes thousands of one-token dispatches
// without introducing a cross-threadgroup dependency.
kernel void kernel_pb_dsv4_compressor_prefill(
        constant pb_dsv4_compressor_prefill_args &args,
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
    for (uint token = 0; token < args.tokens; ++token) {
        const uint position = args.pos0 + token;
        const uint pos_mod = position % args.ratio;
        const uint dst_row = args.ratio == 4u ? args.ratio + pos_mod : pos_mod;
        for (uint d = tid; d < args.width; d += ntg) {
            const ulong src = (ulong)token * args.width + d;
            const ulong dst = (ulong)dst_row * args.width + d;
            state_kv[dst] = kv[src];
            state_score[dst] = score[src] + float(ape[(ulong)pos_mod * args.width + d]);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if ((position + 1u) % args.ratio != 0u) continue;

        for (uint d = tid; d < args.head_dim; d += ntg) {
            float maximum = -INFINITY;
            if (args.ratio == 4u) {
                for (uint row = 0; row < 4u; ++row) {
                    maximum = max(maximum, state_score[(ulong)row * args.width + d]);
                    maximum = max(maximum, state_score[(ulong)(row + 4u) * args.width + args.head_dim + d]);
                }
            } else {
                for (uint row = 0; row < state_rows; ++row) {
                    maximum = max(maximum, state_score[(ulong)row * args.width + d]);
                }
            }
            float denominator = 0.0f;
            float value = 0.0f;
            if (args.ratio == 4u) {
                for (uint row = 0; row < 4u; ++row) {
                    const ulong first = (ulong)row * args.width + d;
                    const ulong second = (ulong)(row + 4u) * args.width + args.head_dim + d;
                    const float w0 = exp(state_score[first] - maximum);
                    const float w1 = exp(state_score[second] - maximum);
                    denominator += w0 + w1;
                    value += state_kv[first] * w0 + state_kv[second] * w1;
                }
            } else {
                for (uint row = 0; row < state_rows; ++row) {
                    const ulong index = (ulong)row * args.width + d;
                    const float weight = exp(state_score[index] - maximum);
                    denominator += weight;
                    value += state_kv[index] * weight;
                }
            }
            const uint emit = (position + 1u) / args.ratio - 1u;
            compressed[(ulong)emit * args.head_dim + d] = value / denominator;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        if (args.ratio == 4u) {
            const ulong half_rows = 4ul * args.width;
            for (ulong index = tid; index < half_rows; index += ntg) {
                state_kv[index] = state_kv[half_rows + index];
                state_score[index] = state_score[half_rows + index];
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }
}

struct pb_dsv4_attention_mask_args {
    uint tokens;
    uint raw_rows;
    uint compressed;
    uint window;
    uint ratio;
    uint pos0;
};

kernel void kernel_pb_dsv4_prefill_attention_mask(
        constant pb_dsv4_attention_mask_args &args,
        device half *mask,
        uint gid [[thread_position_in_grid]]) {
    const uint keys = args.raw_rows + args.compressed;
    const ulong count = (ulong)args.tokens * keys;
    if ((ulong)gid >= count) return;
    const uint query = gid / keys;
    const uint key = gid % keys;
    bool visible;
    if (args.pos0 == 0u && args.raw_rows == args.tokens) {
        if (key < args.tokens) {
            visible = key <= query &&
                (args.window == 0u || query - key < args.window);
        } else {
            visible = args.ratio != 0u &&
                key - args.tokens < (query + 1u) / args.ratio;
        }
        mask[gid] = visible ? half(0.0f) : -INFINITY;
        return;
    }
    const uint qpos = args.pos0 + query;
    if (key < args.raw_rows) {
        const uint first_raw_pos = args.pos0 + args.tokens - args.raw_rows;
        const uint key_pos = first_raw_pos + key;
        visible = key_pos <= qpos &&
            (args.window == 0u || qpos - key_pos < args.window);
    } else {
        visible = args.ratio != 0u &&
            key - args.raw_rows < (qpos + 1u) / args.ratio;
    }
    mask[gid] = visible ? half(0.0f) : -INFINITY;
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
