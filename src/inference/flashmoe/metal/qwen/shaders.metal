#include <metal_stdlib>
using namespace metal;

kernel void q4_fma_matvec(
    device const uchar* packed [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& groups_per_row [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = scales[scale_row + group];
            float bias = biases[scale_row + group];

            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];

            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale0 = scales[scale_row + group0];
            float bias0 = biases[scale_row + group0];
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale1 = scales[scale_row + group1];
                float bias1 = biases[scale_row + group1];
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

inline float bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16u);
}

inline float q4_fma_row_bf16_ptrs(
    device const uchar* packed,
    device const ushort* scales,
    device const ushort* biases,
    device const float* input,
    threadgroup float* input_cache,
    uint row,
    uint cols,
    uint groups_per_row,
    uint group_size,
    bool word_aligned,
    bool use_input_cache,
    uint simd_lane) {
    uint packed_stride = (cols + 1) / 2;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = word_aligned && (cols % 8 == 0) && (group_size % 8 == 0);
    float acc = 0.0f;
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = bf16_to_float(scales[scale_row + group]);
            float bias = bf16_to_float(biases[scale_row + group]);
            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];
            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale0 = bf16_to_float(scales[scale_row + group0]);
            float bias0 = bf16_to_float(biases[scale_row + group0]);
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);
            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale1 = bf16_to_float(scales[scale_row + group1]);
                float bias1 = bf16_to_float(biases[scale_row + group1]);
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    return simd_sum(acc);
}

kernel void q4_fma_matvec_bf16_scale_bias(
    device const uchar* packed [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const ushort* scales [[buffer(2)]],
    device const ushort* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& groups_per_row [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float sum = q4_fma_row_bf16_ptrs(
        packed, scales, biases, input, input_cache,
        row, cols, groups_per_row, group_size, true, use_input_cache, simd_lane);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

inline float mxfp4_e2m1_to_float(uchar nibble) {
    float magnitude;
    switch (nibble & 0x07) {
        case 0: magnitude = 0.0f; break;
        case 1: magnitude = 0.5f; break;
        case 2: magnitude = 1.0f; break;
        case 3: magnitude = 1.5f; break;
        case 4: magnitude = 2.0f; break;
        case 5: magnitude = 3.0f; break;
        case 6: magnitude = 4.0f; break;
        default: magnitude = 6.0f; break;
    }
    return (nibble & 0x08) == 0 ? magnitude : -magnitude;
}

inline float mxfp4_e8m0_to_float(uchar bits) {
    return exp2(float(int(bits) - 127));
}

kernel void mxfp4_fma_matvec_e8m0(
    device const uchar* packed [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& groups_per_row [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = mxfp4_e8m0_to_float(scales[scale_row + group]);
            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];
            acc += mxfp4_e2m1_to_float(uchar((word >>  0) & 0x0f)) * scale * x0;
            acc += mxfp4_e2m1_to_float(uchar((word >>  4) & 0x0f)) * scale * x1;
            acc += mxfp4_e2m1_to_float(uchar((word >>  8) & 0x0f)) * scale * x2;
            acc += mxfp4_e2m1_to_float(uchar((word >> 12) & 0x0f)) * scale * x3;
            acc += mxfp4_e2m1_to_float(uchar((word >> 16) & 0x0f)) * scale * x4;
            acc += mxfp4_e2m1_to_float(uchar((word >> 20) & 0x0f)) * scale * x5;
            acc += mxfp4_e2m1_to_float(uchar((word >> 24) & 0x0f)) * scale * x6;
            acc += mxfp4_e2m1_to_float(uchar((word >> 28) & 0x0f)) * scale * x7;
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            float scale0 = mxfp4_e8m0_to_float(scales[scale_row + col0 / group_size]);
            acc += mxfp4_e2m1_to_float(byte & 0x0f) * scale0 * x0;
            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                float scale1 = mxfp4_e8m0_to_float(scales[scale_row + col1 / group_size]);
                acc += mxfp4_e2m1_to_float(byte >> 4) * scale1 * x1;
            }
        }
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

kernel void q4_swiglu_fused(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* up_packed [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const float* gate_scales [[buffer(3)]],
    device const float* gate_biases [[buffer(4)]],
    device const float* up_scales [[buffer(5)]],
    device const float* up_biases [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& groups_per_row [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* gate_words = reinterpret_cast<device const uint*>(gate_packed);
        device const uint* up_words = reinterpret_cast<device const uint*>(up_packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint gate_word = gate_words[word_row + packed_word];
            uint up_word = up_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float gate_scale = gate_scales[scale_row + group];
            float gate_bias = gate_biases[scale_row + group];
            float up_scale = up_scales[scale_row + group];
            float up_bias = up_biases[scale_row + group];

            for (uint i = 0; i < 8; i++) {
                uint shift = i * 4;
                float x = use_input_cache ? input_cache[col0 + i] : input[col0 + i];
                gate_acc += fma(float((gate_word >> shift) & 0x0f), gate_scale * x, gate_bias * x);
                up_acc += fma(float((up_word >> shift) & 0x0f), up_scale * x, up_bias * x);
            }
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar gate_byte = gate_packed[packed_row + packed_col];
            uchar up_byte = up_packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float gate_scale0 = gate_scales[scale_row + group0];
            float gate_bias0 = gate_biases[scale_row + group0];
            float up_scale0 = up_scales[scale_row + group0];
            float up_bias0 = up_biases[scale_row + group0];
            gate_acc += fma(float(gate_byte & 0x0f), gate_scale0 * x0, gate_bias0 * x0);
            up_acc += fma(float(up_byte & 0x0f), up_scale0 * x0, up_bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float gate_scale1 = gate_scales[scale_row + group1];
                float gate_bias1 = gate_biases[scale_row + group1];
                float up_scale1 = up_scales[scale_row + group1];
                float up_bias1 = up_biases[scale_row + group1];
                gate_acc += fma(float(gate_byte >> 4), gate_scale1 * x1, gate_bias1 * x1);
                up_acc += fma(float(up_byte >> 4), up_scale1 * x1, up_bias1 * x1);
            }
        }
    }
    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0) {
        output[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
    }
}

kernel void q4_swiglu_fused_bf16_scale_bias(
    device const uchar* gate_packed [[buffer(0)]],
    device const uchar* up_packed [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const ushort* gate_scales [[buffer(3)]],
    device const ushort* gate_biases [[buffer(4)]],
    device const ushort* up_scales [[buffer(5)]],
    device const ushort* up_biases [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& cols [[buffer(9)]],
    constant uint& groups_per_row [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 8;
    const uint input_cache_len = 8192;
    uint row = tile * rows_per_threadgroup + simd_group;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[8192];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= rows) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0);
    if (use_word_path) {
        device const uint* gate_words = reinterpret_cast<device const uint*>(gate_packed);
        device const uint* up_words = reinterpret_cast<device const uint*>(up_packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint gate_word = gate_words[word_row + packed_word];
            uint up_word = up_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float gate_scale = bf16_to_float(gate_scales[scale_row + group]);
            float gate_bias = bf16_to_float(gate_biases[scale_row + group]);
            float up_scale = bf16_to_float(up_scales[scale_row + group]);
            float up_bias = bf16_to_float(up_biases[scale_row + group]);

            for (uint i = 0; i < 8; i++) {
                uint shift = i * 4;
                float x = use_input_cache ? input_cache[col0 + i] : input[col0 + i];
                gate_acc += fma(float((gate_word >> shift) & 0x0f), gate_scale * x, gate_bias * x);
                up_acc += fma(float((up_word >> shift) & 0x0f), up_scale * x, up_bias * x);
            }
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar gate_byte = gate_packed[packed_row + packed_col];
            uchar up_byte = up_packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float gate_scale0 = bf16_to_float(gate_scales[scale_row + group0]);
            float gate_bias0 = bf16_to_float(gate_biases[scale_row + group0]);
            float up_scale0 = bf16_to_float(up_scales[scale_row + group0]);
            float up_bias0 = bf16_to_float(up_biases[scale_row + group0]);
            gate_acc += fma(float(gate_byte & 0x0f), gate_scale0 * x0, gate_bias0 * x0);
            up_acc += fma(float(up_byte & 0x0f), up_scale0 * x0, up_bias0 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float gate_scale1 = bf16_to_float(gate_scales[scale_row + group1]);
                float gate_bias1 = bf16_to_float(gate_biases[scale_row + group1]);
                float up_scale1 = bf16_to_float(up_scales[scale_row + group1]);
                float up_bias1 = bf16_to_float(up_biases[scale_row + group1]);
                gate_acc += fma(float(gate_byte >> 4), gate_scale1 * x1, gate_bias1 * x1);
                up_acc += fma(float(up_byte >> 4), up_scale1 * x1, up_bias1 * x1);
            }
        }
    }
    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (simd_lane == 0) {
        output[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
    }
}

kernel void q4_mmap_fma_matvec(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& packed_byte_offset [[buffer(3)]],
    constant ulong& scales_byte_offset [[buffer(4)]],
    constant ulong& biases_byte_offset [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& groups_per_row [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const float* scales = reinterpret_cast<device const float*>(weight_bytes + scales_byte_offset);
    device const float* biases = reinterpret_cast<device const float*>(weight_bytes + biases_byte_offset);
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    uint packed_stride = (cols + 1) / 2;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[4096];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= rows) {
        return;
    }

    bool row1_valid = row1 < rows;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint packed_row0 = row0 * packed_stride;
    uint packed_row1 = row1 * packed_stride;
    uint scale_row0 = row0 * groups_per_row;
    uint scale_row1 = row1 * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0) && ((packed_byte_offset & 3ul) == 0ul);
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row0 = row0 * packed_words_per_row;
        uint word_row1 = row1 * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word0 = packed_words[word_row0 + packed_word];
            uint word1 = row1_valid ? packed_words[word_row1 + packed_word] : 0u;
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale0 = scales[scale_row0 + group];
            float bias0 = biases[scale_row0 + group];
            float scale1 = row1_valid ? scales[scale_row1 + group] : 0.0f;
            float bias1 = row1_valid ? biases[scale_row1 + group] : 0.0f;

            float x0 = use_input_cache ? input_cache[col0 + 0] : input[col0 + 0];
            float x1 = use_input_cache ? input_cache[col0 + 1] : input[col0 + 1];
            float x2 = use_input_cache ? input_cache[col0 + 2] : input[col0 + 2];
            float x3 = use_input_cache ? input_cache[col0 + 3] : input[col0 + 3];
            float x4 = use_input_cache ? input_cache[col0 + 4] : input[col0 + 4];
            float x5 = use_input_cache ? input_cache[col0 + 5] : input[col0 + 5];
            float x6 = use_input_cache ? input_cache[col0 + 6] : input[col0 + 6];
            float x7 = use_input_cache ? input_cache[col0 + 7] : input[col0 + 7];

            acc0 += fma(float((word0 >>  0) & 0x0f), scale0 * x0, bias0 * x0);
            acc0 += fma(float((word0 >>  4) & 0x0f), scale0 * x1, bias0 * x1);
            acc0 += fma(float((word0 >>  8) & 0x0f), scale0 * x2, bias0 * x2);
            acc0 += fma(float((word0 >> 12) & 0x0f), scale0 * x3, bias0 * x3);
            acc0 += fma(float((word0 >> 16) & 0x0f), scale0 * x4, bias0 * x4);
            acc0 += fma(float((word0 >> 20) & 0x0f), scale0 * x5, bias0 * x5);
            acc0 += fma(float((word0 >> 24) & 0x0f), scale0 * x6, bias0 * x6);
            acc0 += fma(float((word0 >> 28) & 0x0f), scale0 * x7, bias0 * x7);

            acc1 += fma(float((word1 >>  0) & 0x0f), scale1 * x0, bias1 * x0);
            acc1 += fma(float((word1 >>  4) & 0x0f), scale1 * x1, bias1 * x1);
            acc1 += fma(float((word1 >>  8) & 0x0f), scale1 * x2, bias1 * x2);
            acc1 += fma(float((word1 >> 12) & 0x0f), scale1 * x3, bias1 * x3);
            acc1 += fma(float((word1 >> 16) & 0x0f), scale1 * x4, bias1 * x4);
            acc1 += fma(float((word1 >> 20) & 0x0f), scale1 * x5, bias1 * x5);
            acc1 += fma(float((word1 >> 24) & 0x0f), scale1 * x6, bias1 * x6);
            acc1 += fma(float((word1 >> 28) & 0x0f), scale1 * x7, bias1 * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte0 = packed[packed_row0 + packed_col];
            uchar byte1 = row1_valid ? packed[packed_row1 + packed_col] : uchar(0);
            uint col0 = packed_col * 2;
            float x0 = use_input_cache ? input_cache[col0] : input[col0];
            uint group0 = col0 / group_size;
            float scale00 = scales[scale_row0 + group0];
            float bias00 = biases[scale_row0 + group0];
            float scale10 = row1_valid ? scales[scale_row1 + group0] : 0.0f;
            float bias10 = row1_valid ? biases[scale_row1 + group0] : 0.0f;
            acc0 += fma(float(byte0 & 0x0f), scale00 * x0, bias00 * x0);
            acc1 += fma(float(byte1 & 0x0f), scale10 * x0, bias10 * x0);

            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = use_input_cache ? input_cache[col1] : input[col1];
                uint group1 = col1 / group_size;
                float scale01 = scales[scale_row0 + group1];
                float bias01 = biases[scale_row0 + group1];
                float scale11 = row1_valid ? scales[scale_row1 + group1] : 0.0f;
                float bias11 = row1_valid ? biases[scale_row1 + group1] : 0.0f;
                acc0 += fma(float(byte0 >> 4), scale01 * x1, bias01 * x1);
                acc1 += fma(float(byte1 >> 4), scale11 * x1, bias11 * x1);
            }
        }
    }
    float sum0 = simd_sum(acc0);
    float sum1 = simd_sum(acc1);
    if (simd_lane == 0) {
        output[row0] = sum0;
        if (row1_valid) {
            output[row1] = sum1;
        }
    }
}

kernel void q4_mmap_fma_matvec_bf16_scale_bias(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& packed_byte_offset [[buffer(3)]],
    constant ulong& scales_byte_offset [[buffer(4)]],
    constant ulong& biases_byte_offset [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& groups_per_row [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const ushort* scales = reinterpret_cast<device const ushort*>(weight_bytes + scales_byte_offset);
    device const ushort* biases = reinterpret_cast<device const ushort*>(weight_bytes + biases_byte_offset);
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    bool use_input_cache = cols <= input_cache_len;
    threadgroup float input_cache[4096];
    if (use_input_cache) {
        for (uint col = lid; col < cols; col += 256) {
            input_cache[col] = input[col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= rows) {
        return;
    }

    bool row1_valid = row1 < rows;
    bool word_aligned = (packed_byte_offset & 3ul) == 0ul;
    float sum0 = q4_fma_row_bf16_ptrs(
        packed, scales, biases, input, input_cache,
        row0, cols, groups_per_row, group_size,
        word_aligned, use_input_cache, simd_lane);
    float sum1 = row1_valid ? q4_fma_row_bf16_ptrs(
        packed, scales, biases, input, input_cache,
        row1, cols, groups_per_row, group_size,
        word_aligned, use_input_cache, simd_lane) : 0.0f;
    if (simd_lane == 0) {
        output[row0] = sum0;
        if (row1_valid) {
            output[row1] = sum1;
        }
    }
}

inline uint q4_batch_projection_for_row(
    uint row,
    device const uint* row_offsets,
    device const uint* rows,
    uint projection_count) {
    for (uint idx = 0; idx < projection_count; idx++) {
        uint start = row_offsets[idx];
        uint end = start + rows[idx];
        if (row >= start && row < end) {
            return idx;
        }
    }
    return projection_count;
}

inline float q4_mmap_fma_row_f32(
    device const uchar* weight_bytes,
    device const float* input,
    threadgroup float* input_cache,
    ulong packed_byte_offset,
    ulong scales_byte_offset,
    ulong biases_byte_offset,
    uint row,
    uint cols,
    uint groups_per_row,
    uint group_size,
    uint simd_lane) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const float* scales = reinterpret_cast<device const float*>(weight_bytes + scales_byte_offset);
    device const float* biases = reinterpret_cast<device const float*>(weight_bytes + biases_byte_offset);
    uint packed_stride = (cols + 1) / 2;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = (cols % 8 == 0) && (group_size % 8 == 0) && ((packed_byte_offset & 3ul) == 0ul);
    float acc = 0.0f;
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = scales[scale_row + group];
            float bias = biases[scale_row + group];
            float x0 = input_cache[col0 + 0];
            float x1 = input_cache[col0 + 1];
            float x2 = input_cache[col0 + 2];
            float x3 = input_cache[col0 + 3];
            float x4 = input_cache[col0 + 4];
            float x5 = input_cache[col0 + 5];
            float x6 = input_cache[col0 + 6];
            float x7 = input_cache[col0 + 7];
            acc += fma(float((word >>  0) & 0x0f), scale * x0, bias * x0);
            acc += fma(float((word >>  4) & 0x0f), scale * x1, bias * x1);
            acc += fma(float((word >>  8) & 0x0f), scale * x2, bias * x2);
            acc += fma(float((word >> 12) & 0x0f), scale * x3, bias * x3);
            acc += fma(float((word >> 16) & 0x0f), scale * x4, bias * x4);
            acc += fma(float((word >> 20) & 0x0f), scale * x5, bias * x5);
            acc += fma(float((word >> 24) & 0x0f), scale * x6, bias * x6);
            acc += fma(float((word >> 28) & 0x0f), scale * x7, bias * x7);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x0 = input_cache[col0];
            uint group0 = col0 / group_size;
            float scale0 = scales[scale_row + group0];
            float bias0 = biases[scale_row + group0];
            acc += fma(float(byte & 0x0f), scale0 * x0, bias0 * x0);
            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x1 = input_cache[col1];
                uint group1 = col1 / group_size;
                float scale1 = scales[scale_row + group1];
                float bias1 = biases[scale_row + group1];
                acc += fma(float(byte >> 4), scale1 * x1, bias1 * x1);
            }
        }
    }
    return simd_sum(acc);
}

inline float q4_mmap_fma_row_bf16(
    device const uchar* weight_bytes,
    device const float* input,
    threadgroup float* input_cache,
    ulong packed_byte_offset,
    ulong scales_byte_offset,
    ulong biases_byte_offset,
    uint row,
    uint cols,
    uint groups_per_row,
    uint group_size,
    uint simd_lane) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const ushort* scales = reinterpret_cast<device const ushort*>(weight_bytes + scales_byte_offset);
    device const ushort* biases = reinterpret_cast<device const ushort*>(weight_bytes + biases_byte_offset);
    bool use_input_cache = cols <= 4096;
    return q4_fma_row_bf16_ptrs(
        packed, scales, biases, input, input_cache,
        row, cols, groups_per_row, group_size,
        (packed_byte_offset & 3ul) == 0ul, use_input_cache, simd_lane);
}

inline float2 q4_mmap_fma_row_bf16_pair(
    device const uchar* weight_bytes,
    threadgroup float* input_cache,
    ulong packed_byte_offset,
    ulong scales_byte_offset,
    ulong biases_byte_offset,
    uint row,
    uint cols,
    uint groups_per_row,
    uint group_size,
    uint simd_lane) {
    device const uchar* packed = weight_bytes + packed_byte_offset;
    device const ushort* scales = reinterpret_cast<device const ushort*>(weight_bytes + scales_byte_offset);
    device const ushort* biases = reinterpret_cast<device const ushort*>(weight_bytes + biases_byte_offset);
    uint packed_stride = (cols + 1) / 2;
    uint packed_row = row * packed_stride;
    uint scale_row = row * groups_per_row;
    bool use_word_path = ((packed_byte_offset & 3ul) == 0ul) &&
        (cols % 8 == 0) && (group_size % 8 == 0);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (use_word_path) {
        device const uint* packed_words = reinterpret_cast<device const uint*>(packed);
        uint packed_words_per_row = cols / 8;
        uint word_row = row * packed_words_per_row;
        for (uint packed_word = simd_lane; packed_word < packed_words_per_row; packed_word += 32) {
            uint word = packed_words[word_row + packed_word];
            uint col0 = packed_word * 8;
            uint group = col0 / group_size;
            float scale = bf16_to_float(scales[scale_row + group]);
            float bias = bf16_to_float(biases[scale_row + group]);
            float quantized = float((word >> 0) & 0x0f);
            float x0 = input_cache[col0 + 0];
            float x1 = input_cache[cols + col0 + 0];
            acc0 += fma(quantized, scale * x0, bias * x0);
            acc1 += fma(quantized, scale * x1, bias * x1);
            quantized = float((word >> 4) & 0x0f);
            x0 = input_cache[col0 + 1];
            x1 = input_cache[cols + col0 + 1];
            acc0 += fma(quantized, scale * x0, bias * x0);
            acc1 += fma(quantized, scale * x1, bias * x1);
            quantized = float((word >> 8) & 0x0f);
            x0 = input_cache[col0 + 2];
            x1 = input_cache[cols + col0 + 2];
            acc0 += fma(quantized, scale * x0, bias * x0);
            acc1 += fma(quantized, scale * x1, bias * x1);
            quantized = float((word >> 12) & 0x0f);
            x0 = input_cache[col0 + 3];
            x1 = input_cache[cols + col0 + 3];
            acc0 += fma(quantized, scale * x0, bias * x0);
            acc1 += fma(quantized, scale * x1, bias * x1);
            quantized = float((word >> 16) & 0x0f);
            x0 = input_cache[col0 + 4];
            x1 = input_cache[cols + col0 + 4];
            acc0 += fma(quantized, scale * x0, bias * x0);
            acc1 += fma(quantized, scale * x1, bias * x1);
            quantized = float((word >> 20) & 0x0f);
            x0 = input_cache[col0 + 5];
            x1 = input_cache[cols + col0 + 5];
            acc0 += fma(quantized, scale * x0, bias * x0);
            acc1 += fma(quantized, scale * x1, bias * x1);
            quantized = float((word >> 24) & 0x0f);
            x0 = input_cache[col0 + 6];
            x1 = input_cache[cols + col0 + 6];
            acc0 += fma(quantized, scale * x0, bias * x0);
            acc1 += fma(quantized, scale * x1, bias * x1);
            quantized = float((word >> 28) & 0x0f);
            x0 = input_cache[col0 + 7];
            x1 = input_cache[cols + col0 + 7];
            acc0 += fma(quantized, scale * x0, bias * x0);
            acc1 += fma(quantized, scale * x1, bias * x1);
        }
    } else {
        for (uint packed_col = simd_lane; packed_col < packed_stride; packed_col += 32) {
            uchar byte = packed[packed_row + packed_col];
            uint col0 = packed_col * 2;
            float x00 = input_cache[col0];
            float x01 = input_cache[cols + col0];
            uint group0 = col0 / group_size;
            float scale0 = bf16_to_float(scales[scale_row + group0]);
            float bias0 = bf16_to_float(biases[scale_row + group0]);
            float quantized0 = float(byte & 0x0f);
            acc0 += fma(quantized0, scale0 * x00, bias0 * x00);
            acc1 += fma(quantized0, scale0 * x01, bias0 * x01);
            uint col1 = col0 + 1;
            if (col1 < cols) {
                float x10 = input_cache[col1];
                float x11 = input_cache[cols + col1];
                uint group1 = col1 / group_size;
                float scale1 = bf16_to_float(scales[scale_row + group1]);
                float bias1 = bf16_to_float(biases[scale_row + group1]);
                float quantized1 = float(byte >> 4);
                acc0 += fma(quantized1, scale1 * x10, bias1 * x10);
                acc1 += fma(quantized1, scale1 * x11, bias1 * x11);
            }
        }
    }
    return float2(simd_sum(acc0), simd_sum(acc1));
}

kernel void q4_mmap_fma_multilinear_bf16_scale_bias(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& packed_byte_offset [[buffer(3)]],
    constant ulong& scales_byte_offset [[buffer(4)]],
    constant ulong& biases_byte_offset [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& groups_per_row [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    constant uint& rows_per_head [[buffer(10)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint row0 = tile * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    uint head = (tile * rows_per_threadgroup) / rows_per_head;
    device const float* input = inputs + head * cols;
    threadgroup float input_cache[4096];
    for (uint col = lid; col < cols && col < input_cache_len; col += 256) {
        input_cache[col] = input[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 < rows) {
        float sum = q4_mmap_fma_row_bf16(
            weight_bytes, input, input_cache,
            packed_byte_offset, scales_byte_offset, biases_byte_offset,
            row0, cols, groups_per_row, group_size, simd_lane);
        if (simd_lane == 0) {
            output[row0] = sum;
        }
    }
    if (row1 < rows) {
        float sum = q4_mmap_fma_row_bf16(
            weight_bytes, input, input_cache,
            packed_byte_offset, scales_byte_offset, biases_byte_offset,
            row1, cols, groups_per_row, group_size, simd_lane);
        if (simd_lane == 0) {
            output[row1] = sum;
        }
    }
}

kernel void q4_mmap_fma_matvec_batch(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const ulong* packed_byte_offsets [[buffer(3)]],
    device const ulong* scales_byte_offsets [[buffer(4)]],
    device const ulong* biases_byte_offsets [[buffer(5)]],
    device const uint* row_offsets [[buffer(6)]],
    device const uint* rows [[buffer(7)]],
    device const uint* groups_per_rows [[buffer(8)]],
    constant uint& projection_count [[buffer(9)]],
    constant uint& cols [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint total_rows = row_offsets[projection_count - 1] + rows[projection_count - 1];
    uint input_row = tile.y;
    input += input_row * cols;
    output += input_row * total_rows;
    uint row0 = tile.x * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    threadgroup float input_cache[4096];
    for (uint col = lid.x; col < cols && col < input_cache_len; col += 256) {
        input_cache[col] = input[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint projection0 = q4_batch_projection_for_row(row0, row_offsets, rows, projection_count);
    if (projection0 < projection_count) {
        uint local_row = row0 - row_offsets[projection0];
        float sum = q4_mmap_fma_row_f32(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection0],
            scales_byte_offsets[projection0],
            biases_byte_offsets[projection0],
            local_row, cols, groups_per_rows[projection0], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row0] = sum;
        }
    }
    uint projection1 = q4_batch_projection_for_row(row1, row_offsets, rows, projection_count);
    if (projection1 < projection_count) {
        uint local_row = row1 - row_offsets[projection1];
        float sum = q4_mmap_fma_row_f32(
            weight_bytes, input, input_cache,
            packed_byte_offsets[projection1],
            scales_byte_offsets[projection1],
            biases_byte_offsets[projection1],
            local_row, cols, groups_per_rows[projection1], group_size, simd_lane);
        if (simd_lane == 0) {
            output[row1] = sum;
        }
    }
}

kernel void q4_mmap_fma_matvec_batch_bf16_scale_bias(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const ulong* packed_byte_offsets [[buffer(3)]],
    device const ulong* scales_byte_offsets [[buffer(4)]],
    device const ulong* biases_byte_offsets [[buffer(5)]],
    device const uint* row_offsets [[buffer(6)]],
    device const uint* rows [[buffer(7)]],
    device const uint* groups_per_rows [[buffer(8)]],
    constant uint& projection_count [[buffer(9)]],
    constant uint& cols [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    constant uint& input_rows [[buffer(12)]],
    constant uint& input_rows_per_threadgroup [[buffer(13)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint rows_per_threadgroup = 16;
    const uint input_cache_len = 4096;
    uint total_rows = row_offsets[projection_count - 1] + rows[projection_count - 1];
    uint input_row = tile.y * input_rows_per_threadgroup;
    uint active_input_rows = min(input_rows_per_threadgroup, input_rows - input_row);
    uint row0 = tile.x * rows_per_threadgroup + simd_group;
    uint row1 = row0 + 8;
    threadgroup float input_cache[4096];
    uint cached_values = min(active_input_rows * cols, input_cache_len);
    for (uint value = lid.x; value < cached_values; value += 256) {
        input_cache[value] = input[input_row * cols + value];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint projection0 = q4_batch_projection_for_row(row0, row_offsets, rows, projection_count);
    if (projection0 < projection_count) {
        uint local_row = row0 - row_offsets[projection0];
        if (active_input_rows == 2) {
            float2 sums = q4_mmap_fma_row_bf16_pair(
                weight_bytes, input_cache,
                packed_byte_offsets[projection0],
                scales_byte_offsets[projection0],
                biases_byte_offsets[projection0],
                local_row, cols, groups_per_rows[projection0], group_size, simd_lane);
            if (simd_lane == 0) {
                output[input_row * total_rows + row0] = sums.x;
                output[(input_row + 1) * total_rows + row0] = sums.y;
            }
        } else {
            float sum = q4_mmap_fma_row_bf16(
                weight_bytes, input + input_row * cols, input_cache,
                packed_byte_offsets[projection0],
                scales_byte_offsets[projection0],
                biases_byte_offsets[projection0],
                local_row, cols, groups_per_rows[projection0], group_size, simd_lane);
            if (simd_lane == 0) {
                output[input_row * total_rows + row0] = sum;
            }
        }
    }
    uint projection1 = q4_batch_projection_for_row(row1, row_offsets, rows, projection_count);
    if (projection1 < projection_count) {
        uint local_row = row1 - row_offsets[projection1];
        if (active_input_rows == 2) {
            float2 sums = q4_mmap_fma_row_bf16_pair(
                weight_bytes, input_cache,
                packed_byte_offsets[projection1],
                scales_byte_offsets[projection1],
                biases_byte_offsets[projection1],
                local_row, cols, groups_per_rows[projection1], group_size, simd_lane);
            if (simd_lane == 0) {
                output[input_row * total_rows + row1] = sums.x;
                output[(input_row + 1) * total_rows + row1] = sums.y;
            }
        } else {
            float sum = q4_mmap_fma_row_bf16(
                weight_bytes, input + input_row * cols, input_cache,
                packed_byte_offsets[projection1],
                scales_byte_offsets[projection1],
                biases_byte_offsets[projection1],
                local_row, cols, groups_per_rows[projection1], group_size, simd_lane);
            if (simd_lane == 0) {
                output[input_row * total_rows + row1] = sum;
            }
        }
    }
}

kernel void dense_mmap_fma_matvec_bf16(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) { return; }
    device const ushort* weights = reinterpret_cast<device const ushort*>(weight_bytes + weight_byte_offset);
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(bf16_to_float(weights[start + col]), input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_mmap_fma_matvec_f16(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) { return; }
    device const half* weights = reinterpret_cast<device const half*>(weight_bytes + weight_byte_offset);
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(float(weights[start + col]), input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_mmap_fma_matvec_f32(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) { return; }
    device const float* weights = reinterpret_cast<device const float*>(weight_bytes + weight_byte_offset);
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(weights[start + col], input[col], acc);
    }
    output[row] = acc;
}

kernel void dense_mmap_fma_matrix_bf16(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    uint row = position.x;
    uint input_row = position.y;
    if (row >= rows) { return; }
    device const ushort* weights = reinterpret_cast<device const ushort*>(weight_bytes + weight_byte_offset);
    input += input_row * cols;
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(bf16_to_float(weights[start + col]), input[col], acc);
    }
    output[input_row * rows + row] = acc;
}

kernel void dense_mmap_fma_matrix_f16(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    uint row = position.x;
    uint input_row = position.y;
    if (row >= rows) { return; }
    device const half* weights = reinterpret_cast<device const half*>(weight_bytes + weight_byte_offset);
    input += input_row * cols;
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(float(weights[start + col]), input[col], acc);
    }
    output[input_row * rows + row] = acc;
}

kernel void dense_mmap_fma_matrix_f32(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& weight_byte_offset [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    uint row = position.x;
    uint input_row = position.y;
    if (row >= rows) { return; }
    device const float* weights = reinterpret_cast<device const float*>(weight_bytes + weight_byte_offset);
    input += input_row * cols;
    float acc = 0.0f;
    uint start = row * cols;
    for (uint col = 0; col < cols; ++col) {
        acc = fma(weights[start + col], input[col], acc);
    }
    output[input_row * rows + row] = acc;
}

kernel void rms_norm_reduced(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint threads = 256;
    uint lid = thread_position.x;
    input += group.y * width;
    output += group.y * width;
    threadgroup float partial[32];
    float sum = 0.0f;
    for (uint i = lid; i < width; i += threads) {
        sum += input[i] * input[i];
    }
    float simd_value = simd_sum(sum);
    if (simd_lane == 0) {
        partial[simd_group] = simd_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float value = simd_lane < 8 ? partial[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            partial[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(partial[0] / float(max(width, 1u)) + epsilon);
    for (uint i = lid; i < width; i += threads) {
        output[i] = input[i] * scale * weight[i];
    }
}

#pragma clang fp contract(off)
kernel void qwen_final_rms_norm_row(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0) {
        return;
    }
    float sum = 0.0f;
    for (uint column = 0; column < width; ++column) {
        float value = input[column];
        sum += value * value;
    }
    float scale = 1.0f / sqrt(sum / float(max(width, 1u)) + epsilon);
    for (uint column = 0; column < width; ++column) {
        float normalized = input[column] * scale;
        output[column] = normalized * weight[column];
    }
}
#pragma clang fp contract(on)

kernel void residual_add_rms_norm(
    device const float* projected [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    device float* normed [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint threads = 256;
    uint lid = thread_position.x;
    projected += group.y * width;
    residual += group.y * width;
    hidden += group.y * width;
    normed += group.y * width;
    threadgroup float partial[32];
    float sum = 0.0f;
    for (uint i = lid; i < width; i += threads) {
        float value = projected[i] + residual[i];
        sum += value * value;
    }
    float simd_value = simd_sum(sum);
    if (simd_lane == 0) {
        partial[simd_group] = simd_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float value = simd_lane < 8 ? partial[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0) {
            partial[0] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(partial[0] / float(max(width, 1u)) + epsilon);
    for (uint i = lid; i < width; i += threads) {
        float value = projected[i] + residual[i];
        hidden[i] = value;
        normed[i] = value * scale * weight[i];
    }
}

kernel void attention_scores(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint token [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < width; ++i) {
        acc = fma(query[i], keys[token * width + i], acc);
    }
    scores[token] = acc * rsqrt(float(max(head_dim, 1u)));
}

#pragma clang fp contract(off)
kernel void qwen_prepare_qkv_rows(
    device const float* projected [[buffer(0)]],
    device const float* q_norm_weight [[buffer(1)]],
    device const float* k_norm_weight [[buffer(2)]],
    device const float* rope_sin [[buffer(3)]],
    device const float* rope_cos [[buffer(4)]],
    device float* queries [[buffer(5)]],
    device float* query_gates [[buffer(6)]],
    device float* keys [[buffer(7)]],
    device float* values [[buffer(8)]],
    constant uint& projection_width [[buffer(9)]],
    constant uint& q_offset [[buffer(10)]],
    constant uint& k_offset [[buffer(11)]],
    constant uint& v_offset [[buffer(12)]],
    constant uint& query_heads [[buffer(13)]],
    constant uint& kv_heads [[buffer(14)]],
    constant uint& head_dim [[buffer(15)]],
    constant uint& rotary_half [[buffer(16)]],
    constant uint& prefix_rows [[buffer(17)]],
    constant uint& gated [[buffer(18)]],
    uint2 position [[thread_position_in_grid]]) {
    uint head = position.x;
    uint row = position.y;
    uint row_projection = row * projection_width;
    uint rotation = row * rotary_half;

    if (head < query_heads) {
        uint source = row_projection + q_offset
            + head * head_dim * (gated != 0 ? 2u : 1u);
        uint target = (row * query_heads + head) * head_dim;
        float sum = 0.0f;
        for (uint dim = 0; dim < head_dim; ++dim) {
            float raw = projected[source + dim];
            sum += raw * raw;
        }
        float scale = 1.0f / sqrt(sum / float(max(head_dim, 1u)) + 1.0e-6f);
        for (uint dim = 0; dim < head_dim; ++dim) {
            float normalized = projected[source + dim] * (scale * q_norm_weight[dim]);
            if (dim < 2u * rotary_half) {
                uint pair = dim < rotary_half ? dim : dim - rotary_half;
                uint other_dim = dim < rotary_half ? dim + rotary_half : dim - rotary_half;
                float other = projected[source + other_dim]
                    * (scale * q_norm_weight[other_dim]);
                float sine = rope_sin[rotation + pair];
                float cosine = rope_cos[rotation + pair];
                normalized = dim < rotary_half
                    ? normalized * cosine - other * sine
                    : other * sine + normalized * cosine;
            }
            queries[target + dim] = normalized;
            if (gated != 0) {
                query_gates[target + dim] = projected[source + head_dim + dim];
            }
        }
    }

    if (head < kv_heads) {
        uint key_source = row_projection + k_offset + head * head_dim;
        uint value_source = row_projection + v_offset + head * head_dim;
        uint target = ((prefix_rows + row) * kv_heads + head) * head_dim;
        float sum = 0.0f;
        for (uint dim = 0; dim < head_dim; ++dim) {
            float raw = projected[key_source + dim];
            sum += raw * raw;
        }
        float scale = 1.0f / sqrt(sum / float(max(head_dim, 1u)) + 1.0e-6f);
        for (uint dim = 0; dim < head_dim; ++dim) {
            float normalized = projected[key_source + dim] * (scale * k_norm_weight[dim]);
            if (dim < 2u * rotary_half) {
                uint pair = dim < rotary_half ? dim : dim - rotary_half;
                uint other_dim = dim < rotary_half ? dim + rotary_half : dim - rotary_half;
                float other = projected[key_source + other_dim]
                    * (scale * k_norm_weight[other_dim]);
                float sine = rope_sin[rotation + pair];
                float cosine = rope_cos[rotation + pair];
                normalized = dim < rotary_half
                    ? normalized * cosine - other * sine
                    : other * sine + normalized * cosine;
            }
            keys[target + dim] = normalized;
            values[target + dim] = projected[value_source + dim];
        }
    }
}
#pragma clang fp contract(on)

#pragma clang fp contract(off)
struct qwen_float_pair {
    float hi;
    float lo;
};

constant uint2 qwen_exp_coefficient_bits[13] = {
    uint2(0x3f800000u, 0x00000000u),
    uint2(0x3f800000u, 0x00000000u),
    uint2(0x3f000000u, 0x00000000u),
    uint2(0x3e2aaaabu, 0xb1aaaaabu),
    uint2(0x3d2aaaabu, 0xb0aaaaabu),
    uint2(0x3c088889u, 0xafeeeeefu),
    uint2(0x3ab60b61u, 0xae13e93fu),
    uint2(0x39500d01u, 0xac3fcbfdu),
    uint2(0x37d00d01u, 0xaabfcbfdu),
    uint2(0x3638ef1du, 0x292ad8e6u),
    uint2(0x3493f27eu, 0xa808760au),
    uint2(0x32d7322bu, 0x25fea89cu),
    uint2(0x310f76c7u, 0x24ff8d8au),
};

inline qwen_float_pair qwen_pair_from_bits(uint2 bits) {
    return qwen_float_pair{as_type<float>(bits.x), as_type<float>(bits.y)};
}

inline qwen_float_pair qwen_pair_normalize(float hi, float lo) {
    float sum = hi + lo;
    return qwen_float_pair{sum, lo - (sum - hi)};
}

inline qwen_float_pair qwen_pair_add(qwen_float_pair a, qwen_float_pair b) {
    float sum = a.hi + b.hi;
    float other_virtual = sum - a.hi;
    float error = (a.hi - (sum - other_virtual)) + (b.hi - other_virtual);
    error += a.lo + b.lo;
    return qwen_pair_normalize(sum, error);
}

inline qwen_float_pair qwen_pair_multiply(qwen_float_pair a, qwen_float_pair b) {
    float product = a.hi * b.hi;
    float error = fma(a.hi, b.hi, -product);
    error += a.hi * b.lo;
    error += a.lo * b.hi;
    error += a.lo * b.lo;
    return qwen_pair_normalize(product, error);
}

inline float qwen_attention_exp(float value) {
    qwen_float_pair input = qwen_float_pair{value, 0.0f};
    qwen_float_pair log2_e = qwen_pair_from_bits(uint2(0x3fb8aa3bu, 0x32a57060u));
    qwen_float_pair ln_2 = qwen_pair_from_bits(uint2(0x3f317218u, 0xb102e308u));
    qwen_float_pair scaled = qwen_pair_multiply(input, log2_e);
    int exponent = int(rint(scaled.hi + scaled.lo));
    qwen_float_pair residual = qwen_pair_add(
        input,
        qwen_pair_multiply(qwen_float_pair{-float(exponent), 0.0f}, ln_2));
    qwen_float_pair polynomial = qwen_pair_from_bits(qwen_exp_coefficient_bits[12]);
    for (int degree = 11; degree >= 0; --degree) {
        polynomial = qwen_pair_add(
            qwen_pair_from_bits(qwen_exp_coefficient_bits[degree]),
            qwen_pair_multiply(polynomial, residual));
    }
    if (exponent == 128) {
        polynomial = qwen_pair_multiply(polynomial, qwen_float_pair{2.0f, 0.0f});
        exponent = 127;
    }
    float scale = as_type<float>(uint(exponent + 127) << 23);
    return polynomial.hi * scale + polynomial.lo * scale;
}

inline float qwen_attention_sigmoid(float value) {
    if (isnan(value)) {
        return value;
    }
    if (value >= 17.0f) {
        return 1.0f;
    }
    if (value <= -88.72284f) {
        return 0.0f;
    }
    return 1.0f / (1.0f + qwen_attention_exp(-value));
}
#pragma clang fp contract(on)

#pragma clang fp contract(off)
kernel void qwen_apply_attention_gate(
    device float* values [[buffer(0)]],
    device const float* gates [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        float gated = values[index] * qwen_attention_sigmoid(gates[index]);
        values[index] = gated;
    }
}
#pragma clang fp contract(on)

kernel void qwen_causal_attention_rows(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const float* query_gates [[buffer(4)]],
    constant uint& query_rows [[buffer(5)]],
    constant uint& prefix_rows [[buffer(6)]],
    constant uint& query_heads [[buffer(7)]],
    constant uint& kv_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& gated [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 thread_position [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint thread_count = 256;
    threadgroup float partial_sums[8];
    threadgroup float shared_score;
    uint query_row = group.x;
    uint query_head = group.y;
    uint dimension = thread_position.x;
    if (query_row >= query_rows || query_head >= query_heads) {
        return;
    }
    uint groups_per_kv = max(query_heads / max(kv_heads, 1u), 1u);
    uint kv_head = min(query_head / groups_per_kv, max(kv_heads, 1u) - 1u);
    uint query_offset = (query_row * query_heads + query_head) * head_dim;
    uint key_count = prefix_rows + query_row + 1;
    float accumulator = 0.0f;
    float denominator = 0.0f;
    float maximum = -INFINITY;
    float query_value = dimension < head_dim ? queries[query_offset + dimension] : 0.0f;
    for (uint key_row = 0; key_row < key_count; ++key_row) {
        uint kv_offset = (key_row * kv_heads + kv_head) * head_dim;
        float product = dimension < head_dim ? query_value * keys[kv_offset + dimension] : 0.0f;
        float subgroup_sum = simd_sum(product);
        if (simd_lane == 0) {
            partial_sums[simd_group] = subgroup_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_position.x == 0) {
            float score = 0.0f;
            for (uint subgroup = 0; subgroup < thread_count / 32; ++subgroup) {
                score += partial_sums[subgroup];
            }
            shared_score = score * rsqrt(float(max(head_dim, 1u)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float next_maximum = max(maximum, shared_score);
        float previous_scale = isinf(maximum) ? 0.0f : exp(maximum - next_maximum);
        float current_scale = exp(shared_score - next_maximum);
        if (dimension < head_dim) {
            accumulator = accumulator * previous_scale
                + values[kv_offset + dimension] * current_scale;
        }
        denominator = denominator * previous_scale + current_scale;
        maximum = next_maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (dimension < head_dim) {
        float value = accumulator / max(denominator, 1.0e-20f);
        output[query_offset + dimension] = value;
    }
}

kernel void glm_mla_prepare_query_kv(
    device const float* query [[buffer(0)]],
    device const float* compressed [[buffer(1)]],
    device float* query_nope [[buffer(2)]],
    device float* query_rope [[buffer(3)]],
    device float* record_latents [[buffer(4)]],
    device float* record_rotary [[buffer(5)]],
    device const float* rope_cos [[buffer(6)]],
    device const float* rope_sin [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    constant uint& nope_dim [[buffer(9)]],
    constant uint& rope_dim [[buffer(10)]],
    constant uint& latent_rank [[buffer(11)]],
    constant uint& sequence [[buffer(12)]],
    uint index [[thread_position_in_grid]]) {
    uint nope_values = heads * nope_dim;
    uint query_rope_values = heads * rope_dim;
    uint rope_half = rope_dim / 2;
    if (index < nope_values) {
        uint head = index / nope_dim;
        uint dim = index - head * nope_dim;
        query_nope[index] = query[head * (nope_dim + rope_dim) + dim];
        return;
    }
    uint section = index - nope_values;
    if (section < query_rope_values) {
        uint head = section / rope_dim;
        uint dim = section - head * rope_dim;
        uint pair = dim < rope_half ? dim : dim - rope_half;
        uint input_offset = head * (nope_dim + rope_dim) + nope_dim + pair * 2;
        float a = query[input_offset];
        float b = query[input_offset + 1];
        query_rope[section] = dim < rope_half
            ? a * rope_cos[pair] - b * rope_sin[pair]
            : b * rope_cos[pair] + a * rope_sin[pair];
        return;
    }
    section -= query_rope_values;
    if (section < latent_rank) {
        record_latents[(sequence - 1) * latent_rank + section] = compressed[section];
        return;
    }
    section -= latent_rank;
    if (section < rope_dim) {
        uint pair = section < rope_half ? section : section - rope_half;
        float a = compressed[latent_rank + pair * 2];
        float b = compressed[latent_rank + pair * 2 + 1];
        record_rotary[(sequence - 1) * rope_dim + section] = section < rope_half
            ? a * rope_cos[pair] - b * rope_sin[pair]
            : b * rope_cos[pair] + a * rope_sin[pair];
    }
}

kernel void glm_mla_absorbed_scores(
    device const float* absorbed_queries [[buffer(0)]],
    device const float* query_rope [[buffer(1)]],
    device const float* record_latents [[buffer(2)]],
    device const float* record_rotary [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& latent_rank [[buffer(6)]],
    constant uint& rope_dim [[buffer(7)]],
    constant uint& sequence [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    uint total = heads * sequence;
    if (index >= total) { return; }
    uint head = index / sequence;
    uint position = index - head * sequence;
    float score = 0.0f;
    uint query_latent_offset = head * latent_rank;
    uint record_latent_offset = position * latent_rank;
    for (uint dim = 0; dim < latent_rank; ++dim) {
        score = fma(
            absorbed_queries[query_latent_offset + dim],
            record_latents[record_latent_offset + dim],
            score);
    }
    uint query_rope_offset = head * rope_dim;
    uint record_rope_offset = position * rope_dim;
    for (uint dim = 0; dim < rope_dim; ++dim) {
        score = fma(
            query_rope[query_rope_offset + dim],
            record_rotary[record_rope_offset + dim],
            score);
    }
    scores[index] = score * scale;
}

kernel void glm_mla_softmax(
    device float* scores [[buffer(0)]],
    constant uint& heads [[buffer(1)]],
    constant uint& sequence [[buffer(2)]],
    uint head [[thread_position_in_grid]]) {
    if (head >= heads) { return; }
    uint offset = head * sequence;
    float maximum = scores[offset];
    for (uint position = 1; position < sequence; ++position) {
        maximum = max(maximum, scores[offset + position]);
    }
    float denominator = 0.0f;
    for (uint position = 0; position < sequence; ++position) {
        float value = exp(scores[offset + position] - maximum);
        scores[offset + position] = value;
        denominator += value;
    }
    float inverse = 1.0f / max(denominator, 1.0e-20f);
    for (uint position = 0; position < sequence; ++position) {
        scores[offset + position] *= inverse;
    }
}

kernel void glm_mla_context(
    device const float* scores [[buffer(0)]],
    device const float* record_latents [[buffer(1)]],
    device float* contexts [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& latent_rank [[buffer(4)]],
    constant uint& sequence [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    uint total = heads * latent_rank;
    if (index >= total) { return; }
    uint head = index / latent_rank;
    uint dim = index - head * latent_rank;
    float value = 0.0f;
    for (uint position = 0; position < sequence; ++position) {
        value = fma(
            scores[head * sequence + position],
            record_latents[position * latent_rank + dim],
            value);
    }
    contexts[index] = value;
}

kernel void expert_mlp_fused(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* down [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& intermediate [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < intermediate; ++i) {
        float g = gate[i] / (1.0f + exp(-gate[i]));
        acc = fma(down[row * intermediate + i], g * up[i], acc);
    }
    output[row] = acc * rsqrt(float(max(intermediate, 1u)));
}

kernel void silu_product(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx];
}

kernel void shared_expert_activation(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* router [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& intermediate [[buffer(4)]],
    constant uint& total_intermediate [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= total_intermediate) { return; }
    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx];
}

kernel void combine_expert_phase(
    device const float* residual [[buffer(0)]],
    device const float* shared [[buffer(1)]],
    device const float* expert_outputs [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* hidden [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& active_experts [[buffer(6)]],
    device const float* shared_router [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    float route = shared_router[0];
    float shared_weight = 1.0f / (1.0f + exp(-route));
    float moe = 0.0f;
    for (uint expert = 0; expert < active_experts; ++expert) {
        moe += weights[expert] * expert_outputs[expert * width + idx];
    }
    hidden[idx] = residual[idx] + moe + shared_weight * shared[idx];
}

kernel void qwen_layer_major_gather(
    device const float* input [[buffer(0)]],
    device const uint* source_rows [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& values [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= values) { return; }
    uint output_row = idx / width;
    uint col = idx - output_row * width;
    output[idx] = input[source_rows[output_row] * width + col];
}

kernel void qwen_layer_major_combine(
    device const float* residual [[buffer(0)]],
    device const float* shared [[buffer(1)]],
    device const float* grouped_expert_outputs [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const uint* grouped_output_indices [[buffer(4)]],
    device const float* shared_router [[buffer(5)]],
    device float* hidden [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& active_experts [[buffer(9)]],
    constant uint& shared_router_width [[buffer(10)]],
    uint idx [[thread_position_in_grid]]) {
    uint total = rows * width;
    if (idx >= total) { return; }
    uint row = idx / width;
    uint col = idx - row * width;
    uint route_base = row * active_experts;
    float moe = 0.0f;
    for (uint active = 0; active < active_experts; ++active) {
        uint route = route_base + active;
        uint grouped = grouped_output_indices[route];
        moe += weights[route] * grouped_expert_outputs[grouped * width + col];
    }
    float shared_weight = 0.0f;
    if (shared_router_width > 0) {
        float route = shared_router[row * shared_router_width];
        shared_weight = 1.0f / (1.0f + exp(-route));
    }
    hidden[idx] = residual[idx] + moe + shared_weight * shared[idx];
}

kernel void fill_zero(
    device float* output [[buffer(0)]],
    constant uint& width [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= width) { return; }
    output[idx] = 0.0f;
}

kernel void topk_vocab(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* values [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    device const uint* allowed_tokens [[buffer(5)]],
    constant uint& use_allowed_tokens [[buffer(6)]],
    uint slot [[thread_position_in_grid]]) {
    if (slot != 0) { return; }
    uint limit = min(top_k, vocab);
    for (uint out = 0; out < limit; ++out) {
        float best = -INFINITY;
        uint best_i = 0;
        bool found = false;
        for (uint i = 0; i < vocab; ++i) {
            if (use_allowed_tokens != 0
                && ((allowed_tokens[i >> 5] >> (i & 31)) & 1u) == 0) {
                continue;
            }
            float raw_value = logits[i];
            float value = isfinite(raw_value) ? raw_value : -INFINITY;
            bool already_used = false;
            for (uint prev = 0; prev < out; ++prev) {
                already_used = already_used || (indices[prev] == i);
            }
            if (!already_used && (!found || value > best)) {
                best = value;
                best_i = i;
                found = true;
            }
        }
        indices[out] = best_i;
        values[out] = best;
    }
}

kernel void linear_conv1d_step_bf16(
    device float* conv_state [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const ushort* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& conv_dim [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= conv_dim || kernel_size == 0) {
        return;
    }
    float acc = 0.0f;
    uint w_base = idx * kernel_size;
    for (uint k = 0; k + 1 < kernel_size; ++k) {
        acc = fma(conv_state[k * conv_dim + idx], bf16_to_float(weights[w_base + k]), acc);
    }
    float inp = input[idx];
    acc = fma(inp, bf16_to_float(weights[w_base + kernel_size - 1]), acc);
    output[idx] = acc / (1.0f + exp(-acc));
    for (uint k = 0; k + 2 < kernel_size; ++k) {
        conv_state[k * conv_dim + idx] = conv_state[(k + 1) * conv_dim + idx];
    }
    if (kernel_size > 1) {
        conv_state[(kernel_size - 2) * conv_dim + idx] = inp;
    }
}

kernel void linear_conv1d_step_f16(
    device float* conv_state [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const half* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& conv_dim [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= conv_dim || kernel_size == 0) {
        return;
    }
    float acc = 0.0f;
    uint w_base = idx * kernel_size;
    for (uint k = 0; k + 1 < kernel_size; ++k) {
        acc = fma(conv_state[k * conv_dim + idx], float(weights[w_base + k]), acc);
    }
    float inp = input[idx];
    acc = fma(inp, float(weights[w_base + kernel_size - 1]), acc);
    output[idx] = acc / (1.0f + exp(-acc));
    for (uint k = 0; k + 2 < kernel_size; ++k) {
        conv_state[k * conv_dim + idx] = conv_state[(k + 1) * conv_dim + idx];
    }
    if (kernel_size > 1) {
        conv_state[(kernel_size - 2) * conv_dim + idx] = inp;
    }
}

kernel void linear_conv1d_step_f32(
    device float* conv_state [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& conv_dim [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= conv_dim || kernel_size == 0) {
        return;
    }
    float acc = 0.0f;
    uint w_base = idx * kernel_size;
    for (uint k = 0; k + 1 < kernel_size; ++k) {
        acc = fma(conv_state[k * conv_dim + idx], weights[w_base + k], acc);
    }
    float inp = input[idx];
    acc = fma(inp, weights[w_base + kernel_size - 1], acc);
    output[idx] = acc / (1.0f + exp(-acc));
    for (uint k = 0; k + 2 < kernel_size; ++k) {
        conv_state[k * conv_dim + idx] = conv_state[(k + 1) * conv_dim + idx];
    }
    if (kernel_size > 1) {
        conv_state[(kernel_size - 2) * conv_dim + idx] = inp;
    }
}

kernel void linear_rms_norm_qk(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant uint& key_dim [[buffer(2)]],
    constant float& inv_scale [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * key_dim;
    threadgroup float partial[256];
    float qval = (tid < key_dim) ? q[base + tid] : 0.0f;
    partial[tid] = qval * qval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < key_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(key_dim, 1u)) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < key_dim) {
        q[base + tid] = qval * partial[0] * inv_scale * inv_scale;
    }

    float kval = (tid < key_dim) ? k[base + tid] : 0.0f;
    partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < key_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(key_dim, 1u)) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < key_dim) {
        k[base + tid] = kval * partial[0] * inv_scale;
    }
}

kernel void linear_compute_decay_beta_bf16(
    device const float* alpha [[buffer(0)]],
    device const float* beta [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const ushort* dt_bias [[buffer(3)]],
    device float* g_decay [[buffer(4)]],
    device float* beta_gate [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= heads) {
        return;
    }
    float softplus_value = log(1.0f + exp(alpha[idx] + bf16_to_float(dt_bias[idx])));
    g_decay[idx] = exp(-exp(a_log[idx]) * softplus_value);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta[idx]));
}

kernel void linear_compute_decay_beta_f16(
    device const float* alpha [[buffer(0)]],
    device const float* beta [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const half* dt_bias [[buffer(3)]],
    device float* g_decay [[buffer(4)]],
    device float* beta_gate [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= heads) {
        return;
    }
    float softplus_value = log(1.0f + exp(alpha[idx] + float(dt_bias[idx])));
    g_decay[idx] = exp(-exp(a_log[idx]) * softplus_value);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta[idx]));
}

kernel void linear_compute_decay_beta_f32(
    device const float* alpha [[buffer(0)]],
    device const float* beta [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device float* g_decay [[buffer(4)]],
    device float* beta_gate [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= heads) {
        return;
    }
    float softplus_value = log(1.0f + exp(alpha[idx] + dt_bias[idx]));
    g_decay[idx] = exp(-exp(a_log[idx]) * softplus_value);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta[idx]));
}

kernel void linear_gated_delta_step(
    device float* state [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* g_decay [[buffer(4)]],
    device const float* beta_gate [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& key_dim [[buffer(7)]],
    constant uint& value_dim [[buffer(8)]],
    constant uint& k_heads_per_v [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]) {
    if (vi >= value_dim || key_dim > 256) {
        return;
    }
    uint key_head = head / max(k_heads_per_v, 1u);
    uint state_base = head * value_dim * key_dim + vi * key_dim;
    uint key_base = key_head * key_dim;
    uint value_base = head * value_dim;
    float decay = g_decay[head];
    float beta = beta_gate[head];
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < key_dim; ++ki) {
        float s = state[state_base + ki] * decay;
        state[state_base + ki] = s;
        kv_mem = fma(s, k[key_base + ki], kv_mem);
    }
    float delta = (v[value_base + vi] - kv_mem) * beta;
    for (uint ki = 0; ki < key_dim; ++ki) {
        state[state_base + ki] = fma(k[key_base + ki], delta, state[state_base + ki]);
    }
    float out_value = 0.0f;
    for (uint ki = 0; ki < key_dim; ++ki) {
        out_value = fma(state[state_base + ki], q[key_base + ki], out_value);
    }
    output[value_base + vi] = out_value;
}

kernel void linear_gated_rms_norm_bf16(
    device const float* values [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const ushort* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * value_dim;
    threadgroup float partial[256];
    float value = (tid < value_dim) ? values[base + tid] : 0.0f;
    partial[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < value_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(value_dim, 1u)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < value_dim) {
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));
        output[base + tid] = value * partial[0] * gate * bf16_to_float(weight[tid]);
    }
}

kernel void linear_gated_rms_norm_f16(
    device const float* values [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * value_dim;
    threadgroup float partial[256];
    float value = (tid < value_dim) ? values[base + tid] : 0.0f;
    partial[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < value_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(value_dim, 1u)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < value_dim) {
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));
        output[base + tid] = value * partial[0] * gate * float(weight[tid]);
    }
}

kernel void linear_gated_rms_norm_f32(
    device const float* values [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint base = head * value_dim;
    threadgroup float partial[256];
    float value = (tid < value_dim) ? values[base + tid] : 0.0f;
    partial[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < value_dim; ++i) {
            sum += partial[i];
        }
        partial[0] = rsqrt(sum / float(max(value_dim, 1u)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < value_dim) {
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));
        output[base + tid] = value * partial[0] * gate * weight[tid];
    }
}
