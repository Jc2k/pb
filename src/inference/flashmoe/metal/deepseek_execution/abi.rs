use std::mem::{offset_of, size_of};

use super::{
    ArgsortArgs, ArgsortMergeArgs, AttentionArgs, AttentionMaskArgs, CompressorArgs,
    CompressorPrefillArgs, CopyArgs, EmbeddingArgs, EmbeddingBatchArgs, FlashAttentionArgs,
    FlashAttentionBlockArgs, FlashAttentionPadArgs, Fp8Args, GroupCopyArgs, HcExpandArgs,
    HcSplitNormArgs, IndexQatArgs, IndexScoresArgs, IndexedAttentionArgs, KvStoreArgs, MatvecArgs,
    MoeActivationArgs, MoeBatchMmArgs, MoeMapArgs, MoeMatvecArgs, MoeSum6Args, MulMmArgs,
    OutputCollapseArgs, RawContextBatchArgs, RawStoreBatchArgs, RmsArgs, RopeArgs, SwigluBatchArgs,
    TopkMaskArgs,
};

pub(super) fn bytes_of<T: bytemuck::NoUninit>(value: &T) -> &[u8] {
    bytemuck::bytes_of(value)
}

macro_rules! assert_abi {
    ($ty:ty, $size:expr; $($field:ident => $offset:expr),+ $(,)?) => {
        assert!(size_of::<$ty>() == $size);
        $(assert!(offset_of!($ty, $field) == $offset);)+
    };
}

// These constants mirror the matching Metal Shading Language argument
// structs. NoUninit proves every uploaded byte is initialized; checking every
// field offset catches reorderings or misplaced explicit padding even when the
// total structure size remains unchanged.
const _: () = {
    assert_abi!(MatvecArgs, 112;
        ne00 => 0, ne01 => 4, ne02 => 8, _pad0 => 12,
        nb00 => 16, nb01 => 24, nb02 => 32, nb03 => 40,
        ne10 => 48, ne11 => 52, ne12 => 56, _pad1 => 60,
        nb10 => 64, nb11 => 72, nb12 => 80, nb13 => 88,
        ne0 => 96, ne1 => 100, nr0 => 104, r2 => 108, r3 => 110,
    );
    assert_abi!(MulMmArgs, 88;
        ne00 => 0, ne02 => 4, nb01 => 8, nb02 => 16, nb03 => 24,
        ne12 => 32, _pad0 => 36, nb10 => 40, nb11 => 48, nb12 => 56,
        nb13 => 64, ne0 => 72, ne1 => 76, r2 => 80, r3 => 82, _pad1 => 84,
    );
    assert_abi!(EmbeddingBatchArgs, 12; tokens => 0, hidden => 4, hc => 8);
    assert_abi!(CopyArgs, 4; elements => 0);
    assert_abi!(SwigluBatchArgs, 8; elements => 0, clamp => 4);
    assert_abi!(RawStoreBatchArgs, 16;
        tokens => 0, raw_cap => 4, head_dim => 8, pos0 => 12,
    );
    assert_abi!(RawContextBatchArgs, 20;
        tokens => 0, prefix_raw => 4, raw_cap => 8, head_dim => 12, pos0 => 16,
    );
    assert_abi!(GroupCopyArgs, 20;
        tokens => 0, groups => 4, group => 8, group_width => 12, rank => 16,
    );
    assert_abi!(CompressorPrefillArgs, 20;
        tokens => 0, width => 4, head_dim => 8, ratio => 12, pos0 => 16,
    );
    assert_abi!(AttentionMaskArgs, 24;
        tokens => 0, raw_rows => 4, compressed => 8, window => 12, ratio => 16, pos0 => 20,
    );
    assert_abi!(FlashAttentionPadArgs, 104;
        ne11 => 0, ne_12_2 => 4, ne_12_3 => 8, _pad0 => 12,
        nb11 => 16, nb12 => 24, nb13 => 32, nb21 => 40, nb22 => 48, nb23 => 56,
        ne31 => 64, ne32 => 68, ne33 => 72, _pad1 => 76,
        nb31 => 80, nb32 => 88, nb33 => 96,
    );
    assert_abi!(FlashAttentionBlockArgs, 48;
        ne01 => 0, ne30 => 4, ne31 => 8, ne32 => 12, ne33 => 16, _pad0 => 20,
        nb31 => 24, nb32 => 32, nb33 => 40,
    );
    assert_abi!(FlashAttentionArgs, 192;
        ne01 => 0, ne02 => 4, ne03 => 8, _pad0 => 12,
        nb01 => 16, nb02 => 24, nb03 => 32,
        ne11 => 40, ne_12_2 => 44, ne_12_3 => 48, ns10 => 52,
        nb11 => 56, nb12 => 64, nb13 => 72, ns20 => 80, _pad1 => 84,
        nb21 => 88, nb22 => 96, nb23 => 104,
        ne31 => 112, ne32 => 116, ne33 => 120, _pad2 => 124,
        nb31 => 128, nb32 => 136, nb33 => 144,
        ne1 => 152, ne2 => 156, ne3 => 160, scale => 164,
        max_bias => 168, m0 => 172, m1 => 176, n_head_log2 => 180,
        logit_softcap => 184, _pad3 => 188,
    );
    assert_abi!(TopkMaskArgs, 64;
        ne00 => 0, ne01 => 8, nb00 => 16, nb01 => 24,
        ne0 => 32, ne1 => 40, nb0 => 48, nb1 => 56,
    );
    assert_abi!(MoeMapArgs, 48;
        ne02 => 0, ne10 => 4, ne11 => 8, _pad0 => 12,
        nb11 => 16, nb12 => 24, ne21 => 32, ne20 => 36, nb21 => 40,
    );
    assert_abi!(MoeBatchMmArgs, 96;
        ne00 => 0, ne02 => 4, nb01 => 8, nb02 => 16, nb03 => 24,
        ne11 => 32, _pad0 => 36, nb10 => 40, nb11 => 48, nb12 => 56,
        nb13 => 64, ne20 => 72, ne21 => 76, ne0 => 80, ne1 => 84,
        r2 => 88, r3 => 90, _pad1 => 92,
    );
    assert_abi!(MoeSum6Args, 24;
        width => 0, tokens => 4, src_token_stride => 8, dst_token_stride => 16,
    );
    assert_abi!(HcSplitNormArgs, 104;
        n_embd => 0, n_hc => 8, sinkhorn_iters => 12, n_rows => 16, mix_hc => 24,
        nb_mix1 => 32, nb_split1 => 40, nb_x0 => 48, nb_x1 => 56, nb_x2 => 64,
        nb0 => 72, nb1 => 80, nb_norm1 => 88, eps => 96, norm_eps => 100,
    );
    assert_abi!(HcExpandArgs, 152;
        n_embd => 0, n_hc => 8, n_tokens => 16,
        nb_block0 => 24, nb_block1 => 32, nb_add0 => 40, nb_add1 => 48,
        nb_res0 => 56, nb_res1 => 64, nb_res2 => 72,
        nb_post0 => 80, nb_post1 => 88, nb_comb0 => 96, nb_comb1 => 104,
        nb_comb2 => 112, nb0 => 120, nb1 => 128, nb2 => 136,
        has_add => 144, _pad0 => 148,
    );
    assert_abi!(RmsArgs, 16; width => 0, rows => 4, weighted => 8, eps => 12);
    assert_abi!(EmbeddingArgs, 12; token => 0, hidden => 4, hc => 8);
    assert_abi!(CompressorArgs, 20;
        width => 0, head_dim => 4, ratio => 8, position => 12, emit_row => 16,
    );
    assert_abi!(AttentionArgs, 48;
        n_head => 0, head_dim => 4, n_raw => 8, raw_cap => 12, raw_start => 16,
        n_comp => 20, top_k => 24, use_top_k => 28, position => 32,
        window => 36, ratio => 40, scale => 44,
    );
    assert_abi!(IndexedAttentionArgs, 112;
        n_tokens => 0, n_head => 4, n_raw => 8, raw_cap => 12, raw_start => 16,
        n_comp => 20, top_k => 24, pos0 => 28, window => 32, ratio => 36,
        comp_kv_f16 => 40, pad0 => 44, q_token_stride => 48, q_head_stride => 56,
        raw_row_stride => 64, comp_row_stride => 72, topk_token_stride => 80,
        dst_token_stride => 88, dst_head_stride => 96, scale => 104, _pad1 => 108,
    );
    assert_abi!(OutputCollapseArgs, 12; hidden => 0, eps => 4, hc_eps => 8);
    assert_abi!(KvStoreArgs, 12; head_dim => 0, n_rot => 4, raw_row => 8);
    assert_abi!(Fp8Args, 104;
        ne00 => 0, ne01 => 8, ne02 => 16, ne03 => 24,
        nb00 => 32, nb01 => 40, nb02 => 48, nb03 => 56,
        nb0 => 64, nb1 => 72, nb2 => 80, nb3 => 88, n_rot => 96, _pad0 => 100,
    );
    assert_abi!(RopeArgs, 144;
        ne00 => 0, ne01 => 8, ne02 => 16, ne03 => 24,
        nb00 => 32, nb01 => 40, nb02 => 48, nb03 => 56,
        nb0 => 64, nb1 => 72, nb2 => 80, nb3 => 88,
        n_dims => 96, mode => 100, n_ctx_orig => 104, inverse => 108,
        freq_base => 112, freq_scale => 116, ext_factor => 120, attn_factor => 124,
        beta_fast => 128, beta_slow => 132, src2 => 136, _pad0 => 137,
    );
    assert_abi!(IndexQatArgs, 16; n_rows => 0, head_dim => 4, row_stride => 8);
    assert_abi!(IndexScoresArgs, 72;
        n_comp => 0, n_tokens => 4, n_head => 8, head_dim => 12, pos0 => 16,
        ratio => 20, q_token_stride => 24, q_head_stride => 32,
        weights_token_stride => 40, index_row_stride => 48, score_token_stride => 56,
        scale => 64, _pad0 => 68,
    );
    assert_abi!(ArgsortArgs, 72;
        ne00 => 0, ne01 => 4, ne02 => 8, ne03 => 12,
        nb00 => 16, nb01 => 24, nb02 => 32, nb03 => 40,
        ne0 => 48, ne1 => 52, ne2 => 56, ne3 => 60, top_k => 64, _pad0 => 68,
    );
    assert_abi!(ArgsortMergeArgs, 88;
        ne00 => 0, ne01 => 8, ne02 => 16, ne03 => 24,
        nb00 => 32, nb01 => 40, nb02 => 48, nb03 => 56,
        ne0 => 64, ne1 => 68, ne2 => 72, ne3 => 76, top_k => 80, len => 84,
    );
    assert_abi!(MoeMatvecArgs, 120;
        nei0 => 0, nei1 => 4, nbi1 => 8, ne00 => 16, ne01 => 20, ne02 => 24,
        _pad0 => 28, nb00 => 32, nb01 => 40, nb02 => 48,
        ne10 => 56, ne11 => 60, ne12 => 64, ne13 => 68,
        nb10 => 72, nb11 => 80, nb12 => 88, ne0 => 96, ne1 => 100,
        nb1 => 104, nr0 => 112, _pad1 => 116,
    );
    assert_abi!(MoeActivationArgs, 48;
        width => 0, rows => 4, gate_row_stride => 8, up_row_stride => 16,
        mid_row_stride => 24, weight_stride => 32, write_clamped => 40, clamp_value => 44,
    );
};
