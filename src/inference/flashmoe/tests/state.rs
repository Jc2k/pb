use super::*;

fn recurrent_session_snapshot() -> FlashMoeLinearAttentionSessionSnapshot {
    FlashMoeLinearAttentionSessionSnapshot::new(vec![Some(
        FlashMoeLinearAttentionLayerSnapshot::new(0, vec![1.0, 2.0], vec![3.0, 4.0, 5.0], 2, 2)
            .unwrap(),
    )])
    .unwrap()
}

#[test]
fn reusable_session_prefix_requires_complete_cached_prefix() {
    assert_eq!(
        reusable_session_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]),
        Some(3)
    );
    assert_eq!(reusable_session_prefix_len(&[1, 2, 3], &[1, 2, 9]), None);
    assert_eq!(reusable_session_prefix_len(&[1, 2, 3], &[1, 2]), None);
}

#[test]
fn reusable_session_entry_moves_state_only_for_matching_prefix() {
    let mut sessions = BTreeMap::new();
    sessions.insert(
        "chat".to_string(),
        FlashMoeSessionState::new(vec![10, 20], "kv-state", vec![1.0, 2.0]),
    );

    assert!(take_reusable_session_cache_entry(&mut sessions, "chat", &[10, 99, 30]).is_none());
    assert!(sessions.contains_key("chat"));

    let (prefix_len, state) =
        take_reusable_session_cache_entry(&mut sessions, "chat", &[10, 20, 30]).unwrap();
    assert_eq!(prefix_len, 2);
    assert_eq!(state.kv_cache, "kv-state");
    assert_eq!(state.last_hidden, vec![1.0, 2.0]);
    assert!(sessions.is_empty());
}

#[test]
fn kv_cache_snapshot_shares_existing_entries_and_grows_independently() {
    let mut cache = KvCache::new(1, 2);
    cache
        .record_kv(1, 0, vec![1.0, 1.5], vec![2.0, 2.5])
        .unwrap();

    let mut snapshot = cache.shallow_snapshot();
    let (cache_key, cache_value) = cache.kv[0][1].as_ref().unwrap();
    let (snapshot_key, snapshot_value) = snapshot.kv[0][1].as_ref().unwrap();
    assert!(Arc::ptr_eq(cache_key, snapshot_key));
    assert!(Arc::ptr_eq(cache_value, snapshot_value));

    snapshot.resize_capacity(3);
    snapshot
        .record_kv(2, 0, vec![3.0, 3.5], vec![4.0, 4.5])
        .unwrap();
    assert_eq!(cache.capacity, 2);
    assert_eq!(snapshot.capacity, 3);
    assert_eq!(snapshot.keys_values(2, 0).unwrap().len(), 2);
}

#[test]
fn kv_cache_rejects_gpu_recurrent_state_without_fallback() {
    let mut cache = KvCache::new(2, 2);
    cache
        .record_recurrent_layer_state(FlashMoeRecurrentLayerState::cpu_visible(1, 0, 99))
        .unwrap();
    assert_eq!(cache.layer_states, vec![(1, 0, 99)]);

    let err = cache
        .record_recurrent_layer_state(FlashMoeRecurrentLayerState::new(
            1,
            0,
            99,
            FlashMoeStatePlacement::GpuResident,
        ))
        .unwrap_err();
    assert!(
        err.to_string().contains("requires CpuVisible placement"),
        "{err:#}"
    );
}

#[test]
fn layer_major_recurrent_trace_batch_matches_typed_scalar_records() {
    let mut scalar = KvCache::new(3, 8);
    let mut layer_major = KvCache::new(3, 8);
    for (position, value) in [41, 42, 43].into_iter().enumerate() {
        scalar.record_layer_state(position + 2, 1, value).unwrap();
    }
    layer_major
        .record_layer_state_values(2, 1, [41, 42, 43].into_iter())
        .unwrap();
    assert_eq!(layer_major.layer_states, scalar.layer_states);

    assert!(
        layer_major
            .record_layer_state_values(7, 1, [1, 2].into_iter())
            .is_err()
    );
    assert!(
        layer_major
            .record_layer_state_values(0, 3, [1].into_iter())
            .is_err()
    );
}

#[test]
fn prefill_state_digest_canonicalizes_token_and_layer_major_record_order() {
    let mut token_major = KvCache::new(2, 2);
    let mut layer_major = KvCache::new(2, 2);
    for cache in [&mut token_major, &mut layer_major] {
        cache.record_kv(0, 1, vec![1.0], vec![2.0]).unwrap();
        cache.record_kv(1, 1, vec![3.0], vec![4.0]).unwrap();
    }
    token_major.record_layer_state(0, 0, 10).unwrap();
    token_major.record_layer_state(0, 1, 11).unwrap();
    token_major.record_layer_state(1, 0, 12).unwrap();
    token_major.record_layer_state(1, 1, 13).unwrap();
    layer_major.record_layer_state(0, 0, 10).unwrap();
    layer_major.record_layer_state(1, 0, 12).unwrap();
    layer_major.record_layer_state(0, 1, 11).unwrap();
    layer_major.record_layer_state(1, 1, 13).unwrap();

    assert_eq!(
        token_major.prefill_state_sha256(),
        layer_major.prefill_state_sha256()
    );
    layer_major.layer_states[3].2 ^= 1;
    assert_ne!(
        token_major.prefill_state_sha256(),
        layer_major.prefill_state_sha256()
    );
}

#[test]
fn linear_attention_state_digest_includes_exact_float_bits() {
    let first = recurrent_session_snapshot();
    let changed = FlashMoeLinearAttentionSessionSnapshot::new(vec![Some(
        FlashMoeLinearAttentionLayerSnapshot::new(
            0,
            vec![1.0, f32::from_bits(2.0f32.to_bits() + 1)],
            vec![3.0, 4.0, 5.0],
            2,
            2,
        )
        .unwrap(),
    )])
    .unwrap();
    assert_ne!(first.state_sha256(), changed.state_sha256());
}

#[test]
fn generation_lifecycle_commits_and_reuses_state_owned_prompt_snapshot() {
    let mut sessions = FlashMoeSessionCache::default();
    let mut generation = sessions.begin_generation(Some("chat"), vec![10, 20], 2, 1);
    assert_eq!(generation.prefill_start(), 0);
    {
        let (prompt, start, kv_cache) = generation.prefill_inputs();
        assert_eq!(prompt, &[10, 20]);
        assert_eq!(start, 0);
        kv_cache
            .record_kv(0, 0, vec![1.0, 1.5], vec![2.0, 2.5])
            .unwrap();
        kv_cache
            .record_kv(1, 0, vec![3.0, 3.5], vec![4.0, 4.5])
            .unwrap();
    }
    generation.capture_prompt_cache(vec![9.0, 9.5], recurrent_session_snapshot());
    generation.record_sampled_token(30, false, false);
    assert_eq!(generation.generated, vec![30]);
    sessions.commit_generation(&mut generation).unwrap();

    let mut reused = sessions.begin_generation(Some("chat"), vec![10, 20], 1, 1);
    assert_eq!(reused.prefill_start(), 2);
    assert_eq!(reused.take_cached_last_hidden(), Some(vec![9.0, 9.5]));
    assert_eq!(
        reused.take_cached_recurrent(),
        Some(recurrent_session_snapshot())
    );
    assert_eq!(reused.kv_cache.keys_values(1, 0).unwrap().len(), 2);
    assert!(reused.generated.is_empty());
}

#[test]
fn generation_lifecycle_prefers_the_exact_generated_head_checkpoint() {
    let mut sessions = FlashMoeSessionCache::default();
    let mut generation = sessions.begin_generation(Some("chat"), vec![10, 20], 3, 1);
    generation.capture_prompt_cache(vec![2.0], recurrent_session_snapshot());
    generation.record_sampled_token(30, false, false);
    generation
        .kv_cache
        .record_kv(2, 0, vec![3.0], vec![4.0])
        .unwrap();
    generation.capture_generated_cache(1, vec![3.0], recurrent_session_snapshot());
    sessions.commit_generation(&mut generation).unwrap();

    let mut reused = sessions.begin_generation(Some("chat"), vec![10, 20, 30, 40], 1, 1);
    assert_eq!(reused.prefill_start(), 3);
    assert_eq!(reused.cache_source(), PromptCacheSource::MemorySession);
    assert_eq!(reused.take_cached_last_hidden(), None);
    assert_eq!(reused.kv_cache.keys_values(2, 0).unwrap().len(), 1);
}

#[test]
fn stable_base_checkpoint_is_shared_across_logical_sessions() {
    let mut sessions = FlashMoeSessionCache::default();
    let mut first = sessions.begin_generation_with_base(Some("first"), vec![10, 20, 30], 2, 1, 1);
    first
        .kv_cache
        .record_kv(0, 0, vec![1.0], vec![2.0])
        .unwrap();
    first
        .kv_cache
        .record_kv(1, 0, vec![3.0], vec![4.0])
        .unwrap();
    first.capture_base_cache(vec![5.0], recurrent_session_snapshot());
    first.capture_prompt_cache(vec![6.0], recurrent_session_snapshot());
    sessions.commit_generation(&mut first).unwrap();

    let mut second = sessions.begin_generation_with_base(Some("second"), vec![10, 20, 99], 2, 1, 1);
    assert_eq!(second.prefill_start(), 2);
    assert_eq!(second.cache_source(), PromptCacheSource::MemoryPrefix);
    assert_eq!(second.take_cached_last_hidden(), None);
    assert_eq!(second.kv_cache.keys_values(1, 0).unwrap().len(), 2);
}

#[test]
fn memory_session_cache_evicts_the_least_recently_used_conversation() {
    let mut sessions = FlashMoeSessionCache::default();
    for (session_id, token) in [("first", 10), ("second", 20), ("third", 30)] {
        let mut generation = sessions.begin_generation(Some(session_id), vec![token], 1, 1);
        generation.capture_prompt_cache(vec![token as f32], recurrent_session_snapshot());
        sessions.commit_generation(&mut generation).unwrap();
    }

    sessions.evict_excess_sessions(2);
    assert!(!sessions.entries.contains_key("first"));
    assert!(sessions.entries.contains_key("second"));
    assert!(sessions.entries.contains_key("third"));
    assert_eq!(
        sessions.session_order,
        VecDeque::from(["second".into(), "third".into()])
    );
}

#[test]
fn configured_memory_session_limit_is_enforced_on_commit() {
    let mut sessions = FlashMoeSessionCache::new(
        None,
        1,
        crate::config::DEFAULT_FLASHMOE_MEMORY_PROMPT_ROOT_MAX_BYTES,
    );
    for (session_id, token) in [("first", 10), ("second", 20)] {
        let mut generation = sessions.begin_generation(Some(session_id), vec![token], 1, 1);
        generation.capture_prompt_cache(vec![token as f32], recurrent_session_snapshot());
        sessions.commit_generation(&mut generation).unwrap();
    }

    assert!(!sessions.entries.contains_key("first"));
    assert!(sessions.entries.contains_key("second"));
    assert_eq!(sessions.session_order, VecDeque::from(["second".into()]));
}

#[test]
fn generation_lifecycle_evicts_a_nonmatching_session_before_fresh_prefill() {
    let mut sessions = FlashMoeSessionCache::default();
    let mut generation = sessions.begin_generation(Some("chat"), vec![10, 20], 1, 1);
    generation.capture_prompt_cache(vec![9.0, 9.5], recurrent_session_snapshot());
    sessions.commit_generation(&mut generation).unwrap();
    assert_eq!(sessions.entries.len(), 1);

    let fresh = sessions.begin_generation(Some("chat"), vec![30, 40], 1, 1);

    assert_eq!(fresh.prefill_start(), 0);
    assert!(fresh.cached_last_hidden.is_none());
    assert!(fresh.cached_recurrent.is_none());
    assert!(sessions.entries.is_empty());
}

#[test]
fn generation_lifecycle_owns_decode_position_and_stop_state() {
    let mut sessions = FlashMoeSessionCache::default();
    let mut generation = sessions.begin_generation(None, vec![10, 20], 2, 1);
    assert!(generation.should_sample_first());
    generation.record_sampled_token(30, false, false);
    assert!(generation.should_decode());
    let (prompt, generated, _, position) = generation.decode_inputs().unwrap();
    assert_eq!(prompt, &[10, 20]);
    assert_eq!(generated, &[30]);
    assert_eq!(position, 2);

    generation.record_sampled_token(0, true, false);
    assert!(!generation.should_decode());
    assert_eq!(generation.into_generated(), vec![30]);
}

#[test]
fn generation_lifecycle_keeps_the_token_that_closes_a_terminal_tool_call() {
    let mut sessions = FlashMoeSessionCache::default();
    let mut generation = sessions.begin_generation(None, vec![10, 20], 4, 1);

    generation.record_sampled_token(30, false, true);

    assert!(!generation.should_decode());
    assert!(generation.stopped_by_terminal_tool_call());
    assert_eq!(generation.into_generated(), vec![30]);
}

#[test]
fn generation_lifecycle_keeps_the_token_that_completes_json() {
    let mut sessions = FlashMoeSessionCache::default();
    let mut generation = sessions.begin_generation(None, vec![10, 20], 4, 1);

    generation.record_sampled_token(30, false, false);
    generation.stop_at_json_value();

    assert!(!generation.should_decode());
    assert!(!generation.stopped_by_terminal_tool_call());
    assert!(!generation.stopped_by_constraint_payload_limit());
    assert_eq!(generation.into_generated(), vec![30]);
}

#[test]
fn generation_lifecycle_stops_before_a_constraint_payload_limit_sentinel() {
    let mut sessions = FlashMoeSessionCache::default();
    let mut generation = sessions.begin_generation(None, vec![10, 20], 4, 1);
    generation.record_sampled_token(30, false, false);

    generation.stop_at_constraint_payload_limit();

    assert!(!generation.should_decode());
    assert!(generation.stopped_by_constraint_payload_limit());
    assert_eq!(generation.into_generated(), vec![30]);
}

#[test]
fn recurrent_session_snapshot_requires_declared_layer_shape_and_order() {
    let snapshot = recurrent_session_snapshot();
    let layer = snapshot.layer(0).unwrap();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(layer.state().layer(), 0);
    assert_eq!(
        layer.state().placement(),
        FlashMoeStatePlacement::CpuVisible
    );
    assert_eq!(layer.conv_state(), &[1.0, 2.0]);
    assert_eq!(layer.ssm_state(), &[3.0, 4.0, 5.0]);

    let empty =
        FlashMoeLinearAttentionLayerSnapshot::new(0, Vec::new(), vec![1.0], 1, 1).unwrap_err();
    assert!(
        empty
            .to_string()
            .contains("not declared CPU-visible graph state"),
        "{empty:#}"
    );

    let misplaced = FlashMoeLinearAttentionSessionSnapshot::new(vec![
        None,
        Some(FlashMoeLinearAttentionLayerSnapshot::new(0, vec![1.0], vec![2.0], 1, 1).unwrap()),
    ])
    .unwrap_err();
    assert!(
        misplaced.to_string().contains("layer 1 does not match"),
        "{misplaced:#}"
    );
}

#[test]
fn stable_session_cache_tokens_keep_prompt_only() {
    assert_eq!(stable_session_cache_tokens(&[4, 5, 6]), vec![4, 5, 6]);
}

#[test]
fn cpu_hidden_buffer_declares_role_and_cpu_visibility() {
    let mut hidden = FlashMoeCpuBuffer::hidden(vec![1.0, 2.0]);
    assert_eq!(hidden.role(), FlashMoeStateBufferRole::Hidden);
    assert_eq!(hidden.placement(), FlashMoeStatePlacement::CpuVisible);
    assert_eq!(&hidden[..], &[1.0, 2.0]);

    hidden[1] = 3.0;
    assert_eq!(hidden.clone_values(), vec![1.0, 3.0]);
    assert_eq!(hidden.into_values(), vec![1.0, 3.0]);
}

#[test]
fn cpu_buffers_cover_normed_residual_and_next_layer_transitions() {
    let residual = FlashMoeCpuBuffer::residual(vec![0.5, -1.0]);
    assert_eq!(residual.role(), FlashMoeStateBufferRole::Residual);
    assert!(residual.is_declared_graph_state());

    let next_normed = FlashMoeCpuBuffer::next_layer_normed(vec![2.0, 4.0]);
    assert_eq!(next_normed.role(), FlashMoeStateBufferRole::NextLayerNormed);
    let normed = next_normed.into_role(FlashMoeStateBufferRole::Normed);
    assert_eq!(normed.role(), FlashMoeStateBufferRole::Normed);
    assert_eq!(&normed[..], &[2.0, 4.0]);
}

#[test]
fn gpu_buffer_descriptor_declares_role_length_and_residency() {
    let hidden = FlashMoeGpuBufferDescriptor::hidden(4096);
    assert_eq!(hidden.role(), FlashMoeStateBufferRole::Hidden);
    assert_eq!(hidden.len(), 4096);
    assert_eq!(hidden.placement(), FlashMoeStatePlacement::GpuResident);
    assert!(hidden.is_declared_graph_state());

    let next_normed = FlashMoeGpuBufferDescriptor::next_layer_normed(4096);
    assert_eq!(next_normed.role(), FlashMoeStateBufferRole::NextLayerNormed);
    assert_eq!(next_normed.placement(), FlashMoeStatePlacement::GpuResident);
}

#[test]
fn gpu_buffer_descriptor_rejects_zero_length_graph_state() {
    let hidden = FlashMoeGpuBufferDescriptor::hidden(0);

    assert_eq!(hidden.len(), 0);
    assert!(!hidden.is_declared_graph_state());
}

#[test]
fn gpu_matrix_descriptor_preserves_rows_columns_and_checked_bytes() {
    let hidden = FlashMoeGpuMatrixDescriptor::hidden(17, 4096).unwrap();

    assert_eq!(hidden.role(), FlashMoeStateBufferRole::Hidden);
    assert_eq!(hidden.rows(), 17);
    assert_eq!(hidden.cols(), 4096);
    assert_eq!(hidden.values(), 17 * 4096);
    assert_eq!(hidden.bytes().unwrap(), 17 * 4096 * 4);
    assert_eq!(hidden.placement(), FlashMoeStatePlacement::GpuResident);
    assert!(hidden.is_declared_graph_state());
}

#[test]
fn gpu_matrix_descriptor_rejects_empty_or_overflowing_geometry() {
    assert!(FlashMoeGpuMatrixDescriptor::hidden(0, 4096).is_err());
    assert!(FlashMoeGpuMatrixDescriptor::normed(17, 0).is_err());
    assert!(
        FlashMoeGpuMatrixDescriptor::new(FlashMoeStateBufferRole::RouterScores, usize::MAX, 2,)
            .is_err()
    );
}

#[test]
fn post_attention_prep_state_names_gpu_residual_normed_and_routes() {
    let state = FlashMoePostAttentionPrepState::new(12, 4096, 128, 4);

    assert_eq!(state.width(), 4096);
    assert_eq!(state.active_experts(), 4);
    assert_eq!(state.routing().layer(), 12);
    assert_eq!(state.routing().experts(), 128);
    assert_eq!(
        state.routing().source(),
        FlashMoeRoutingOutputSource::FusedMetalPostAttentionPrepCpuTopK
    );
    assert_eq!(state.residual().role(), FlashMoeStateBufferRole::Residual);
    assert_eq!(state.normed().role(), FlashMoeStateBufferRole::Normed);
    assert_eq!(
        state.residual().placement(),
        FlashMoeStatePlacement::GpuResident
    );
    assert!(state.is_declared_graph_state());
}

#[test]
fn post_attention_prep_state_rejects_empty_width_or_routes() {
    assert!(!FlashMoePostAttentionPrepState::new(12, 0, 128, 4).is_declared_graph_state());
    assert!(!FlashMoePostAttentionPrepState::new(12, 4096, 128, 0).is_declared_graph_state());
    assert!(!FlashMoePostAttentionPrepState::new(12, 4096, 2, 4).is_declared_graph_state());
}

#[test]
fn routing_output_state_declares_cpu_visible_scores_and_topk() {
    let scores = FlashMoeRoutingOutputState::cpu_router_scores(2, 8, 4);
    assert_eq!(scores.role(), FlashMoeStateBufferRole::RouterScores);
    assert_eq!(scores.len(), 8);
    assert_eq!(scores.placement(), FlashMoeStatePlacement::CpuVisible);
    assert!(scores.is_declared_graph_state());

    let topk = FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(2, 8, 4);
    assert_eq!(topk.role(), FlashMoeStateBufferRole::RoutingTopK);
    assert_eq!(topk.len(), 4);
    assert_eq!(topk.active_experts(), 4);
    assert!(topk.is_declared_graph_state());
}

#[test]
fn routing_output_state_rejects_empty_or_oversized_active_routes() {
    assert!(!FlashMoeRoutingOutputState::cpu_router_scores(2, 0, 0).is_declared_graph_state());
    assert!(
        !FlashMoeRoutingOutputState::fused_metal_post_attention_cpu_topk(2, 2, 4)
            .is_declared_graph_state()
    );
}

#[test]
fn cmd2_input_state_declares_attention_and_residual_placements() {
    let cpu = FlashMoeCmd2InputState::new(
        3,
        2048,
        FlashMoeStatePlacement::CpuVisible,
        4096,
        FlashMoeStatePlacement::CpuVisible,
    );
    assert_eq!(cpu.layer(), 3);
    assert_eq!(
        cpu.attention().role(),
        FlashMoeStateBufferRole::AttentionValues
    );
    assert_eq!(cpu.residual().role(), FlashMoeStateBufferRole::Residual);
    assert_eq!(cpu.attention().len(), 2048);
    assert_eq!(cpu.residual().len(), 4096);
    assert_eq!(
        cpu.attention().placement(),
        FlashMoeStatePlacement::CpuVisible
    );
    assert!(cpu.is_declared_graph_state());

    let gpu = FlashMoeCmd2InputState::new(
        3,
        2048,
        FlashMoeStatePlacement::GpuResident,
        4096,
        FlashMoeStatePlacement::GpuResident,
    );
    assert_eq!(
        gpu.attention().placement(),
        FlashMoeStatePlacement::GpuResident
    );
    assert_eq!(
        gpu.residual().placement(),
        FlashMoeStatePlacement::GpuResident
    );
    assert!(gpu.is_declared_graph_state());
}

#[test]
fn cmd2_input_state_rejects_empty_attention_or_residual() {
    assert!(
        !FlashMoeCmd2InputState::new(
            3,
            0,
            FlashMoeStatePlacement::CpuVisible,
            4096,
            FlashMoeStatePlacement::CpuVisible,
        )
        .is_declared_graph_state()
    );
    assert!(
        !FlashMoeCmd2InputState::new(
            3,
            2048,
            FlashMoeStatePlacement::CpuVisible,
            0,
            FlashMoeStatePlacement::CpuVisible,
        )
        .is_declared_graph_state()
    );
}

#[test]
fn cmd3_output_state_declares_gpu_hidden_and_optional_next_normed() {
    let output = FlashMoeCmd3OutputState::gpu_resident(4096, true);

    assert_eq!(output.width(), 4096);
    assert_eq!(output.hidden().role(), FlashMoeStateBufferRole::Hidden);
    assert_eq!(
        output.hidden().placement(),
        FlashMoeStatePlacement::GpuResident
    );
    let next_normed = output.next_normed().unwrap();
    assert_eq!(next_normed.role(), FlashMoeStateBufferRole::NextLayerNormed);
    assert_eq!(next_normed.len(), 4096);
    assert!(output.has_next_normed());
    assert!(output.is_declared_graph_state());

    let hidden_only = FlashMoeCmd3OutputState::gpu_resident(4096, false);
    assert!(!hidden_only.has_next_normed());
    assert!(hidden_only.next_normed().is_none());
    assert!(hidden_only.is_declared_graph_state());
}

#[test]
fn cmd3_output_state_rejects_zero_width() {
    let output = FlashMoeCmd3OutputState::gpu_resident(0, true);

    assert_eq!(output.width(), 0);
    assert!(!output.is_declared_graph_state());
}

#[test]
fn cmd3_input_state_declares_cpu_or_gpu_normed_residual_pair() {
    let cpu = FlashMoeCmd3InputState::cpu_normed_residual(5, 4096, 4096);
    assert_eq!(cpu.layer(), 5);
    assert_eq!(cpu.width(), 4096);
    assert_eq!(cpu.residual().role(), FlashMoeStateBufferRole::Residual);
    assert_eq!(cpu.normed().role(), FlashMoeStateBufferRole::Normed);
    assert_eq!(cpu.placement(), FlashMoeStatePlacement::CpuVisible);
    assert!(cpu.is_declared_graph_state());

    let prep = FlashMoePostAttentionPrepState::new(5, 4096, 128, 4);
    let gpu = FlashMoeCmd3InputState::metal_post_attention_prep(5, prep);
    assert_eq!(gpu.width(), 4096);
    assert_eq!(
        gpu.residual().placement(),
        FlashMoeStatePlacement::GpuResident
    );
    assert_eq!(
        gpu.normed().placement(),
        FlashMoeStatePlacement::GpuResident
    );
    assert_eq!(gpu.placement(), FlashMoeStatePlacement::GpuResident);
    assert!(gpu.is_declared_graph_state());
}

#[test]
fn cmd3_input_state_rejects_zero_or_mismatched_buffers() {
    assert!(!FlashMoeCmd3InputState::cpu_normed_residual(5, 0, 0).is_declared_graph_state());
    assert!(!FlashMoeCmd3InputState::cpu_normed_residual(5, 4096, 2048).is_declared_graph_state());
    let prep = FlashMoePostAttentionPrepState::new(5, 0, 128, 4);
    assert!(!FlashMoeCmd3InputState::metal_post_attention_prep(5, prep).is_declared_graph_state());
}

#[test]
fn cmd1_input_state_declares_cpu_normed_or_gpu_next_normed() {
    let cpu = FlashMoeCmd1InputState::cpu_normed(12, 4096);
    assert_eq!(cpu.layer(), 12);
    assert_eq!(cpu.role(), FlashMoeStateBufferRole::Normed);
    assert_eq!(cpu.len(), 4096);
    assert_eq!(cpu.placement(), FlashMoeStatePlacement::CpuVisible);
    assert!(cpu.is_declared_graph_state());

    let gpu = FlashMoeCmd1InputState::gpu_next_layer_normed(
        12,
        FlashMoeGpuBufferDescriptor::next_layer_normed(4096),
    );
    assert_eq!(gpu.layer(), 12);
    assert_eq!(gpu.role(), FlashMoeStateBufferRole::NextLayerNormed);
    assert_eq!(gpu.len(), 4096);
    assert_eq!(gpu.placement(), FlashMoeStatePlacement::GpuResident);
    assert!(gpu.is_declared_graph_state());

    assert!(!FlashMoeCmd1InputState::cpu_normed(12, 0).is_declared_graph_state());
    assert!(
        !FlashMoeCmd1InputState::gpu_next_layer_normed(
            12,
            FlashMoeGpuBufferDescriptor::hidden(4096),
        )
        .is_declared_graph_state()
    );
}

#[test]
fn recurrent_state_mixes_active_experts_with_wrapping_math() {
    let mut state = FlashMoeRecurrentState::new(u64::MAX - 4);
    state.mix_active_expert(3, 1.0);
    let expected = (u64::MAX - 4).wrapping_add(3_u64.wrapping_mul(1.0f32.to_bits() as u64));
    assert_eq!(state.value(), expected);
}

#[test]
fn recurrent_state_uses_nonzero_multiplier_for_zero_weight_bits() {
    let mut state = FlashMoeRecurrentState::new(10);
    state.mix_active_expert(7, 0.0);
    assert_eq!(state.value(), 17);
}

#[test]
fn recurrent_state_records_the_same_layer_fingerprint_as_token_state() {
    let mut recurrent = FlashMoeRecurrentState::new(41);
    let mut token = FlashMoeTokenState::new(vec![1.0, 2.0], 41);
    for (hash, weight) in [(3, 0.25), (u64::MAX - 7, -0.0), (19, 1.5)] {
        recurrent.mix_active_expert(hash, weight);
        token.mix_active_expert(hash, weight);
    }

    assert_eq!(
        recurrent.layer_state_record(17, 9),
        token.layer_state_record(17, 9)
    );
}

#[test]
fn token_state_owns_hidden_next_normed_and_recurrent_values() {
    let mut state = FlashMoeTokenState::new(vec![1.0, 2.0], 10);
    assert_eq!(state.hidden().role(), FlashMoeStateBufferRole::Hidden);
    assert_eq!(&state.hidden()[..], &[1.0, 2.0]);

    let residual = state.residual_snapshot();
    assert_eq!(residual.role(), FlashMoeStateBufferRole::Residual);
    assert_eq!(&residual[..], &[1.0, 2.0]);

    state.set_next_layer_normed(Some(vec![3.0, 4.0]));
    let normed = state.take_next_layer_normed_as_normed().unwrap();
    assert_eq!(normed.role(), FlashMoeStateBufferRole::Normed);
    assert_eq!(&normed[..], &[3.0, 4.0]);
    assert!(state.take_next_layer_normed_as_normed().is_none());

    state.apply_expert_phase_output(FlashMoeExpertPhaseOutput::new(
        vec![8.0, 9.0],
        Some(vec![10.0, 11.0]),
    ));
    assert_eq!(&state.hidden()[..], &[8.0, 9.0]);
    let normed = state.take_next_layer_normed_as_normed().unwrap();
    assert_eq!(normed.role(), FlashMoeStateBufferRole::Normed);
    assert_eq!(&normed[..], &[10.0, 11.0]);

    state.apply_expert_phase_hidden_only(FlashMoeExpertPhaseOutput::new(
        vec![12.0],
        Some(vec![13.0]),
    ));
    assert_eq!(&state.hidden()[..], &[12.0]);
    assert!(state.take_next_layer_normed_as_normed().is_none());

    state.mix_active_expert(7, 0.0);
    assert_eq!(state.recurrent_value(), 17);
    assert_eq!(
        state.layer_state_record(5, 2),
        FlashMoeLayerStateRecord {
            position: 5,
            layer: 2,
            recurrent_value: 17
        }
    );
    state.replace_hidden(vec![5.0]);
    assert_eq!(state.into_hidden_values(), vec![5.0]);
}

#[test]
fn token_state_requires_declared_expert_output_for_scheduled_application() {
    let mut state = FlashMoeTokenState::new(vec![1.0, 2.0], 10);
    let raw_err = state
        .apply_declared_expert_phase(
            FlashMoeExpertPhaseOutput::new(vec![3.0, 4.0], Some(vec![5.0, 6.0])),
            FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
        )
        .unwrap_err();
    assert!(
        raw_err
            .to_string()
            .contains("refused undeclared expert phase output"),
        "{raw_err:#}"
    );

    let declared = FlashMoeExpertPhaseOutput::new(vec![3.0, 4.0], Some(vec![5.0, 6.0]))
        .with_declared_cmd3_output(FlashMoeCmd3OutputState::gpu_resident(2, true));
    state
        .apply_declared_expert_phase(
            declared,
            FlashMoeExpertPhaseApplication::HiddenAndNextNormed,
        )
        .unwrap();
    assert_eq!(&state.hidden()[..], &[3.0, 4.0]);
    let normed = state.take_next_layer_normed_as_normed().unwrap();
    assert_eq!(&normed[..], &[5.0, 6.0]);

    let declared_hidden_only =
        FlashMoeExpertPhaseOutput::new(vec![7.0, 8.0], Some(vec![9.0, 10.0]))
            .with_declared_cmd3_output(FlashMoeCmd3OutputState::gpu_resident(2, true));
    state
        .apply_declared_expert_phase(
            declared_hidden_only,
            FlashMoeExpertPhaseApplication::HiddenOnly,
        )
        .unwrap();
    assert_eq!(&state.hidden()[..], &[7.0, 8.0]);
    assert!(state.take_next_layer_normed_as_normed().is_none());
}

#[test]
fn recurrent_layer_state_declares_cpu_visible_layer_transition() {
    let record = FlashMoeLayerStateRecord {
        position: 5,
        layer: 2,
        recurrent_value: 17,
    };
    let state = record.state(FlashMoeStatePlacement::CpuVisible);

    assert_eq!(state, FlashMoeRecurrentLayerState::cpu_visible(5, 2, 17));
    assert_eq!(state.position(), 5);
    assert_eq!(state.layer(), 2);
    assert_eq!(state.value(), 17);
    assert_eq!(state.role(), FlashMoeStateBufferRole::Recurrent);
    assert_eq!(state.placement(), FlashMoeStatePlacement::CpuVisible);
    assert!(state.is_declared_graph_state());
}

#[test]
fn linear_attention_cache_state_declares_lengths_and_placement() {
    let cpu = FlashMoeLinearAttentionCacheState::cpu_visible(3, 8, 16, 4, 6);
    assert_eq!(cpu.layer(), 3);
    assert_eq!(cpu.conv_state_len(), 8);
    assert_eq!(cpu.ssm_state_len(), 16);
    assert_eq!(cpu.conv_output_len(), 4);
    assert_eq!(cpu.output_len(), 6);
    assert_eq!(cpu.role(), FlashMoeStateBufferRole::Recurrent);
    assert_eq!(cpu.placement(), FlashMoeStatePlacement::CpuVisible);
    assert!(cpu.is_declared_graph_state());

    let gpu = FlashMoeLinearAttentionCacheState::gpu_resident(3, 8, 16, 4, 6);
    assert_eq!(gpu.placement(), FlashMoeStatePlacement::GpuResident);
    assert!(gpu.is_declared_graph_state());

    assert!(
        !FlashMoeLinearAttentionCacheState::cpu_visible(3, 0, 16, 4, 6).is_declared_graph_state()
    );
    assert!(
        !FlashMoeLinearAttentionCacheState::gpu_resident(3, 8, 0, 4, 6).is_declared_graph_state()
    );
    assert!(
        !FlashMoeLinearAttentionCacheState::cpu_visible(3, 8, 16, 0, 6).is_declared_graph_state()
    );
    assert!(
        !FlashMoeLinearAttentionCacheState::gpu_resident(3, 8, 16, 4, 0).is_declared_graph_state()
    );
}

#[test]
fn linear_attention_state_owns_recurrent_buffers_for_declared_shape() {
    let shape = FlashMoeLinearAttentionCacheShape::new(8, 16, 4, 6, 2);
    assert!(shape.is_declared_graph_shape());

    let mut state = FlashMoeLinearAttentionState::new(shape);
    assert_eq!(state.conv_state.len(), 8);
    assert_eq!(state.ssm_state.len(), 16);
    assert_eq!(state.conv_out.len(), 4);
    assert_eq!(state.out_values.len(), 6);
    assert_eq!(state.kv_mem.len(), 2);
    assert_eq!(state.delta.len(), 2);
    assert!(state.matches_shape(shape));
    assert!(!state.matches_shape(FlashMoeLinearAttentionCacheShape::new(8, 16, 4, 6, 3)));

    state.conv_state[0] = 1.5;
    assert_eq!(state.conv_state[0], 1.5);
    assert_eq!(
        FlashMoeLinearAttentionState::expected_state(3, shape, FlashMoeStatePlacement::CpuVisible),
        FlashMoeLinearAttentionCacheState::cpu_visible(3, 8, 16, 4, 6)
    );
    assert_eq!(
        state.state(3, FlashMoeStatePlacement::GpuResident),
        FlashMoeLinearAttentionCacheState::gpu_resident(3, 8, 16, 4, 6)
    );
}

#[test]
fn expert_phase_output_owns_hidden_and_optional_next_normed_transition() {
    let output = FlashMoeExpertPhaseOutput::new(vec![1.0, 2.0], Some(vec![0.5, 1.5]));

    let (hidden, next_normed) = output.into_hidden_and_next_normed();

    assert_eq!(hidden, vec![1.0, 2.0]);
    assert_eq!(next_normed, Some(vec![0.5, 1.5]));
}

#[test]
fn generated_token_record_names_position_and_token() {
    let record = FlashMoeGeneratedTokenRecord::new(12, 99);
    assert_eq!(record.position(), 12);
    assert_eq!(record.token(), 99);
}

#[test]
fn prompt_token_record_names_position_and_token() {
    let record = FlashMoePromptTokenRecord::new(3, 42);
    assert_eq!(record.position(), 3);
    assert_eq!(record.token(), 42);
}

#[test]
fn full_attention_kv_record_owns_position_layer_and_values() {
    let record = FlashMoeFullAttentionKvRecord::new(5, 2, vec![1.0, 1.5], vec![2.0, 2.5]);
    assert_eq!(record.position(), 5);
    assert_eq!(record.layer(), 2);
    assert_eq!(record.key(), &[1.0, 1.5]);
    assert_eq!(record.value(), &[2.0, 2.5]);
    assert_eq!(
        record.state(FlashMoeStatePlacement::CpuVisible),
        FlashMoeFullAttentionKvState::cpu_visible(5, 2, 2, 2)
    );
    let (key, value) = record.into_key_value();
    assert_eq!(key, vec![1.0, 1.5]);
    assert_eq!(value, vec![2.0, 2.5]);
}

#[test]
fn full_attention_kv_state_declares_placement_width_and_role() {
    let cpu = FlashMoeFullAttentionKvState::cpu_visible(7, 3, 4, 4);
    assert_eq!(cpu.position(), 7);
    assert_eq!(cpu.layer(), 3);
    assert_eq!(cpu.width(), 4);
    assert_eq!(cpu.key_len(), 4);
    assert_eq!(cpu.value_len(), 4);
    assert_eq!(cpu.role(), FlashMoeStateBufferRole::Kv);
    assert_eq!(cpu.placement(), FlashMoeStatePlacement::CpuVisible);
    assert!(cpu.is_declared_graph_state());

    let gpu = FlashMoeFullAttentionKvState::gpu_resident(7, 3, 4, 4);
    assert_eq!(gpu.placement(), FlashMoeStatePlacement::GpuResident);
    assert!(gpu.is_declared_graph_state());

    assert!(!FlashMoeFullAttentionKvState::cpu_visible(7, 3, 0, 0).is_declared_graph_state());
    assert!(!FlashMoeFullAttentionKvState::gpu_resident(7, 3, 4, 5).is_declared_graph_state());
}

#[test]
fn mla_kv_state_declares_distinct_latent_and_rotary_widths() {
    let state = FlashMoeMlaKvState::cpu_visible(7, 3, 512, 64);
    assert_eq!(state.position(), 7);
    assert_eq!(state.layer(), 3);
    assert_eq!(state.latent_len(), 512);
    assert_eq!(state.rotary_len(), 64);
    assert_eq!(state.role(), FlashMoeStateBufferRole::Kv);
    assert_eq!(state.placement(), FlashMoeStatePlacement::CpuVisible);
    assert!(state.is_declared_graph_state());
    assert!(!FlashMoeMlaKvState::cpu_visible(7, 3, 0, 64).is_declared_graph_state());
    assert!(!FlashMoeMlaKvState::cpu_visible(7, 3, 512, 0).is_declared_graph_state());
}
