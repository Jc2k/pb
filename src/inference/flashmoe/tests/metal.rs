#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::super::capabilities::{
    FlashMoeGraphStage, FlashMoeStageCapability, FlashMoeStageImplementation,
    FlashMoeStagePlacement,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::super::experts::Q4MatvecPayload;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::super::scheduler::ScheduledRoutingTopK;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::super::weights::DenseQ4MmapMatvecProjection;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::super::weights::SharedExpertPhaseShape;
use super::*;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_working_set_limit_preserves_documented_headroom() {
    let gib = 1024 * 1024 * 1024;
    assert_eq!(default_metal_working_set_limit(10 * gib), 9 * gib);
    assert_eq!(default_metal_working_set_limit(4 * gib), 3 * gib);
    assert_eq!(default_metal_working_set_limit(gib), gib / 2);
    assert_eq!(default_metal_working_set_limit(0), usize::MAX);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_resource_ledger_balances_buffer_ownership_transitions() {
    let gib = 1024 * 1024 * 1024;
    let ledger = MetalResourceLedger {
        state: Mutex::new(MetalResourceLedgerState::new(10 * gib, 2 * gib)),
    };
    ledger.record_resident_resources(100, 200);
    ledger.register_buffer(
        0x1000usize as MetalObjcId,
        64,
        MetalTrackedBufferClass::ActiveGeneral,
    );
    ledger.register_buffer(
        0x2000usize as MetalObjcId,
        128,
        MetalTrackedBufferClass::TransientExpert,
    );
    ledger.register_buffer(
        0x3000usize as MetalObjcId,
        256,
        MetalTrackedBufferClass::ResidentExpertWrapper,
    );

    let active = ledger.snapshot();
    assert_eq!(active.active_general_buffers, 1);
    assert_eq!(active.transient_expert_buffers, 1);
    assert_eq!(active.resident_expert_wrapper_buffers, 1);
    assert_eq!(active.resident_expert_wrapper_bytes, 256);
    assert_eq!(active.ledger_live_bytes, 748);
    assert_eq!(active.driver_high_water_bytes, 2 * gib);

    ledger.transition_buffer(0x1000usize as MetalObjcId, MetalTrackedBufferClass::Pooled);
    ledger.release_buffer(0x2000usize as MetalObjcId);
    let pooled = ledger.snapshot();
    assert_eq!(pooled.active_general_buffers, 0);
    assert_eq!(pooled.transient_expert_buffers, 0);
    assert_eq!(pooled.pooled_buffers, 1);
    assert_eq!(pooled.ledger_live_bytes, 620);

    ledger.release_buffer(0x1000usize as MetalObjcId);
    ledger.release_buffer(0x3000usize as MetalObjcId);
    ledger.record_resident_resources(0, 0);
    let released = ledger.snapshot();
    assert_eq!(released.ledger_live_bytes, 0);
    assert_eq!(released.pooled_buffers, 0);
    assert_eq!(released.resident_expert_wrapper_buffers, 0);
    assert_eq!(released.ledger_high_water_bytes, 748);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_resource_ledger_balances_injected_error_cleanup_and_commands() {
    let resources = Arc::new(MetalResourceLedger::default());
    resources.register_buffer(
        0x1000usize as MetalObjcId,
        16,
        MetalTrackedBufferClass::ActiveGeneral,
    );
    resources.register_buffer(
        0x2000usize as MetalObjcId,
        32,
        MetalTrackedBufferClass::ActiveGeneral,
    );
    let first = MetalCommandLease::new(Arc::clone(&resources));
    let second = MetalCommandLease::new(Arc::clone(&resources));
    assert_eq!(resources.snapshot().in_flight_commands, 2);
    assert_eq!(resources.snapshot().command_high_water, 2);
    assert_eq!(resources.snapshot().command_submissions, 2);
    resources.record_host_upload(4_096);
    resources.record_host_readback(2_048);

    resources.release_buffer(0x1000usize as MetalObjcId);
    resources.release_buffer(0x2000usize as MetalObjcId);
    drop(first);
    drop(second);
    let cleaned = resources.snapshot();
    assert_eq!(cleaned.active_general_buffers, 0);
    assert_eq!(cleaned.in_flight_commands, 0);
    assert_eq!(cleaned.command_submissions, 2);
    assert_eq!(cleaned.host_upload_bytes, 4_096);
    assert_eq!(cleaned.host_readback_bytes, 2_048);
    assert_eq!(cleaned.ledger_live_bytes, 0);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn explicit_metal_limit_can_only_lower_default_policy() {
    let gib = 1024 * 1024 * 1024;
    let ledger = MetalResourceLedger {
        state: Mutex::new(MetalResourceLedgerState::new(10 * gib, 2 * gib)),
    };
    ledger.set_working_set_limit_bytes(6 * gib).unwrap();
    assert_eq!(ledger.snapshot().working_set_limit_bytes, 6 * gib);
    assert!(ledger.allocation_would_exceed_limit(6 * gib - 1, 2));

    ledger.set_working_set_limit_bytes(20 * gib).unwrap();
    assert_eq!(ledger.snapshot().working_set_limit_bytes, 9 * gib);
    assert!(ledger.set_working_set_limit_bytes(0).is_err());
}

#[test]
fn command_context_label_includes_actionable_details() {
    let context = MetalCommandContext::new("deferred_expert_phase")
        .with("position", 17)
        .with("layer", 3)
        .with("experts", "1,7,9,11")
        .with("width", 4096);

    assert_eq!(
        context.label(),
        "Flash-MoE deferred_expert_phase position=17 layer=3 experts=1,7,9,11 width=4096"
    );
    assert_eq!(
        context.detail_summary(),
        "position=17, layer=3, experts=1,7,9,11, width=4096"
    );
}

#[test]
fn command_status_names_known_and_unknown_values() {
    assert_eq!(MetalCommandStatus::from_raw(0).to_string(), "not_enqueued");
    assert_eq!(MetalCommandStatus::from_raw(3).to_string(), "scheduled");
    assert_eq!(MetalCommandStatus::from_raw(4).to_string(), "completed");
    assert_eq!(MetalCommandStatus::from_raw(5).to_string(), "error");
    assert_eq!(MetalCommandStatus::from_raw(99).to_string(), "unknown(99)");
    assert!(MetalCommandStatus::Completed.is_terminal());
    assert!(MetalCommandStatus::Error.is_terminal());
    assert!(!MetalCommandStatus::Scheduled.is_terminal());
}

#[test]
fn command_failure_diagnostic_is_actionable() {
    let context = MetalCommandContext::new("gqa_attention_scores")
        .with("layer", 12)
        .with("position", 128)
        .with("tokens", 129)
        .with("q_heads", 32)
        .with("kv_heads", 8);

    let message = format_metal_command_failure(
        MetalCommandFailureKind::Timeout,
        &context,
        Duration::from_millis(1234),
        MetalCommandStatus::Scheduled,
        Some("GPU timeout"),
    );

    assert!(message.contains("timed out"));
    assert!(message.contains("label=\"Flash-MoE gqa_attention_scores"));
    assert!(message.contains("elapsed=1234ms"));
    assert!(message.contains("status=scheduled"));
    assert!(message.contains("metal_error=\"GPU timeout\""));
    assert!(message.contains("layer=12"));
    assert!(message.contains("position=128"));
    assert!(message.contains("tokens=129"));
}

#[test]
fn command_failure_marks_buffers_for_release() {
    let context = MetalCommandContext::new("lm_head_topk").with("rows", 42);
    let error = MetalCommandBufferFailure::failed(
        &context,
        Duration::from_millis(7),
        MetalCommandStatus::Error,
        None,
    );
    assert!(error.should_release_buffers());
    assert!(error.to_string().contains("none reported"));
}

#[test]
fn command_wait_policy_uses_upstream_shaped_timeout_defaults() {
    let policy = MetalCommandWaitPolicy::default();
    assert_eq!(policy.timeout, Duration::from_secs(120));
    assert_eq!(policy.poll_interval, Duration::from_micros(100));
}

#[test]
fn command_wait_resolution_handles_completed_failed_timeout_and_pending() {
    let context = MetalCommandContext::new("cmd3");

    assert_eq!(
        resolve_metal_command_wait(
            &context,
            Duration::from_millis(4),
            MetalCommandStatus::Completed,
            None,
            false,
        ),
        MetalCommandWaitResult::Finished(Ok(()))
    );

    let failed = resolve_metal_command_wait(
        &context,
        Duration::from_millis(5),
        MetalCommandStatus::Error,
        Some("encoder failed".to_string()),
        false,
    );
    assert!(matches!(
        failed,
        MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure {
            kind: MetalCommandFailureKind::Failed,
            ..
        }))
    ));

    let timed_out = resolve_metal_command_wait(
        &context,
        Duration::from_secs(120),
        MetalCommandStatus::Scheduled,
        Some("still running".to_string()),
        true,
    );
    assert!(matches!(
        timed_out,
        MetalCommandWaitResult::Finished(Err(MetalCommandBufferFailure {
            kind: MetalCommandFailureKind::Timeout,
            ..
        }))
    ));

    assert_eq!(
        resolve_metal_command_wait(
            &context,
            Duration::from_millis(1),
            MetalCommandStatus::Scheduled,
            None,
            false,
        ),
        MetalCommandWaitResult::Pending
    );
}

#[test]
fn shader_source_defines_full_forward_kernel_set() {
    for kernel in REQUIRED_FORWARD_KERNELS {
        assert!(
            METAL_SHADERS.contains(&format!("kernel void {kernel}")),
            "missing Metal kernel {kernel}"
        );
    }
    assert!(METAL_SHADERS.contains("threadgroup float input_cache"));
    assert!(METAL_SHADERS.contains("simd_sum(acc)"));
    assert!(METAL_SHADERS.contains("thread_index_in_simdgroup"));
    assert!(METAL_SHADERS.contains("constant uint& group_size"));
    assert!(METAL_SHADERS.contains("constant uint& top_k"));
    assert!(METAL_SHADERS.contains("fma(float(byte & 0x0f), scale0 * x0, bias0 * x0)"));
    assert!(
        !METAL_SHADERS.contains("uint half"),
        "`half` is a Metal scalar type and cannot be reused as a variable name"
    );
}

#[test]
fn pipeline_name_set_matches_declared_forward_kernel_surface() {
    let mut compiled = MetalPipelineNameSet::new().kernel_names();
    compiled.sort_unstable();
    compiled.dedup();

    let mut required = REQUIRED_FORWARD_KERNELS.to_vec();
    required.sort_unstable();
    required.dedup();

    assert_eq!(compiled, required);
}

#[test]
fn deepseek_shader_source_defines_load_resolved_kernel_surface() {
    for kernel in DEEPSEEK_V4_REQUIRED_METAL_KERNELS {
        assert!(
            super::super::deepseek_metal::DEEPSEEK_V4_METAL_SHADERS.contains(kernel),
            "missing DeepSeek V4 Metal kernel {kernel}"
        );
    }
    assert!(
        super::super::deepseek_metal::DEEPSEEK_V4_METAL_SHADERS
            .contains("kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32")
    );
    assert!(
        super::super::deepseek_metal::DEEPSEEK_V4_METAL_SHADERS
            .contains("kernel_mul_mv_slots6_q2_K_sum6_f32")
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn deepseek_shader_library_compiles_every_required_pipeline() {
    objc2::rc::autoreleasepool(|_| unsafe {
        let device = OwnedMetalObject::new(metal_default_device()).unwrap();
        let pipelines = DeepSeekMetalPipelineSet::compile(device.id()).unwrap();
        for &kernel in DEEPSEEK_V4_REQUIRED_METAL_KERNELS {
            pipelines.require(kernel).unwrap();
        }
    });
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn native_mxfp4_matvec_matches_e2m1_e8m0_reference() {
    objc2::rc::autoreleasepool(|_| unsafe {
        let runtime = MetalRuntime::compile(METAL_SHADERS, MetalPipelineNameSet::new()).unwrap();
        let packed = [vec![0x22u8; 16], vec![0x95u8; 16]].concat();
        let input = (1..=32).map(|value| value as f32).collect::<Vec<_>>();
        let scales = [127u8, 127u8];
        let packed_buffer = OwnedMetalObject::new(msg_send_id3_ptr_usize_u64(
            runtime.device,
            sel("newBufferWithBytes:length:options:"),
            packed.as_ptr().cast(),
            packed.len(),
            0,
        ))
        .unwrap();
        let input_bytes = f32_as_bytes(&input);
        let input_buffer = OwnedMetalObject::new(msg_send_id3_ptr_usize_u64(
            runtime.device,
            sel("newBufferWithBytes:length:options:"),
            input_bytes.as_ptr().cast(),
            input_bytes.len(),
            0,
        ))
        .unwrap();
        let scale_buffer = OwnedMetalObject::new(msg_send_id3_ptr_usize_u64(
            runtime.device,
            sel("newBufferWithBytes:length:options:"),
            scales.as_ptr().cast(),
            scales.len(),
            0,
        ))
        .unwrap();
        let output_buffer = OwnedMetalObject::new(msg_send_id2_usize_u64(
            runtime.device,
            sel("newBufferWithLength:options:"),
            2 * std::mem::size_of::<f32>(),
            0,
        ))
        .unwrap();
        let mut encoding = MetalCommandEncoding::new(
            runtime.command_queue,
            Arc::new(MetalResourceLedger::default()),
            "failed to create MXFP4 test command buffer",
            "failed to create MXFP4 test encoder",
        )
        .unwrap();
        let encoder = encoding.encoder();
        msg_send_void1_id(
            encoder,
            sel("setComputePipelineState:"),
            runtime.pipelines.mxfp4_e8m0_pipeline,
        );
        set_buffer(encoder, packed_buffer.id(), 0);
        set_buffer(encoder, input_buffer.id(), 1);
        set_buffer(encoder, scale_buffer.id(), 2);
        set_buffer(encoder, output_buffer.id(), 4);
        for (index, value) in [(5, 2u32), (6, 32), (7, 1), (8, 32)] {
            set_bytes(encoder, u32_as_bytes(&value), index);
        }
        dispatch_q4_threadgroups(encoder, 2);
        encoding.end_encoding();
        let (command_buffer, command_lease) = encoding.into_command_buffer();
        let context = MetalCommandContext::new("native MXFP4 reference test");
        commit_metal_command_buffer(command_buffer, &context);
        wait_for_metal_command_buffer(command_buffer, &context).unwrap();
        let actual = read_f32_buffer(output_buffer.id(), 2);
        release(command_buffer);
        drop(command_lease);
        assert!((actual[0] - 528.0).abs() < 1e-4, "{actual:?}");
        assert!((actual[1] - 632.0).abs() < 1e-4, "{actual:?}");
    });
}

#[test]
fn pipeline_set_releases_every_resolved_pipeline() {
    let pipelines = test_pipeline_set();
    let mut released = Vec::new();
    pipelines.release_with(|pipeline| released.push(pipeline));
    assert_eq!(
        released,
        [
            vec![1, 2, 38, 3, 4, 5, 6, 7, 8],
            vec![33, 37, 34, 35, 36],
            vec![24, 25, 26, 41, 42, 43],
            vec![9, 10, 11, 45, 39, 47, 46, 12, 13, 14, 15, 44, 40, 16],
            vec![18, 19, 27, 28, 20, 21, 29, 30, 22, 23, 31, 32],
        ]
        .concat()
    );
}

fn test_pipeline_set() -> MetalPipelineSet<i32> {
    MetalPipelineSet {
        q4_pipeline: 1,
        q4_bf16_scale_bias_pipeline: 2,
        mxfp4_e8m0_pipeline: 38,
        q4_swiglu_pipeline: 3,
        q4_swiglu_bf16_scale_bias_pipeline: 4,
        q4_mmap_pipeline: 5,
        q4_mmap_bf16_scale_bias_pipeline: 6,
        q4_mmap_batch_pipeline: 7,
        q4_mmap_batch_bf16_scale_bias_pipeline: 8,
        q4_mmap_multilinear_bf16_scale_bias_pipeline: 33,
        glm_mla_prepare_query_kv_pipeline: 37,
        glm_mla_absorbed_scores_pipeline: 34,
        glm_mla_softmax_pipeline: 35,
        glm_mla_context_pipeline: 36,
        dense_mmap_bf16_pipeline: 24,
        dense_mmap_f16_pipeline: 25,
        dense_mmap_f32_pipeline: 26,
        dense_matrix_bf16_pipeline: 41,
        dense_matrix_f16_pipeline: 42,
        dense_matrix_f32_pipeline: 43,
        rms_norm_reduced_pipeline: 9,
        residual_rms_norm_pipeline: 10,
        attention_pipeline: 11,
        qwen_prepare_qkv_rows_pipeline: 45,
        qwen_causal_attention_rows_pipeline: 39,
        qwen_apply_attention_gate_pipeline: 47,
        qwen_final_rms_norm_row_pipeline: 46,
        expert_mlp_pipeline: 12,
        silu_product_pipeline: 13,
        shared_expert_activation_pipeline: 14,
        combine_expert_phase_pipeline: 15,
        qwen_layer_major_gather_pipeline: 44,
        qwen_layer_major_combine_pipeline: 40,
        fill_zero_pipeline: 16,
        topk_vocab_pipeline: 18,
        linear_conv1d_bf16_pipeline: 19,
        linear_conv1d_f16_pipeline: 27,
        linear_conv1d_f32_pipeline: 28,
        linear_rms_norm_qk_pipeline: 20,
        linear_decay_beta_bf16_pipeline: 21,
        linear_decay_beta_f16_pipeline: 29,
        linear_decay_beta_f32_pipeline: 30,
        linear_delta_step_pipeline: 22,
        linear_gated_rms_norm_bf16_pipeline: 23,
        linear_gated_rms_norm_f16_pipeline: 31,
        linear_gated_rms_norm_f32_pipeline: 32,
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn test_objc_pipeline_set(id: MetalObjcId) -> MetalPipelineSet<MetalObjcId> {
    MetalPipelineSet {
        q4_pipeline: id,
        q4_bf16_scale_bias_pipeline: id,
        mxfp4_e8m0_pipeline: id,
        q4_swiglu_pipeline: id,
        q4_swiglu_bf16_scale_bias_pipeline: id,
        q4_mmap_pipeline: id,
        q4_mmap_bf16_scale_bias_pipeline: id,
        q4_mmap_batch_pipeline: id,
        q4_mmap_batch_bf16_scale_bias_pipeline: id,
        q4_mmap_multilinear_bf16_scale_bias_pipeline: id,
        glm_mla_prepare_query_kv_pipeline: id,
        glm_mla_absorbed_scores_pipeline: id,
        glm_mla_softmax_pipeline: id,
        glm_mla_context_pipeline: id,
        dense_mmap_bf16_pipeline: id,
        dense_mmap_f16_pipeline: id,
        dense_mmap_f32_pipeline: id,
        dense_matrix_bf16_pipeline: id,
        dense_matrix_f16_pipeline: id,
        dense_matrix_f32_pipeline: id,
        rms_norm_reduced_pipeline: id,
        residual_rms_norm_pipeline: id,
        attention_pipeline: id,
        qwen_prepare_qkv_rows_pipeline: id,
        qwen_causal_attention_rows_pipeline: id,
        qwen_apply_attention_gate_pipeline: id,
        qwen_final_rms_norm_row_pipeline: id,
        expert_mlp_pipeline: id,
        silu_product_pipeline: id,
        shared_expert_activation_pipeline: id,
        combine_expert_phase_pipeline: id,
        qwen_layer_major_gather_pipeline: id,
        qwen_layer_major_combine_pipeline: id,
        fill_zero_pipeline: id,
        topk_vocab_pipeline: id,
        linear_conv1d_bf16_pipeline: id,
        linear_conv1d_f16_pipeline: id,
        linear_conv1d_f32_pipeline: id,
        linear_rms_norm_qk_pipeline: id,
        linear_decay_beta_bf16_pipeline: id,
        linear_decay_beta_f16_pipeline: id,
        linear_decay_beta_f32_pipeline: id,
        linear_delta_step_pipeline: id,
        linear_gated_rms_norm_bf16_pipeline: id,
        linear_gated_rms_norm_f16_pipeline: id,
        linear_gated_rms_norm_f32_pipeline: id,
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn batch_projection_input_reports_declared_input_len() {
    let values = [1.0, 2.0, 3.0];
    assert_eq!(MetalBatchProjectionInput::Cpu(&values).len(), values.len());

    let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
    assert_eq!(
        MetalBatchProjectionInput::Buffer { buffer: id, len: 7 }.len(),
        7
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn qwen_layer_major_group_plan_preserves_route_and_expert_order() {
    let (source_rows, output_indices, groups) =
        qwen_layer_major_group_plan(&[1, 0, 0, 1], 2, 2).unwrap();

    assert_eq!(source_rows, vec![0, 1, 0, 1]);
    assert_eq!(output_indices, vec![2, 0, 1, 3]);
    assert_eq!(groups, vec![(0, 2), (2, 2)]);

    assert!(qwen_layer_major_group_plan(&[0, 0], 2, 1).is_err());
    assert!(qwen_layer_major_group_plan(&[2], 2, 1).is_err());
    assert!(qwen_layer_major_group_plan(&[], 0, 0).is_err());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn post_attention_prep_builds_declared_cmd3_metal_input() {
    let mut prep = MetalPostAttentionPrep::new(
        3,
        8,
        16,
        vec![(2, 0.75), (5, 0.25)],
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    )
    .unwrap();

    assert_eq!(prep.width, 8);
    assert_eq!(prep.active, vec![(2, 0.75), (5, 0.25)]);
    assert!(prep.routing_command().is_none());
    assert_eq!(prep.input.state(), prep.state);
    assert!(prep.state.is_declared_graph_state());

    let command = test_fused_prep_routing_command(3, 16, &prep.active);
    let attached = prep.attach_routing_command(command.clone()).unwrap();
    assert_eq!(attached, command);
    assert_eq!(prep.routing_command(), Some(&command));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn post_attention_prep_rejects_mismatched_routing_command() {
    let mut prep = MetalPostAttentionPrep::new(
        3,
        8,
        16,
        vec![(2, 0.75), (5, 0.25)],
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    )
    .unwrap();
    let command = test_fused_prep_routing_command(4, 16, &prep.active);

    let err = prep.attach_routing_command(command).unwrap_err();
    assert!(err.to_string().contains("routing layer 3"), "{err:#}");
    assert!(prep.routing_command().is_none());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn post_attention_prep_rejects_undeclared_cmd3_metal_input() {
    let err = MetalPostAttentionPrep::new(
        3,
        0,
        16,
        vec![(2, 1.0)],
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    )
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("Metal post-attention input for layer 3")
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_deferred_output_declares_gpu_resident_buffers() {
    let hidden = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
    let next_normed = hidden;
    let output = MetalCmd3DeferredOutput::new(
        hidden,
        Some(next_normed),
        FlashMoeCmd3OutputState::gpu_resident(16, true),
    )
    .unwrap();

    assert_eq!(output.hidden_buffer, hidden);
    assert_eq!(output.next_normed_buffer, Some(next_normed));
    assert_eq!(output.output_state.width(), 16);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_deferred_output_rejects_undeclared_buffer_state() {
    let hidden = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();

    let missing_next = MetalCmd3DeferredOutput::new(
        hidden,
        None,
        FlashMoeCmd3OutputState::gpu_resident(16, true),
    )
    .unwrap_err();
    assert!(
        missing_next
            .to_string()
            .contains("next-norm buffer presence"),
        "{missing_next:#}"
    );

    let null_hidden = MetalCmd3DeferredOutput::new(
        std::ptr::null_mut(),
        None,
        FlashMoeCmd3OutputState::gpu_resident(16, false),
    )
    .unwrap_err();
    assert!(
        null_hidden.to_string().contains("non-null hidden buffer"),
        "{null_hidden:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_phase_plan_declares_supported_command_shape() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(16, true);

    let plan = MetalCmd3PhasePlan::new(9, 3, 4, 16, 4, 4, output_state, true).unwrap();

    assert_eq!(plan.position, 9);
    assert_eq!(plan.layer, 3);
    assert_eq!(plan.expert_count, 4);
    assert_eq!(plan.width, 16);
    assert_eq!(plan.output_state, output_state);
    assert!(plan.has_next_norm);
    assert_eq!(plan.width_u32(), 16);
    assert_eq!(plan.expert_outputs_bytes().unwrap(), 4 * 16 * 4);
    assert_eq!(plan.shared_output_bytes().unwrap(), 16 * 4);
    assert_eq!(plan.hidden_output_bytes().unwrap(), 16 * 4);
    assert_eq!(plan.next_normed_output_bytes().unwrap(), Some(16 * 4));
    assert_eq!(plan.expert_output_offset(0).unwrap(), 0);
    assert_eq!(plan.expert_output_offset(3).unwrap(), 3 * 16 * 4);

    let combine = MetalCmd3CombinePlan::new(plan);
    assert_eq!(combine.width, 16);
    assert_eq!(combine.active_count, 4);
    assert_eq!(combine.active_count_u32(), 4);
    assert_eq!(combine.dispatch_threads, 16);
    assert_eq!(combine.routing_weights_bytes().unwrap(), 4 * 4);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_phase_plan_rejects_unsupported_command_shape() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(16, true);

    let count_err = MetalCmd3PhasePlan::new(9, 3, 4, 16, 3, 4, output_state, true).unwrap_err();
    assert!(
        count_err.to_string().contains("expert count 4"),
        "{count_err:#}"
    );

    let output_err = MetalCmd3PhasePlan::new(9, 3, 4, 16, 4, 4, output_state, false).unwrap_err();
    assert!(
        output_err
            .to_string()
            .contains("next-norm output declaration"),
        "{output_err:#}"
    );

    let wide_state = FlashMoeCmd3OutputState::gpu_resident(u32::MAX as usize + 1, false);
    let width_err =
        MetalCmd3PhasePlan::new(9, 3, 4, u32::MAX as usize + 1, 4, 4, wide_state, false)
            .unwrap_err();
    assert!(
        width_err.to_string().contains("does not fit Metal u32"),
        "{width_err:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_input_buffers_carry_declared_phase_inputs() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
    let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();

    let inputs = MetalCmd3InputBuffers::new(
        phase,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
    )
    .unwrap();

    assert_eq!(inputs.normed, 0x1000usize as MetalObjcId);
    assert_eq!(inputs.residual, 0x2000usize as MetalObjcId);
    assert_eq!(inputs.phase, phase);

    let missing_normed =
        MetalCmd3InputBuffers::new(phase, std::ptr::null_mut(), inputs.residual).unwrap_err();
    assert!(
        missing_normed.to_string().contains("non-null normed"),
        "{missing_normed:#}"
    );

    let missing_residual =
        MetalCmd3InputBuffers::new(phase, inputs.normed, std::ptr::null_mut()).unwrap_err();
    assert!(
        missing_residual.to_string().contains("non-null residual"),
        "{missing_residual:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_combine_buffers_carry_declared_bindings() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
    let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
    let plan = MetalCmd3CombinePlan::new(phase);

    let buffers = MetalCmd3CombineBuffers::new(
        plan,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
        0x3000usize as MetalObjcId,
    )
    .unwrap();

    assert_eq!(buffers.routing_weights, 0x1000usize as MetalObjcId);
    assert_eq!(buffers.width, 0x2000usize as MetalObjcId);
    assert_eq!(buffers.active_count, 0x3000usize as MetalObjcId);
    assert_eq!(buffers.layout.width_u32, 4);
    assert_eq!(buffers.layout.active_count_u32, 2);
    assert_eq!(buffers.layout.routing_weights_bytes, 2 * 4);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_combine_stage_buffers_match_declared_layout() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
    let payloads = vec![
        ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
        ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
    ];
    let execution = MetalCmd3ExecutionPlan::new(
        9,
        3,
        2,
        4,
        2,
        output_state,
        ScheduledSharedExpertPhaseRef::None,
        None,
        &payloads,
    )
    .unwrap();
    let inputs = MetalCmd3InputBuffers::new(
        execution.phase,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
    )
    .unwrap();
    let outputs = MetalCmd3OutputBuffers::new(
        &execution,
        0x3000usize as MetalObjcId,
        0x4000usize as MetalObjcId,
        0x5000usize as MetalObjcId,
        None,
    )
    .unwrap();
    let combine = MetalCmd3CombineBuffers::new(
        execution.combine,
        0x6000usize as MetalObjcId,
        0x7000usize as MetalObjcId,
        0x8000usize as MetalObjcId,
    )
    .unwrap();

    let stage =
        MetalCmd3CombineStageBuffers::new(execution.combine, inputs, &outputs, combine).unwrap();

    assert_eq!(stage.residual, inputs.residual);
    assert_eq!(stage.shared_output, outputs.shared_output);
    assert_eq!(stage.expert_outputs, outputs.expert_outputs);
    assert_eq!(stage.routing_weights, combine.routing_weights);
    assert_eq!(stage.hidden, outputs.hidden);
    assert_eq!(stage.width, combine.width);
    assert_eq!(stage.active_count, combine.active_count);
    assert_eq!(stage.plan, execution.combine);

    let stale_plan = MetalCmd3CombinePlan {
        width: execution.combine.width,
        active_count: execution.combine.active_count + 1,
        dispatch_threads: execution.combine.dispatch_threads,
    };
    let stale =
        MetalCmd3CombineStageBuffers::new(stale_plan, inputs, &outputs, combine).unwrap_err();
    assert!(
        stale.to_string().contains("constants do not match plan"),
        "{stale:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_next_norm_buffers_carry_declared_bindings() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, true);
    let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, true).unwrap();
    let plan = MetalCmd3NextNormPlan::new(phase, Some(4)).unwrap().unwrap();

    let buffers = MetalCmd3NextNormBuffers::new(
        plan,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
        0x3000usize as MetalObjcId,
        0x4000usize as MetalObjcId,
    )
    .unwrap();

    assert_eq!(buffers.hidden, 0x1000usize as MetalObjcId);
    assert_eq!(buffers.weight, 0x2000usize as MetalObjcId);
    assert_eq!(buffers.next_normed, 0x3000usize as MetalObjcId);
    assert_eq!(buffers.width, 0x4000usize as MetalObjcId);
    assert_eq!(buffers.layout.width_u32, 4);
    assert_eq!(buffers.layout.weight_bytes, 4 * 4);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_shared_stage_buffers_require_declared_source() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
    let shared = SharedExpertPhaseResidentProjections {
        gate: test_resident_q4_projection("gate", 6, 4),
        up: test_resident_q4_projection("up", 6, 4),
        down: test_resident_q4_projection("down", 4, 6),
        router: Some(test_resident_q4_projection("router", 2, 4)),
        shared_experts: 2,
        intermediate: 3,
        width: 4,
    };
    let payloads = vec![
        ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
        ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
    ];
    let plan = MetalCmd3ExecutionPlan::new(
        9,
        3,
        2,
        4,
        2,
        output_state,
        ScheduledSharedExpertPhaseRef::Resident(&shared),
        None,
        &payloads,
    )
    .unwrap();
    let inputs = MetalCmd3InputBuffers::new(
        plan.phase,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
    )
    .unwrap();
    let outputs = MetalCmd3OutputBuffers::new(
        &plan,
        0x3000usize as MetalObjcId,
        0x4000usize as MetalObjcId,
        0x5000usize as MetalObjcId,
        None,
    )
    .unwrap();
    let combine = MetalCmd3CombineBuffers::new(
        plan.combine,
        0x6000usize as MetalObjcId,
        0x7000usize as MetalObjcId,
        0x8000usize as MetalObjcId,
    )
    .unwrap();
    let work = MetalCmd3SharedWorkBuffers::new(
        plan.shared,
        0x9000usize as MetalObjcId,
        0xa000usize as MetalObjcId,
        0xb000usize as MetalObjcId,
        0xc000usize as MetalObjcId,
        0xd000usize as MetalObjcId,
        0xe000usize as MetalObjcId,
    )
    .unwrap();

    let projected =
        MetalCmd3SharedStageBuffers::projected(plan.shared, inputs, &outputs, combine, work)
            .unwrap();

    assert_eq!(projected.source, MetalCmd3SharedPhaseSource::Resident);
    assert_eq!(projected.normed, inputs.normed);
    assert_eq!(projected.width, combine.width);
    assert_eq!(projected.shared_output, outputs.shared_output);
    assert_eq!(projected.work, Some(work));

    let no_shared = MetalCmd3SharedPhasePlan::none(4);
    let fill_zero =
        MetalCmd3SharedStageBuffers::fill_zero(no_shared, inputs, &outputs, combine).unwrap();
    assert_eq!(fill_zero.source, MetalCmd3SharedPhaseSource::None);
    assert_eq!(fill_zero.work, None);
    assert_eq!(fill_zero.shared_output, outputs.shared_output);

    let projected_none =
        MetalCmd3SharedStageBuffers::projected(no_shared, inputs, &outputs, combine, work)
            .unwrap_err();
    assert!(
        projected_none
            .to_string()
            .contains("declared shared expert source"),
        "{projected_none:#}"
    );

    let fill_projected =
        MetalCmd3SharedStageBuffers::fill_zero(plan.shared, inputs, &outputs, combine).unwrap_err();
    assert!(
        fill_projected
            .to_string()
            .contains("no shared expert source"),
        "{fill_projected:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_execution_plan_declares_full_command_topology() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, true);
    let shared = SharedExpertPhaseResidentProjections {
        gate: test_resident_q4_projection("gate", 6, 4),
        up: test_resident_q4_projection("up", 6, 4),
        down: test_resident_q4_projection("down", 4, 6),
        router: Some(test_resident_q4_projection("router", 2, 4)),
        shared_experts: 2,
        intermediate: 3,
        width: 4,
    };
    let payloads = vec![
        ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
        ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
    ];

    let plan = MetalCmd3ExecutionPlan::new(
        9,
        3,
        2,
        4,
        2,
        output_state,
        ScheduledSharedExpertPhaseRef::Resident(&shared),
        Some(4),
        &payloads,
    )
    .unwrap();

    assert_eq!(plan.phase.position, 9);
    assert_eq!(plan.phase.layer, 3);
    assert_eq!(plan.phase.output_state, output_state);
    assert_eq!(plan.shared.source, MetalCmd3SharedPhaseSource::Resident);
    assert_eq!(plan.shared.total_intermediate, 6);
    assert_eq!(plan.active_experts.len(), 2);
    assert_eq!(plan.active_experts[0].intermediate, 5);
    assert_eq!(plan.active_experts[0].output_offset, 0);
    assert_eq!(plan.active_experts[1].intermediate, 7);
    assert_eq!(plan.active_experts[1].output_offset, 4 * 4);
    assert_eq!(plan.combine.active_count, 2);
    assert_eq!(plan.next_norm.unwrap().width, 4);

    let layout = plan.buffer_layout().unwrap();
    assert_eq!(layout.width_u32, 4);
    assert_eq!(layout.active_count_u32, 2);
    assert_eq!(layout.expert_outputs_bytes, 2 * 4 * 4);
    assert_eq!(layout.shared_output_bytes, 4 * 4);
    assert_eq!(layout.hidden_output_bytes, 4 * 4);
    assert_eq!(layout.next_normed_output_bytes, Some(4 * 4));

    let context = plan.command_context("1,7");
    assert_eq!(
        context.label(),
        "Flash-MoE deferred_expert_phase_from_buffers position=9 layer=3 active_experts=2 experts=1,7 width=4 shared=true next_norm=true"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_output_buffers_match_declared_output_state() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, true);
    let payloads = vec![ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(
        5, 4,
    ))];
    let plan = MetalCmd3ExecutionPlan::new(
        9,
        3,
        1,
        4,
        1,
        output_state,
        ScheduledSharedExpertPhaseRef::None,
        Some(4),
        &payloads,
    )
    .unwrap();

    let buffers = MetalCmd3OutputBuffers::new(
        &plan,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
        0x3000usize as MetalObjcId,
        Some(0x4000usize as MetalObjcId),
    )
    .unwrap();

    assert_eq!(buffers.layout.width_u32, 4);
    assert_eq!(buffers.layout.active_count_u32, 1);
    assert_eq!(buffers.layout.expert_outputs_bytes, 4 * 4);
    assert_eq!(buffers.layout.shared_output_bytes, 4 * 4);
    assert_eq!(buffers.layout.hidden_output_bytes, 4 * 4);
    assert_eq!(buffers.layout.next_normed_output_bytes, Some(4 * 4));
    assert_eq!(buffers.hidden, 0x3000usize as MetalObjcId);

    let missing_next = MetalCmd3OutputBuffers::new(
        &plan,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
        0x3000usize as MetalObjcId,
        None,
    )
    .unwrap_err();
    assert!(
        missing_next
            .to_string()
            .contains("does not match declared output state"),
        "{missing_next:#}"
    );

    let no_next_plan = MetalCmd3ExecutionPlan::new(
        9,
        3,
        1,
        4,
        1,
        FlashMoeCmd3OutputState::gpu_resident(4, false),
        ScheduledSharedExpertPhaseRef::None,
        None,
        &payloads,
    )
    .unwrap();
    let unexpected_next = MetalCmd3OutputBuffers::new(
        &no_next_plan,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
        0x3000usize as MetalObjcId,
        Some(0x4000usize as MetalObjcId),
    )
    .unwrap_err();
    assert!(
        unexpected_next
            .to_string()
            .contains("does not match declared output state"),
        "{unexpected_next:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_active_expert_work_buffers_carry_staged_projection_layout() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
    let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
    let payload = ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(6, 4));
    let plan = MetalCmd3ActiveExpertPlan::new(phase, 1, &payload).unwrap();

    let staged = MetalCmd3ActiveExpertWorkBuffers::new(
        plan,
        Some(0x1100usize as MetalObjcId),
        Some(0x1200usize as MetalObjcId),
        0x1000usize as MetalObjcId,
    )
    .unwrap();

    assert_eq!(staged.gate_out, Some(0x1100usize as MetalObjcId));
    assert_eq!(staged.up_out, Some(0x1200usize as MetalObjcId));
    assert_eq!(staged.activated, 0x1000usize as MetalObjcId);
    assert_eq!(staged.layout.intermediate_u32, 6);
    assert_eq!(staged.layout.activation_bytes, 6 * 4);
    assert_eq!(staged.layout.projection_output_bytes, Some(6 * 4));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_active_expert_stage_buffers_match_declared_layout() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
    let payloads = vec![
        ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(5, 4)),
        ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(7, 4)),
    ];
    let execution = MetalCmd3ExecutionPlan::new(
        9,
        3,
        2,
        4,
        2,
        output_state,
        ScheduledSharedExpertPhaseRef::None,
        None,
        &payloads,
    )
    .unwrap();
    let inputs = MetalCmd3InputBuffers::new(
        execution.phase,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
    )
    .unwrap();
    let outputs = MetalCmd3OutputBuffers::new(
        &execution,
        0x3000usize as MetalObjcId,
        0x4000usize as MetalObjcId,
        0x5000usize as MetalObjcId,
        None,
    )
    .unwrap();
    let active_plan = execution.active_experts[1];
    let work = MetalCmd3ActiveExpertWorkBuffers::new(
        active_plan,
        Some(0x6100usize as MetalObjcId),
        Some(0x6200usize as MetalObjcId),
        0x6000usize as MetalObjcId,
    )
    .unwrap();

    let stage =
        MetalCmd3ActiveExpertStageBuffers::new(active_plan, inputs, &outputs, work).unwrap();

    assert_eq!(stage.normed, inputs.normed);
    assert_eq!(stage.activated, work.activated);
    assert_eq!(stage.expert_outputs, outputs.expert_outputs);
    assert_eq!(stage.output_offset, active_plan.output_offset);
    assert_eq!(stage.plan, active_plan);
    assert_eq!(stage.work, work);

    let stale_plan = MetalCmd3ActiveExpertPlan {
        intermediate: active_plan.intermediate + 1,
        ..active_plan
    };
    let stale =
        MetalCmd3ActiveExpertStageBuffers::new(stale_plan, inputs, &outputs, work).unwrap_err();
    assert!(
        stale
            .to_string()
            .contains("work layout does not match plan"),
        "{stale:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_execution_plan_rejects_mismatched_subplans() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
    let payloads = vec![ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(
        5, 6,
    ))];

    let payload_err = MetalCmd3ExecutionPlan::new(
        9,
        3,
        1,
        4,
        1,
        output_state,
        ScheduledSharedExpertPhaseRef::None,
        None,
        &payloads,
    )
    .unwrap_err();
    assert!(
        payload_err
            .to_string()
            .contains("does not match phase width 4"),
        "{payload_err:#}"
    );

    let shared = SharedExpertPhaseWeights::new(
        Arc::new(vec![1.0; 24]),
        Arc::new(vec![2.0; 24]),
        Arc::new(vec![3.0; 24]),
        Arc::new(vec![4.0; 8]),
        2,
        3,
        4,
    )
    .unwrap();
    let payloads = vec![ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(
        5, 8,
    ))];
    let shared_err = MetalCmd3ExecutionPlan::new(
        9,
        3,
        1,
        8,
        1,
        FlashMoeCmd3OutputState::gpu_resident(8, false),
        ScheduledSharedExpertPhaseRef::Dense(&shared),
        None,
        &payloads,
    )
    .unwrap_err();
    assert!(
        shared_err.to_string().contains("shared expert width 4"),
        "{shared_err:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn scheduled_cmd3_builder_names_missing_shared_implementation() {
    assert!(
        MetalScheduledCmd3Builder::require_shared_implementation(
            MetalCmd3SharedPhaseSource::Resident
        )
        .is_ok()
    );
    assert!(
        MetalScheduledCmd3Builder::require_shared_implementation(MetalCmd3SharedPhaseSource::None)
            .is_ok()
    );
    let error =
        MetalScheduledCmd3Builder::require_shared_implementation(MetalCmd3SharedPhaseSource::Dense)
            .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("dense CPU shared-expert weights are not a declared implementation"),
        "{error:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_next_norm_plan_declares_weight_slice_and_dispatch() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(16, true);
    let phase = MetalCmd3PhasePlan::new(9, 3, 4, 16, 4, 4, output_state, true).unwrap();

    let plan = MetalCmd3NextNormPlan::new(phase, Some(32))
        .unwrap()
        .unwrap();

    assert_eq!(plan.width, 16);
    assert_eq!(plan.dispatch_threads, 256);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_next_norm_plan_rejects_undeclared_or_short_weights() {
    let with_next = MetalCmd3PhasePlan::new(
        9,
        3,
        4,
        16,
        4,
        4,
        FlashMoeCmd3OutputState::gpu_resident(16, true),
        true,
    )
    .unwrap();
    let short = MetalCmd3NextNormPlan::new(with_next, Some(15)).unwrap_err();
    assert!(
        short.to_string().contains("smaller than width 16"),
        "{short:#}"
    );

    let missing = MetalCmd3NextNormPlan::new(with_next, None).unwrap_err();
    assert!(
        missing.to_string().contains("no next-norm weights"),
        "{missing:#}"
    );

    let without_next = MetalCmd3PhasePlan::new(
        9,
        3,
        4,
        16,
        4,
        4,
        FlashMoeCmd3OutputState::gpu_resident(16, false),
        false,
    )
    .unwrap();
    let unexpected = MetalCmd3NextNormPlan::new(without_next, Some(16)).unwrap_err();
    assert!(
        unexpected
            .to_string()
            .contains("provided for a no-next-norm phase"),
        "{unexpected:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_active_expert_plan_declares_payload_layout() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
    let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
    let payload = ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(6, 4));

    let plan = MetalCmd3ActiveExpertPlan::new(phase, 1, &payload).unwrap();

    assert_eq!(plan.index, 1);
    assert_eq!(plan.source, MetalCmd3ActiveExpertSource::Q4);
    assert_eq!(plan.intermediate, 6);
    assert_eq!(plan.intermediate_u32().unwrap(), 6);
    assert_eq!(plan.activation_bytes().unwrap(), 6 * 4);
    assert_eq!(plan.projection_output_bytes().unwrap(), 6 * 4);
    assert_eq!(plan.output_offset, 4 * 4);
    assert_eq!(
        plan.buffer_layout().unwrap(),
        MetalCmd3ActiveExpertBufferLayout {
            intermediate_u32: 6,
            activation_bytes: 6 * 4,
            projection_output_bytes: Some(6 * 4),
        }
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_active_expert_plan_rejects_mismatched_payload() {
    let output_state = FlashMoeCmd3OutputState::gpu_resident(4, false);
    let phase = MetalCmd3PhasePlan::new(9, 3, 2, 4, 2, 2, output_state, false).unwrap();
    let payload = ScheduledExpertPhaseMlpPayload::Q4(test_q4_expert_payload(6, 5));

    let err = MetalCmd3ActiveExpertPlan::new(phase, 0, &payload).unwrap_err();

    assert!(
        err.to_string().contains("does not match phase width 4"),
        "{err:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_shared_phase_plan_declares_dense_shape() {
    let shared = SharedExpertPhaseWeights::new(
        Arc::new(vec![1.0; 24]),
        Arc::new(vec![2.0; 24]),
        Arc::new(vec![3.0; 24]),
        Arc::new(vec![4.0; 8]),
        2,
        3,
        4,
    )
    .unwrap();

    let plan = MetalCmd3SharedPhasePlan::dense(4, &shared).unwrap();

    assert_eq!(plan.source, MetalCmd3SharedPhaseSource::Dense);
    assert_eq!(plan.width, 4);
    assert_eq!(plan.shared_experts, 2);
    assert_eq!(plan.intermediate, 3);
    assert_eq!(plan.total_intermediate, 6);
    assert_eq!(plan.total_intermediate_u32().unwrap(), 6);
    assert_eq!(plan.intermediate_u32().unwrap(), 3);
    assert_eq!(plan.projection_output_bytes().unwrap(), 6 * 4);
    assert_eq!(plan.router_output_bytes().unwrap(), 2 * 4);
    assert_eq!(plan.projection_rows(), 6);
    assert_eq!(plan.router_rows(), 2);
    assert_eq!(plan.activation_dispatch_threads(), 6);
    assert_eq!(
        plan.buffer_layout().unwrap(),
        MetalCmd3SharedBufferLayout {
            total_intermediate_u32: 6,
            intermediate_u32: 3,
            projection_output_bytes: 6 * 4,
            router_output_bytes: 2 * 4,
        }
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_shared_phase_plan_declares_resident_shape() {
    let shared = SharedExpertPhaseResidentProjections {
        gate: test_resident_q4_projection("gate", 6, 4),
        up: test_resident_q4_projection("up", 6, 4),
        down: test_resident_q4_projection("down", 4, 6),
        router: Some(test_resident_q4_projection("router", 2, 4)),
        shared_experts: 2,
        intermediate: 3,
        width: 4,
    };

    let plan = MetalCmd3SharedPhasePlan::resident(4, &shared).unwrap();

    assert_eq!(plan.source, MetalCmd3SharedPhaseSource::Resident);
    assert_eq!(plan.width, 4);
    assert_eq!(plan.shared_experts, 2);
    assert_eq!(plan.intermediate, 3);
    assert_eq!(plan.total_intermediate, 6);
    assert_eq!(plan.total_intermediate_u32().unwrap(), 6);
    assert_eq!(plan.intermediate_u32().unwrap(), 3);
    assert_eq!(plan.projection_output_bytes().unwrap(), 6 * 4);
    assert_eq!(plan.router_output_bytes().unwrap(), 2 * 4);
    assert_eq!(plan.projection_rows(), 6);
    assert_eq!(plan.router_rows(), 2);
    assert_eq!(plan.activation_dispatch_threads(), 6);
    assert_eq!(
        plan.buffer_layout().unwrap(),
        MetalCmd3SharedBufferLayout {
            total_intermediate_u32: 6,
            intermediate_u32: 3,
            projection_output_bytes: 6 * 4,
            router_output_bytes: 2 * 4,
        }
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_shared_work_buffers_carry_declared_layout() {
    let shared = SharedExpertPhaseResidentProjections {
        gate: test_resident_q4_projection("gate", 6, 4),
        up: test_resident_q4_projection("up", 6, 4),
        down: test_resident_q4_projection("down", 4, 6),
        router: Some(test_resident_q4_projection("router", 2, 4)),
        shared_experts: 2,
        intermediate: 3,
        width: 4,
    };
    let plan = MetalCmd3SharedPhasePlan::resident(4, &shared).unwrap();

    let buffers = MetalCmd3SharedWorkBuffers::new(
        plan,
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
        0x3000usize as MetalObjcId,
        0x4000usize as MetalObjcId,
        0x5000usize as MetalObjcId,
        0x6000usize as MetalObjcId,
    )
    .unwrap();

    assert_eq!(buffers.layout.total_intermediate_u32, 6);
    assert_eq!(buffers.layout.intermediate_u32, 3);
    assert_eq!(buffers.layout.projection_output_bytes, 6 * 4);
    assert_eq!(buffers.layout.router_output_bytes, 2 * 4);
    assert_eq!(buffers.gate_out, 0x1000usize as MetalObjcId);

    let err = MetalCmd3SharedWorkBuffers::new(
        MetalCmd3SharedPhasePlan::none(4),
        0x1000usize as MetalObjcId,
        0x2000usize as MetalObjcId,
        0x3000usize as MetalObjcId,
        0x4000usize as MetalObjcId,
        0x5000usize as MetalObjcId,
        0x6000usize as MetalObjcId,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("declared shared expert source"),
        "{err:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd3_shared_phase_plan_rejects_width_mismatch() {
    let shared = SharedExpertPhaseWeights::new(
        Arc::new(vec![1.0; 24]),
        Arc::new(vec![2.0; 24]),
        Arc::new(vec![3.0; 24]),
        Arc::new(vec![4.0; 8]),
        2,
        3,
        4,
    )
    .unwrap();

    let err = MetalCmd3SharedPhasePlan::dense(8, &shared).unwrap_err();

    assert!(err.to_string().contains("shared expert width 4"), "{err:#}");

    let huge_shape = SharedExpertPhaseShape::new(1, u32::MAX as usize + 1, 1).unwrap();
    let huge_err =
        MetalCmd3SharedPhasePlan::from_shape(MetalCmd3SharedPhaseSource::Dense, 1, huge_shape)
            .unwrap_err();
    assert!(
        huge_err.to_string().contains("does not fit Metal u32"),
        "{huge_err:#}"
    );

    let none = MetalCmd3SharedPhasePlan::none(4);
    assert_eq!(none.fill_zero_width(), 4);
    assert_eq!(none.activation_dispatch_threads(), 0);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn test_q4_expert_payload(
    intermediate: usize,
    width: usize,
) -> ScheduledQ4ExpertPhaseMlpPayload<'static> {
    let gate = test_q4_matvec_payload(intermediate, width);
    let up = test_q4_matvec_payload(intermediate, width);
    let down = test_q4_matvec_payload(width, intermediate);
    ScheduledQ4ExpertPhaseMlpPayload::new(3, 1, width, gate, up, down).unwrap()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
static TEST_Q4_EXPERT_SLOT: [u8; 4096] = [0; 4096];

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn test_q4_matvec_payload(rows: usize, cols: usize) -> Q4MatvecPayload<'static> {
    let packed_bytes = rows * cols.div_ceil(2);
    let scale_bias_groups = rows * cols.div_ceil(16);
    let scale_bias_bytes = scale_bias_groups * 2;
    Q4MatvecPayload {
        rows,
        cols,
        group_size: 16,
        packed: &TEST_Q4_EXPERT_SLOT[..packed_bytes],
        scales: &[],
        biases: &[],
        scale_bias_groups,
        scale_bias_dtype: "BF16",
        scale_bytes: &TEST_Q4_EXPERT_SLOT[1024..1024 + scale_bias_bytes],
        bias_bytes: &TEST_Q4_EXPERT_SLOT[2048..2048 + scale_bias_bytes],
        source: Some(super::super::experts::Q4MatvecSource {
            bytes: &TEST_Q4_EXPERT_SLOT,
            packed_offset: 0,
            scale_offset: 1024,
            bias_offset: 2048,
            reusable_bytes: None,
        }),
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn test_q4_projection(
    tensor_name: &str,
    output_width: usize,
    input_width: usize,
) -> DenseQ4MmapMatvecProjection {
    DenseQ4MmapMatvecProjection {
        tensor_name: tensor_name.to_string(),
        packed_byte_offset: 0,
        scales_byte_offset: 64,
        biases_byte_offset: 96,
        rows: output_width,
        cols: input_width,
        output_width,
        row_packed_bytes: input_width.div_ceil(2),
        groups_per_row: input_width.div_ceil(16),
        group_size: 16,
        scale_bias_dtype: "F32".to_string(),
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn test_resident_q4_projection(
    tensor_name: &str,
    output_width: usize,
    input_width: usize,
) -> ResidentMmapMatvecProjection {
    test_q4_projection(tensor_name, output_width, input_width).into()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn test_fused_prep_routing_command(
    layer: usize,
    experts: usize,
    routes: &[(usize, f32)],
) -> ScheduledRoutingCommand {
    let stage = FlashMoeStageCapability::new(
        FlashMoeGraphStage::RoutingSoftmaxTopK,
        FlashMoeStagePlacement::CpuDeclared,
        FlashMoeStageImplementation::CpuSoftmaxTopK,
    );
    let routing = ScheduledRoutingTopK {
        stage,
        layer,
        experts,
        active_experts: routes.len(),
        source: ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
    };
    ScheduledRoutingCommand {
        routing,
        layer,
        active_experts: routes.len(),
        source: ScheduledRoutingCandidateSource::FusedMetalPostAttentionPrepCpuTopK,
        routes: routes.to_vec(),
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_linear_attention_state_cache_preserves_gpu_buffer_roles() {
    let base = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
    let state =
        MetalLinearAttentionLayerState::new(base, base, base, base, base, base, 12, 20, 4, 8, 2);
    let cache = MetalLinearAttentionStateCache::new(vec![None, Some(state)]);
    let layer = cache.layers[1].as_ref().unwrap();

    assert_eq!(layer.conv_state, base);
    assert_eq!(layer.ssm_state, base);
    assert_eq!(layer.conv_output, base);
    assert_eq!(layer.delta_output, base);
    assert_eq!(layer.g_decay, base);
    assert_eq!(layer.beta_gate, base);
    assert_eq!(layer.conv_state_len, 12);
    assert_eq!(layer.ssm_state_len, 20);
    assert_eq!(layer.conv_dim, 4);
    assert_eq!(layer.total_value_width, 8);
    assert_eq!(layer.num_value_heads, 2);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_state_buffer_carries_validated_gpu_state_with_raw_binding() {
    let buffer = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
    let hidden = MetalStateBuffer::new(buffer, FlashMoeGpuBufferDescriptor::hidden(8)).unwrap();
    assert_eq!(hidden.buffer(), buffer);
    assert_eq!(hidden.len(), 8);
    assert_eq!(hidden.state(), FlashMoeGpuBufferDescriptor::hidden(8));

    let null_err = MetalStateBuffer::new(
        std::ptr::null_mut(),
        FlashMoeGpuBufferDescriptor::next_layer_normed(8),
    )
    .unwrap_err();
    assert!(
        null_err.to_string().contains("non-null buffer"),
        "{null_err:#}"
    );

    let empty_err =
        MetalStateBuffer::new(buffer, FlashMoeGpuBufferDescriptor::hidden(0)).unwrap_err();
    assert!(
        empty_err.to_string().contains("declared GpuResident state"),
        "{empty_err:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn recurrent_session_snapshot_validates_complete_resident_layer_table() {
    let base = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
    let resident = MetalLinearAttentionStateCache::new(vec![
        None,
        Some(MetalLinearAttentionLayerState::new(
            base, base, base, base, base, base, 2, 3, 4, 5, 2,
        )),
    ]);
    let matching = FlashMoeLinearAttentionSessionSnapshot::new(vec![
        None,
        Some(
            FlashMoeLinearAttentionLayerSnapshot::new(1, vec![1.0; 2], vec![2.0; 3], 4, 5).unwrap(),
        ),
    ])
    .unwrap();
    validate_linear_attention_session_snapshot(&resident, &matching).unwrap();

    let missing = FlashMoeLinearAttentionSessionSnapshot::new(vec![None, None]).unwrap();
    let missing_err = validate_linear_attention_session_snapshot(&resident, &missing).unwrap_err();
    assert!(
        missing_err
            .to_string()
            .contains("missing resolved linear-attention layer 1"),
        "{missing_err:#}"
    );

    let wrong_shape = FlashMoeLinearAttentionSessionSnapshot::new(vec![
        None,
        Some(
            FlashMoeLinearAttentionLayerSnapshot::new(1, vec![1.0; 2], vec![2.0; 4], 4, 5).unwrap(),
        ),
    ])
    .unwrap();
    let shape_err =
        validate_linear_attention_session_snapshot(&resident, &wrong_shape).unwrap_err();
    assert!(
        shape_err
            .to_string()
            .contains("does not match the resolved resident state"),
        "{shape_err:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn metal_nested_autorelease_releases_completed_command_resources() {
    objc2::rc::autoreleasepool(|_| unsafe {
        let physical_footprint = || {
            let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v2>::uninit();
            let result = libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V2,
                usage.as_mut_ptr().cast(),
            );
            assert_eq!(result, 0, "proc_pid_rusage failed");
            usage.assume_init().ri_phys_footprint
        };
        let runtime = MetalRuntime::compile(METAL_SHADERS, MetalPipelineNameSet::new()).unwrap();
        let resources = Arc::new(MetalResourceLedger::default());
        let baseline = msg_send_usize0(runtime.device, sel("currentAllocatedSize"));
        let baseline_footprint = physical_footprint();

        for _ in 0..16 {
            let (command_buffer, command_lease, buffers) = objc2::rc::autoreleasepool(|_| {
                let buffers = (0..30)
                    .map(|_| {
                        OwnedMetalObject::new(msg_send_id2_usize_u64(
                            runtime.device,
                            sel("newBufferWithLength:options:"),
                            1024 * 1024,
                            0,
                        ))
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                for buffer in &buffers {
                    ptr::write_bytes(
                        msg_send_ptr0(buffer.id(), sel("contents")).cast::<u8>(),
                        0xA5,
                        1024 * 1024,
                    );
                }
                let mut encoding = MetalCommandEncoding::new(
                    runtime.command_queue,
                    Arc::clone(&resources),
                    "test command buffer allocation failed",
                    "test command encoder allocation failed",
                )
                .unwrap();
                msg_send_void1_id(
                    encoding.encoder(),
                    sel("setComputePipelineState:"),
                    runtime.pipelines.fill_zero_pipeline,
                );
                set_buffer(encoding.encoder(), buffers[0].id(), 0);
                for (index, buffer) in buffers.iter().enumerate().skip(1) {
                    set_buffer(encoding.encoder(), buffer.id(), index as u64 + 1);
                }
                for width in 1..=64u32 {
                    set_bytes(encoding.encoder(), u32_as_bytes(&width), 1);
                }
                dispatch_threads(encoding.encoder(), 1);
                encoding.end_encoding();
                let (command_buffer, command_lease) = encoding.into_command_buffer();
                commit_metal_command_buffer(
                    command_buffer,
                    &MetalCommandContext::new("nested autorelease test"),
                );
                (command_buffer, command_lease, buffers)
            });
            objc2::rc::autoreleasepool(|_| {
                wait_for_metal_command_buffer(
                    command_buffer,
                    &MetalCommandContext::new("nested autorelease completion test"),
                )
                .unwrap();
                release(command_buffer);
                drop(command_lease);
                for buffer in buffers {
                    purge_and_release_metal_buffer(buffer.into_raw());
                }
            });
        }

        let allocated = msg_send_usize0(runtime.device, sel("currentAllocatedSize"));
        let footprint = physical_footprint();
        assert!(
            allocated.saturating_sub(baseline) < 64 * 1024 * 1024,
            "completed commands retained {} bytes across nested autorelease pools",
            allocated.saturating_sub(baseline)
        );
        assert!(
            footprint.saturating_sub(baseline_footprint) < 192 * 1024 * 1024,
            "completed commands grew physical footprint by {} bytes",
            footprint.saturating_sub(baseline_footprint)
        );
    });
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn resident_q4_multilinear_uses_each_heads_own_input() {
    let heads = 2usize;
    let rows_per_head = 16usize;
    let cols = 8usize;
    let rows = heads * rows_per_head;
    let packed_bytes = rows * cols.div_ceil(2);
    let scalar_bytes = rows * std::mem::size_of::<u16>();
    let scales_offset = packed_bytes;
    let biases_offset = scales_offset + scalar_bytes;
    let mut mmap = memmap2::MmapMut::map_anon(4096).unwrap();
    for row in 0..rows {
        let nibble = (row % 7 + 1) as u8;
        mmap[row * 4..row * 4 + 4].fill(nibble | (nibble << 4));
        mmap[scales_offset + row * 2..scales_offset + row * 2 + 2]
            .copy_from_slice(&0x3f80u16.to_le_bytes());
        mmap[biases_offset + row * 2..biases_offset + row * 2 + 2]
            .copy_from_slice(&0u16.to_le_bytes());
    }
    let mmap = Arc::new(mmap.make_read_only().unwrap());
    let context = MetalExecutionContext::compile(mmap, 4096, &[], 1e-6).unwrap();
    let projection = DenseQ4MmapMatvecProjection {
        tensor_name: "test.multilinear.weight".to_string(),
        packed_byte_offset: 0,
        scales_byte_offset: scales_offset as u64,
        biases_byte_offset: biases_offset as u64,
        rows,
        cols,
        output_width: rows,
        row_packed_bytes: cols.div_ceil(2),
        groups_per_row: 1,
        group_size: cols,
        scale_bias_dtype: "BF16".to_string(),
    };
    let inputs = [vec![1.0; cols], vec![2.0; cols]].concat();
    let actual = context
        .resident_q4_multilinear(&projection, heads, rows_per_head, &inputs)
        .unwrap()
        .unwrap();
    let expected = (0..rows)
        .map(|row| {
            let nibble = (row % 7 + 1) as f32;
            let input = if row < rows_per_head { 1.0 } else { 2.0 };
            nibble * cols as f32 * input
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn resident_glm_mla_absorbed_attention_matches_reference() {
    fn write_projection(
        mmap: &mut [u8],
        offset: &mut usize,
        name: &str,
        rows: usize,
        cols: usize,
        nibble: impl Fn(usize) -> u8,
    ) -> DenseQ4MmapMatvecProjection {
        let packed_byte_offset = *offset;
        let row_packed_bytes = cols.div_ceil(2);
        for row in 0..rows {
            let value = nibble(row) & 0x0f;
            mmap[packed_byte_offset + row * row_packed_bytes
                ..packed_byte_offset + (row + 1) * row_packed_bytes]
                .fill(value | (value << 4));
        }
        let scales_byte_offset = packed_byte_offset + rows * row_packed_bytes;
        for row in 0..rows {
            mmap[scales_byte_offset + row * 2..scales_byte_offset + row * 2 + 2]
                .copy_from_slice(&0x3f80u16.to_le_bytes());
        }
        let biases_byte_offset = scales_byte_offset + rows * 2;
        mmap[biases_byte_offset..biases_byte_offset + rows * 2].fill(0);
        *offset = biases_byte_offset + rows * 2;
        DenseQ4MmapMatvecProjection {
            tensor_name: name.to_string(),
            packed_byte_offset: packed_byte_offset as u64,
            scales_byte_offset: scales_byte_offset as u64,
            biases_byte_offset: biases_byte_offset as u64,
            rows,
            cols,
            output_width: rows,
            row_packed_bytes,
            groups_per_row: 1,
            group_size: cols,
            scale_bias_dtype: "BF16".to_string(),
        }
    }

    let heads = 2usize;
    let latent_rank = 16usize;
    let nope_dim = 8usize;
    let rope_dim = 4usize;
    let output_per_head = 16usize;
    let sequence = 2usize;
    let mut mmap = memmap2::MmapMut::map_anon(64 * 1024).unwrap();
    let mut offset = 0usize;
    let embed_q = write_projection(
        &mut mmap,
        &mut offset,
        "test.embed_q",
        heads * latent_rank,
        nope_dim,
        |row| (row % 3 + 1) as u8,
    );
    let unembed_out = write_projection(
        &mut mmap,
        &mut offset,
        "test.unembed_out",
        heads * output_per_head,
        latent_rank,
        |row| (row % 5 + 1) as u8,
    );
    let mmap = Arc::new(mmap.make_read_only().unwrap());
    let context = MetalExecutionContext::compile(mmap, 64 * 1024, &[], 1e-6).unwrap();
    let query_nope = [vec![1.0; nope_dim], vec![2.0; nope_dim]].concat();
    let query_rope = vec![0.25, -0.5, 0.75, 1.0, -0.25, 0.5, -0.75, 1.5];
    let record_latents = (0..sequence)
        .flat_map(|position| {
            (0..latent_rank).map(move |dim| {
                if position == 0 {
                    (dim + 1) as f32 / latent_rank as f32
                } else {
                    (latent_rank - dim) as f32 / latent_rank as f32
                }
            })
        })
        .collect::<Vec<_>>();
    let record_rotary = vec![0.5, 0.25, -0.5, 1.0, -0.25, 0.75, 0.5, -1.0];
    let scale = (nope_dim + rope_dim) as f32;
    let scale = scale.sqrt().recip();

    let actual = context
        .resident_glm_mla_absorbed_attention(
            &embed_q,
            &unembed_out,
            MetalGlmMlaAbsorbedAttentionInput {
                heads,
                latent_rank,
                query_nope: &query_nope,
                query_rope: &query_rope,
                record_latents: &record_latents,
                record_rotary: &record_rotary,
                sequence,
                rope_dim,
                scale,
            },
        )
        .unwrap()
        .unwrap();

    let mut expected = Vec::with_capacity(heads * output_per_head);
    for head in 0..heads {
        let query_sum = query_nope[head * nope_dim..(head + 1) * nope_dim]
            .iter()
            .sum::<f32>();
        let absorbed = (0..latent_rank)
            .map(|dim| ((head * latent_rank + dim) % 3 + 1) as f32 * query_sum)
            .collect::<Vec<_>>();
        let mut scores = (0..sequence)
            .map(|position| {
                let latent = &record_latents[position * latent_rank..(position + 1) * latent_rank];
                let rotary = &record_rotary[position * rope_dim..(position + 1) * rope_dim];
                let latent_score = absorbed
                    .iter()
                    .zip(latent)
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                let rotary_score = query_rope[head * rope_dim..(head + 1) * rope_dim]
                    .iter()
                    .zip(rotary)
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                (latent_score + rotary_score) * scale
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f32>();
        for score in &mut scores {
            *score = (*score - maximum).exp() / denominator;
        }
        let context_values = (0..latent_rank)
            .map(|dim| {
                (0..sequence)
                    .map(|position| scores[position] * record_latents[position * latent_rank + dim])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let context_sum = context_values.iter().sum::<f32>();
        expected.extend(
            (0..output_per_head)
                .map(|row| ((head * output_per_head + row) % 5 + 1) as f32 * context_sum),
        );
    }
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-3,
            "output {index}: actual={actual} expected={expected}"
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn resident_glm_mla_input_projection_chain_matches_reference() {
    fn write_q4_projection(
        mmap: &mut [u8],
        offset: &mut usize,
        name: &str,
        rows: usize,
        cols: usize,
        nibble: impl Fn(usize) -> u8,
    ) -> ResidentMmapMatvecProjection {
        let packed_byte_offset = *offset;
        let packed_bytes = rows * cols.div_ceil(2);
        for row in 0..rows {
            let value = nibble(row) & 0x0f;
            mmap[packed_byte_offset + row * cols.div_ceil(2)
                ..packed_byte_offset + (row + 1) * cols.div_ceil(2)]
                .fill(value | (value << 4));
        }
        let scales_byte_offset = packed_byte_offset + packed_bytes;
        for row in 0..rows {
            mmap[scales_byte_offset + row * 2..scales_byte_offset + row * 2 + 2]
                .copy_from_slice(&0x3f80u16.to_le_bytes());
        }
        let biases_byte_offset = scales_byte_offset + rows * 2;
        mmap[biases_byte_offset..biases_byte_offset + rows * 2].fill(0);
        *offset = biases_byte_offset + rows * 2;
        ResidentMmapMatvecProjection::Q4(DenseQ4MmapMatvecProjection {
            tensor_name: name.to_string(),
            packed_byte_offset: packed_byte_offset as u64,
            scales_byte_offset: scales_byte_offset as u64,
            biases_byte_offset: biases_byte_offset as u64,
            rows,
            cols,
            output_width: rows,
            row_packed_bytes: cols.div_ceil(2),
            groups_per_row: 1,
            group_size: cols,
            scale_bias_dtype: "BF16".to_string(),
        })
    }

    let input = (1..=8).map(|value| value as f32).collect::<Vec<_>>();
    let mut mmap = memmap2::MmapMut::map_anon(16 * 1024).unwrap();
    let mut offset = 0usize;
    let q_a = write_q4_projection(&mut mmap, &mut offset, "q_a", 16, 8, |row| {
        (row % 7 + 1) as u8
    });
    let kv_a = write_q4_projection(&mut mmap, &mut offset, "kv_a", 16, 8, |row| {
        (row % 5 + 1) as u8
    });
    let q_b = write_q4_projection(&mut mmap, &mut offset, "q_b", 16, 16, |_| 1);
    let mmap = Arc::new(mmap.make_read_only().unwrap());
    let context = MetalExecutionContext::compile(mmap, 16 * 1024, &[], 1e-6).unwrap();
    let q_norm = vec![1.0; 16];
    let kv_norm = vec![1.0; 8];
    let (query, compressed) = context
        .resident_glm_mla_input_projection_chain(
            &q_a,
            &kv_a,
            &q_b,
            MetalBatchProjectionInput::Cpu(&input),
            &q_norm,
            &kv_norm,
            8,
            1e-6,
        )
        .unwrap()
        .unwrap();

    let input_sum = input.iter().sum::<f32>();
    let mut q_a_reference = (0..16)
        .map(|row| (row % 7 + 1) as f32 * input_sum)
        .collect::<Vec<_>>();
    let q_scale = (q_a_reference.iter().map(|value| value * value).sum::<f32>()
        / q_a_reference.len() as f32
        + 1e-6)
        .sqrt()
        .recip();
    for value in &mut q_a_reference {
        *value *= q_scale;
    }
    let expected_query = vec![q_a_reference.iter().sum::<f32>(); 16];
    for (actual, expected) in query.iter().zip(expected_query) {
        assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
    }

    let mut expected_compressed = (0..16)
        .map(|row| (row % 5 + 1) as f32 * input_sum)
        .collect::<Vec<_>>();
    let kv_scale = (expected_compressed[..8]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        / 8.0
        + 1e-6)
        .sqrt()
        .recip();
    for value in &mut expected_compressed[..8] {
        *value *= kv_scale;
    }
    for (actual, expected) in compressed.iter().zip(expected_compressed) {
        assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn resident_glm_mla_fused_attention_matches_two_command_reference() {
    fn write_q4_projection(
        mmap: &mut [u8],
        offset: &mut usize,
        name: &str,
        rows: usize,
        cols: usize,
        nibble: impl Fn(usize) -> u8,
    ) -> DenseQ4MmapMatvecProjection {
        let packed_byte_offset = *offset;
        let packed_bytes = rows * cols.div_ceil(2);
        for row in 0..rows {
            let value = nibble(row) & 0x0f;
            mmap[packed_byte_offset + row * cols.div_ceil(2)
                ..packed_byte_offset + (row + 1) * cols.div_ceil(2)]
                .fill(value | (value << 4));
        }
        let scales_byte_offset = packed_byte_offset + packed_bytes;
        for row in 0..rows {
            mmap[scales_byte_offset + row * 2..scales_byte_offset + row * 2 + 2]
                .copy_from_slice(&0x3f80u16.to_le_bytes());
        }
        let biases_byte_offset = scales_byte_offset + rows * 2;
        mmap[biases_byte_offset..biases_byte_offset + rows * 2].fill(0);
        *offset = biases_byte_offset + rows * 2;
        DenseQ4MmapMatvecProjection {
            tensor_name: name.to_string(),
            packed_byte_offset: packed_byte_offset as u64,
            scales_byte_offset: scales_byte_offset as u64,
            biases_byte_offset: biases_byte_offset as u64,
            rows,
            cols,
            output_width: rows,
            row_packed_bytes: cols.div_ceil(2),
            groups_per_row: 1,
            group_size: cols,
            scale_bias_dtype: "BF16".to_string(),
        }
    }

    let input = (1..=8).map(|value| value as f32).collect::<Vec<_>>();
    let heads = 2usize;
    let q_rank = 16usize;
    let latent_rank = 16usize;
    let nope_dim = 8usize;
    let rope_dim = 4usize;
    let output_per_head = 16usize;
    let mut mmap = memmap2::MmapMut::map_anon(64 * 1024).unwrap();
    let mut offset = 0usize;
    let q_a_q4 = write_q4_projection(&mut mmap, &mut offset, "q_a", q_rank, 8, |row| {
        (row % 7 + 1) as u8
    });
    let kv_a_q4 = write_q4_projection(
        &mut mmap,
        &mut offset,
        "kv_a",
        latent_rank + rope_dim,
        8,
        |row| (row % 5 + 1) as u8,
    );
    let q_b_q4 = write_q4_projection(
        &mut mmap,
        &mut offset,
        "q_b",
        heads * (nope_dim + rope_dim),
        q_rank,
        |row| (row % 3 + 1) as u8,
    );
    let embed_q = write_q4_projection(
        &mut mmap,
        &mut offset,
        "embed_q",
        heads * latent_rank,
        nope_dim,
        |row| (row % 3 + 1) as u8,
    );
    let unembed_out = write_q4_projection(
        &mut mmap,
        &mut offset,
        "unembed_out",
        heads * output_per_head,
        latent_rank,
        |row| (row % 5 + 1) as u8,
    );
    let out_proj = ResidentMmapMatvecProjection::Q4(write_q4_projection(
        &mut mmap,
        &mut offset,
        "o_proj",
        16,
        heads * output_per_head,
        |row| (row % 3 + 1) as u8,
    ));
    let router = ResidentMmapMatvecProjection::Q4(write_q4_projection(
        &mut mmap,
        &mut offset,
        "router",
        4,
        16,
        |row| (row % 4 + 1) as u8,
    ));
    let post_projections = Cmd2ResidentPostAttentionPrepProjections::new(
        3,
        out_proj,
        router,
        4,
        16,
        heads * output_per_head,
        2,
    )
    .unwrap();
    let q_a = ResidentMmapMatvecProjection::Q4(q_a_q4);
    let kv_a = ResidentMmapMatvecProjection::Q4(kv_a_q4);
    let q_b = ResidentMmapMatvecProjection::Q4(q_b_q4);
    let mmap = Arc::new(mmap.make_read_only().unwrap());
    let context = MetalExecutionContext::compile(mmap, 64 * 1024, &[], 1e-6).unwrap();
    let q_norm = vec![1.0; q_rank];
    let kv_norm = vec![1.0; latent_rank];
    let previous_latent = (0..latent_rank)
        .map(|dim| (dim + 1) as f32 / latent_rank as f32)
        .collect::<Vec<_>>();
    let previous_rotary = vec![0.5, -0.25, 0.75, -1.0];
    let rope_position = 3usize;
    let theta = 10_000.0f64;
    let mut rope_cos = Vec::new();
    let mut rope_sin = Vec::new();
    for pair in 0..rope_dim / 2 {
        let frequency = theta.powf(-((2 * pair) as f64) / rope_dim as f64);
        let (sin, cos) = (rope_position as f64 * frequency).sin_cos();
        rope_cos.push(cos as f32);
        rope_sin.push(sin as f32);
    }

    let (mut query, mut compressed) = context
        .resident_glm_mla_input_projection_chain(
            &q_a,
            &kv_a,
            &q_b,
            MetalBatchProjectionInput::Cpu(&input),
            &q_norm,
            &kv_norm,
            latent_rank,
            1e-6,
        )
        .unwrap()
        .unwrap();
    for head in 0..heads {
        let start = head * (nope_dim + rope_dim) + nope_dim;
        super::super::math::apply_rotary_interleaved_to_split_half(
            &mut query[start..start + rope_dim],
            rope_position,
            rope_dim,
            theta,
        )
        .unwrap();
    }
    let mut current_rotary = compressed.split_off(latent_rank);
    super::super::math::apply_rotary_interleaved_to_split_half(
        &mut current_rotary,
        rope_position,
        rope_dim,
        theta,
    )
    .unwrap();
    let mut query_nope = Vec::new();
    let mut query_rope = Vec::new();
    for head in 0..heads {
        let start = head * (nope_dim + rope_dim);
        query_nope.extend_from_slice(&query[start..start + nope_dim]);
        query_rope.extend_from_slice(&query[start + nope_dim..start + nope_dim + rope_dim]);
    }
    let record_latents = [previous_latent.clone(), compressed.clone()].concat();
    let record_rotary = [previous_rotary.clone(), current_rotary.clone()].concat();
    let scale = ((nope_dim + rope_dim) as f32).sqrt().recip();
    let reference = context
        .resident_glm_mla_absorbed_attention(
            &embed_q,
            &unembed_out,
            MetalGlmMlaAbsorbedAttentionInput {
                heads,
                latent_rank,
                query_nope: &query_nope,
                query_rope: &query_rope,
                record_latents: &record_latents,
                record_rotary: &record_rotary,
                sequence: 2,
                rope_dim,
                scale,
            },
        )
        .unwrap()
        .unwrap();
    let fused = context
        .resident_glm_mla_fused_attention(
            &q_a,
            &kv_a,
            &q_b,
            &embed_q,
            &unembed_out,
            MetalGlmMlaFusedAttentionInput {
                input: MetalBatchProjectionInput::Cpu(&input),
                heads,
                latent_rank,
                nope_dim,
                rope_dim,
                previous_record_latents: &previous_latent,
                previous_record_rotary: &previous_rotary,
                rope_cos: &rope_cos,
                rope_sin: &rope_sin,
                scale,
                post_attention: None,
            },
            &q_norm,
            &kv_norm,
            1e-6,
        )
        .unwrap()
        .unwrap();

    let MetalGlmMlaFusedAttentionTerminal::Attention(fused_attention) = &fused.terminal else {
        panic!("reference-only fused MLA test unexpectedly produced post-attention state")
    };
    for (index, (actual, expected)) in fused_attention.iter().zip(&reference).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-3,
            "attention {index}: actual={actual} expected={expected}"
        );
    }
    for (actual, expected) in fused.latent.iter().zip(&compressed) {
        assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
    }
    for (actual, expected) in fused.rotary.iter().zip(&current_rotary) {
        assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
    }

    let residual = (0..16)
        .map(|index| (index as f32 - 7.0) / 8.0)
        .collect::<Vec<_>>();
    let post_norm = vec![1.0; 16];
    let correction_bias = vec![0.0, 0.2, -0.1, 0.1];
    let reference_post = context
        .resident_post_attention_prep_topk(
            &post_projections,
            &reference,
            MetalBatchProjectionInput::Cpu(&residual),
            &post_norm,
            Some(&correction_bias),
        )
        .unwrap();
    let fused_post = context
        .resident_glm_mla_fused_attention(
            &q_a,
            &kv_a,
            &q_b,
            &embed_q,
            &unembed_out,
            MetalGlmMlaFusedAttentionInput {
                input: MetalBatchProjectionInput::Cpu(&input),
                heads,
                latent_rank,
                nope_dim,
                rope_dim,
                previous_record_latents: &previous_latent,
                previous_record_rotary: &previous_rotary,
                rope_cos: &rope_cos,
                rope_sin: &rope_sin,
                scale,
                post_attention: Some(MetalGlmMlaPostAttentionInput {
                    projections: &post_projections,
                    residual: MetalBatchProjectionInput::Cpu(&residual),
                    post_norm_weight: &post_norm,
                    router_correction_bias: Some(&correction_bias),
                }),
            },
            &q_norm,
            &kv_norm,
            1e-6,
        )
        .unwrap()
        .unwrap();
    let MetalGlmMlaFusedAttentionTerminal::PostAttention(fused_post) = fused_post.terminal else {
        panic!("fused MLA post-attention test did not produce post-attention state")
    };
    assert_eq!(fused_post.active, reference_post.active);
    unsafe {
        let reference_residual = read_f32_buffer(reference_post.residual_buffer, 16);
        let fused_residual = read_f32_buffer(fused_post.residual_buffer, 16);
        let reference_normed = read_f32_buffer(reference_post.normed_buffer, 16);
        let fused_normed = read_f32_buffer(fused_post.normed_buffer, 16);
        for (actual, expected) in fused_residual.iter().zip(reference_residual) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
        for (actual, expected) in fused_normed.iter().zip(reference_normed) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
        for buffer in [
            reference_post.residual_buffer,
            reference_post.normed_buffer,
            fused_post.residual_buffer,
            fused_post.normed_buffer,
        ] {
            context.buffers.recycle(buffer);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn recurrent_session_snapshot_round_trips_metal_recurrent_buffers() {
    let device = unsafe { OwnedMetalObject::new(metal_default_device()).unwrap() };
    let allocate = |len: usize, label: &str| {
        OwnedMetalObject::new(allocate_zeroed_buffer(device.id(), len * 4, label).unwrap()).unwrap()
    };
    let conv_state = allocate(2, "test recurrent conv state");
    let ssm_state = allocate(3, "test recurrent SSM state");
    let conv_output = allocate(4, "test recurrent conv output");
    let delta_output = allocate(5, "test recurrent delta output");
    let g_decay = allocate(2, "test recurrent decay");
    let beta_gate = allocate(2, "test recurrent beta");
    let resident =
        MetalLinearAttentionStateCache::new(vec![Some(MetalLinearAttentionLayerState::new(
            conv_state.id(),
            ssm_state.id(),
            conv_output.id(),
            delta_output.id(),
            g_decay.id(),
            beta_gate.id(),
            2,
            3,
            4,
            5,
            2,
        ))]);
    unsafe {
        write_f32_buffer(conv_state.id(), &[1.0, 2.0]);
        write_f32_buffer(ssm_state.id(), &[3.0, 4.0, 5.0]);
        write_f32_buffer(conv_output.id(), &[6.0; 4]);
        write_f32_buffer(delta_output.id(), &[7.0; 5]);
        write_f32_buffer(g_decay.id(), &[8.0; 2]);
        write_f32_buffer(beta_gate.id(), &[9.0; 2]);
    }

    let snapshot = capture_linear_attention_session_snapshot(&resident).unwrap();
    unsafe {
        write_f32_buffer(conv_state.id(), &[10.0; 2]);
        write_f32_buffer(ssm_state.id(), &[11.0; 3]);
    }
    restore_linear_attention_session_snapshot(&resident, &snapshot).unwrap();

    unsafe {
        assert_eq!(read_f32_buffer(conv_state.id(), 2), vec![1.0, 2.0]);
        assert_eq!(read_f32_buffer(ssm_state.id(), 3), vec![3.0, 4.0, 5.0]);
        assert_eq!(read_f32_buffer(conv_output.id(), 4), vec![0.0; 4]);
        assert_eq!(read_f32_buffer(delta_output.id(), 5), vec![0.0; 5]);
        assert_eq!(read_f32_buffer(g_decay.id(), 2), vec![0.0; 2]);
        assert_eq!(read_f32_buffer(beta_gate.id(), 2), vec![0.0; 2]);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn reusable_metal_buffer_selection_is_best_fit_and_size_aware() {
    let id = std::ptr::null_mut();
    let buffers = vec![
        MetalReusableBuffer::new(id, 2_654_208),
        MetalReusableBuffer::new(id, 16),
        MetalReusableBuffer::new(id, 4_096),
        MetalReusableBuffer::new(id, 8_192),
    ];

    assert_eq!(best_fit_reusable_buffer_index(&buffers, 1), Some(1));
    assert_eq!(best_fit_reusable_buffer_index(&buffers, 64), Some(2));
    assert_eq!(best_fit_reusable_buffer_index(&buffers, 2_654_208), Some(0));
    assert_eq!(best_fit_reusable_buffer_index(&buffers, 2_654_209), None);
    assert_eq!(reusable_buffer_replacement_index(&buffers, 32), Some(1));
    assert_eq!(reusable_buffer_replacement_index(&buffers, 16), None);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_dispatch_plans_preserve_command_geometry() {
    assert_eq!(
        MetalDispatchPlan::threads(96),
        MetalDispatchPlan {
            mode: MetalDispatchMode::Threads,
            grid: MetalDispatchSize::new(96, 1, 1),
            threadgroup: MetalDispatchSize::new(64, 1, 1),
        }
    );
    assert_eq!(
        MetalDispatchPlan::q4_threadgroups(17),
        MetalDispatchPlan {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(3, 1, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    );
    assert_eq!(
        MetalDispatchPlan::q4_mmap_threadgroups(17),
        MetalDispatchPlan {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(2, 1, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    );
    assert_eq!(
        MetalDispatchPlan::q4_mmap_matrix_threadgroups(17, 64),
        MetalDispatchPlan {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(2, 64, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    );
    assert_eq!(
        MetalDispatchPlan::q4_mmap_matrix_bf16_threadgroups(17, 65, 2),
        MetalDispatchPlan {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(2, 33, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    );
    assert_eq!(
        MetalDispatchPlan::qwen_attention_threadgroups(64, 16),
        MetalDispatchPlan {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(64, 16, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    );
    assert_eq!(
        MetalDispatchPlan::single_threadgroup(512),
        MetalDispatchPlan {
            mode: MetalDispatchMode::Threadgroups,
            grid: MetalDispatchSize::new(1, 1, 1),
            threadgroup: MetalDispatchSize::new(256, 1, 1),
        }
    );
    assert_eq!(MetalDispatchPlan::q4_threadgroups(0).grid.width, 1);
    assert_eq!(
        MetalDispatchPlan::q4_mmap_matrix_threadgroups(0, 0).grid,
        MetalDispatchSize::new(1, 1, 1)
    );
    assert_eq!(
        MetalDispatchPlan::q4_mmap_matrix_bf16_threadgroups(0, 0, 0).grid,
        MetalDispatchSize::new(1, 1, 1)
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_dense_weights_hold_buffer_len_and_mmap_owner() {
    let mmap = Arc::new(
        memmap2::MmapMut::map_anon(16)
            .unwrap()
            .make_read_only()
            .unwrap(),
    );
    let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
    let dense = MetalDenseWeights::new(id, Arc::clone(&mmap), 16);

    assert_eq!(dense.buffer, id);
    assert_eq!(dense.len, 16);
    assert_eq!(Arc::strong_count(&mmap), 2);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn resident_topk_builder_rejects_invalid_bindings_before_encoding() {
    let mmap = Arc::new(
        memmap2::MmapMut::map_anon(128)
            .unwrap()
            .make_read_only()
            .unwrap(),
    );
    let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
    let dense = MetalDenseWeights::new(id, mmap, 128);
    let buffers = MetalBufferPool::default();
    let pipelines = test_objc_pipeline_set(id);
    let builder = MetalResidentTopKBuilder::new(id, id, &pipelines, &dense, &buffers);

    let projection = ResidentMmapMatvecProjection::Q4(test_q4_projection("lm_head.weight", 4, 16));
    let input_error = builder.execute(&projection, &[0.0; 15], 4, 2).unwrap_err();
    assert!(
        input_error.to_string().contains("input len 15"),
        "{input_error:#}"
    );

    let mut unsupported_dtype = test_q4_projection("lm_head.weight", 4, 16);
    unsupported_dtype.scale_bias_dtype = "F16".to_string();
    let dtype_error = builder
        .execute(
            &ResidentMmapMatvecProjection::Q4(unsupported_dtype),
            &[0.0; 16],
            4,
            2,
        )
        .unwrap_err();
    assert!(
        dtype_error.to_string().contains("scale/bias dtype F16"),
        "{dtype_error:#}"
    );

    let mut out_of_range = test_q4_projection("lm_head.weight", 4, 16);
    out_of_range.biases_byte_offset = 124;
    let range_error = builder
        .execute(
            &ResidentMmapMatvecProjection::Q4(out_of_range),
            &[0.0; 16],
            4,
            2,
        )
        .unwrap_err();
    assert!(
        range_error
            .to_string()
            .contains("biases range for lm_head.weight exceeds resident dense weights"),
        "{range_error:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cmd2_resident_builder_rejects_state_width_mismatch_before_encoding() {
    let mmap = Arc::new(
        memmap2::MmapMut::map_anon(4096)
            .unwrap()
            .make_read_only()
            .unwrap(),
    );
    let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
    let dense = MetalDenseWeights::new(id, mmap, 4096);
    let buffers = MetalBufferPool::default();
    let pipelines = test_objc_pipeline_set(id);
    let builder =
        MetalResidentPostAttentionPrepBuilder::new(id, id, &pipelines, &dense, &buffers, 1e-6);
    let projections = Cmd2ResidentPostAttentionPrepProjections::new(
        7,
        ResidentMmapMatvecProjection::Q4(test_q4_projection(
            "model.layers.7.self_attn.o_proj.weight",
            4,
            16,
        )),
        ResidentMmapMatvecProjection::Q4(test_q4_projection(
            "model.layers.7.mlp.gate.weight",
            8,
            4,
        )),
        8,
        4,
        16,
        4,
    )
    .unwrap();

    let error = builder
        .execute(
            &projections,
            &[0.0; 15],
            MetalBatchProjectionInput::Cpu(&[0.0; 4]),
            &[1.0; 4],
            None,
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("projection shapes out=4x16 rows=4"),
        "{error:#}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_reusable_buffer_records_pool_entry_shape() {
    let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();
    let buffer = MetalReusableBuffer::new(id, 4096);

    assert_eq!(buffer.id, id);
    assert_eq!(buffer.len, 4096);
    assert_eq!(METAL_REUSABLE_BUFFER_POOL_LIMIT, 64);
    assert_eq!(METAL_REUSABLE_EXPERT_STAGING_POOL_LIMIT, 16);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn metal_expert_staging_pool_reuses_completed_slot_allocation() {
    objc2::rc::autoreleasepool(|_| unsafe {
        let runtime = MetalRuntime::compile(METAL_SHADERS, MetalPipelineNameSet::new()).unwrap();
        let resources = Arc::new(MetalResourceLedger::from_device(runtime.device));
        let buffers = MetalBufferPool::new(Arc::clone(&resources));
        let bytes = vec![0x5au8; 64 * 1024];

        let first = buffers
            .transient_expert_buffer_with_bytes(runtime.device, &bytes)
            .unwrap();
        buffers.recycle_or_release_phase(vec![MetalPhaseBuffer::transient_expert(first)], false);
        let pooled = resources.snapshot();
        assert_eq!(pooled.transient_expert_buffers, 0);
        assert_eq!(pooled.pooled_buffers, 1);

        let second = buffers
            .transient_expert_buffer_with_bytes(runtime.device, &bytes)
            .unwrap();
        assert_eq!(second, first);
        let checked_out = resources.snapshot();
        assert_eq!(checked_out.transient_expert_buffers, 1);
        assert_eq!(checked_out.pooled_buffers, 0);

        buffers.recycle_or_release_phase(vec![MetalPhaseBuffer::transient_expert(second)], true);
        let released = resources.snapshot();
        assert_eq!(released.transient_expert_buffers, 0);
        assert_eq!(released.pooled_buffers, 0);
    });
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_expert_source_buffer_cache_reuses_same_fixed_payload_key() {
    let first = [1u8; 16];
    let second = [2u8; 16];
    let buffer = 0x1000usize as MetalObjcId;
    let mut cache = MetalExpertSourceBufferCache::default();

    cache.insert(&first, buffer);

    assert_eq!(cache.get(&first), Some(buffer));
    assert_eq!(cache.get(&second), None);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn phase_buffer_tracks_recycling_class() {
    let id = std::ptr::NonNull::<std::ffi::c_void>::dangling().as_ptr();

    let recyclable = MetalPhaseBuffer::recyclable(id);
    assert_eq!(recyclable.id, id);
    assert_eq!(recyclable.class, MetalPhaseBufferClass::General);

    let expert = MetalPhaseBuffer::transient_expert(id);
    assert_eq!(expert.id, id);
    assert_eq!(expert.class, MetalPhaseBufferClass::TransientExpert);

    let borrowed = MetalPhaseBuffer::borrowed_expert(id);
    assert_eq!(borrowed.id, id);
    assert_eq!(borrowed.class, MetalPhaseBufferClass::BorrowedExpert);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn metal_wraps_page_aligned_expert_slot_without_copying() {
    objc2::rc::autoreleasepool(|_| unsafe {
        let device = metal_default_device();
        assert!(!device.is_null());
        let page_size = metal_page_size();
        let mut bytes = memmap2::MmapMut::map_anon(page_size).unwrap();
        bytes.fill(0x5a);

        let buffer = wrap_expert_slot_as_metal_buffer(device, &bytes).unwrap();

        assert_eq!(
            msg_send_ptr0(buffer, sel("contents")).cast::<u8>(),
            bytes.as_mut_ptr()
        );
        assert_eq!(msg_send_usize0(buffer, sel("length")), page_size);
        release(buffer);
        release(device);
    });
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a local Metal device"]
fn metal_reuses_wrapper_attached_to_page_aligned_expert_slot() {
    objc2::rc::autoreleasepool(|_| unsafe {
        let device = metal_default_device();
        assert!(!device.is_null());
        let resources = Arc::new(MetalResourceLedger::from_device(device));
        let buffers = MetalBufferPool::new(Arc::clone(&resources));
        let page_size = metal_page_size();
        let mut scratch = super::super::experts::ReusableExpertBuffer::default();
        scratch
            .prepare_payload(page_size, page_size)
            .unwrap()
            .fill(0x5a);
        let bytes = scratch.take_payload();

        let first = persistent_expert_source_buffer(device, &bytes, &bytes, &buffers)
            .unwrap()
            .unwrap();
        let second = persistent_expert_source_buffer(device, &bytes, &bytes, &buffers)
            .unwrap()
            .unwrap();

        assert_eq!(second, first);
        assert_eq!(
            msg_send_ptr0(first, sel("contents")).cast::<u8>(),
            bytes.as_ptr() as *mut u8
        );
        assert_eq!(msg_send_usize0(first, sel("length")), page_size);
        let attached = resources.snapshot();
        assert_eq!(attached.resident_expert_wrapper_buffers, 1);
        assert_eq!(attached.resident_expert_wrapper_bytes, page_size);
        assert_eq!(attached.active_general_buffers, 0);
        drop(bytes);
        let released = resources.snapshot();
        assert_eq!(released.resident_expert_wrapper_buffers, 0);
        assert_eq!(released.resident_expert_wrapper_bytes, 0);
        assert_eq!(released.buffer_releases, 1);
        drop(buffers);
        release(device);
    });
}
