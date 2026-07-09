use super::capabilities::{
    FlashMoeCapabilityPlan, FlashMoeGraphStage, FlashMoeStageCapability, FlashMoeStagePlacement,
    FlashMoeUnsupportedCapability,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashMoeScheduledGraph {
    stages: Vec<FlashMoeStageCapability>,
}

impl FlashMoeScheduledGraph {
    pub fn from_capabilities(
        capabilities: &FlashMoeCapabilityPlan,
    ) -> Result<Self, FlashMoeUnsupportedCapability> {
        capabilities.validate_complete()?;
        let stages = FlashMoeGraphStage::ALL
            .iter()
            .map(|stage| {
                capabilities.stage(*stage).copied().ok_or_else(|| {
                    FlashMoeUnsupportedCapability::new(
                        capabilities.family,
                        *stage,
                        "graph stage has no declared implementation",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { stages })
    }

    pub fn stages(&self) -> &[FlashMoeStageCapability] {
        &self.stages
    }

    pub fn stage(&self, stage: FlashMoeGraphStage) -> &FlashMoeStageCapability {
        self.stages
            .iter()
            .find(|capability| capability.stage == stage)
            .expect("scheduled graph contains every FlashMoe stage")
    }

    pub fn active_expert_reads(&self) -> &FlashMoeStageCapability {
        self.stage(FlashMoeGraphStage::ActiveExpertReads)
    }

    pub fn cmd_sequence(&self) -> [&FlashMoeStageCapability; 3] {
        [
            self.stage(FlashMoeGraphStage::DeferredPreviousCmd3),
            self.stage(FlashMoeGraphStage::Cmd1AttentionProjections),
            self.stage(FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection),
        ]
    }

    pub fn declares_scheduler_owned_expert_reads(&self) -> bool {
        self.active_expert_reads().placement == FlashMoeStagePlacement::SchedulerIo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::flashmoe::{QWEN35_MODEL, QwenModelConfig, QwenMoeModelLayout};

    fn qwen35_layout() -> QwenMoeModelLayout {
        let config: QwenModelConfig = serde_json::from_slice(
            br#"{
  "model_type": "qwen3_5_moe",
  "architectures": ["Qwen3_5MoeForCausalLM"],
  "num_hidden_layers": 60,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 2,
  "vocab_size": 248320,
  "rope_theta": 10000000.0,
  "torch_dtype": "bfloat16",
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 1024,
  "num_shared_experts": 1,
  "shared_expert_intermediate_size": 1024
}"#,
        )
        .unwrap();
        QwenMoeModelLayout::from_config(QWEN35_MODEL, &config).unwrap()
    }

    #[test]
    fn scheduled_graph_preserves_the_declared_stage_order() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();

        let stages: Vec<_> = graph.stages().iter().map(|stage| stage.stage).collect();

        assert_eq!(stages, FlashMoeGraphStage::ALL);
        assert!(graph.declares_scheduler_owned_expert_reads());
    }

    #[test]
    fn scheduled_graph_exposes_the_upstream_command_sequence() {
        let capabilities = FlashMoeCapabilityPlan::for_model_layout(&qwen35_layout()).unwrap();
        let graph = FlashMoeScheduledGraph::from_capabilities(&capabilities).unwrap();
        let cmd_sequence: Vec<_> = graph
            .cmd_sequence()
            .iter()
            .map(|stage| stage.stage)
            .collect();

        assert_eq!(
            cmd_sequence,
            vec![
                FlashMoeGraphStage::DeferredPreviousCmd3,
                FlashMoeGraphStage::Cmd1AttentionProjections,
                FlashMoeGraphStage::Cmd2PostAttentionAndRoutingProjection,
            ]
        );
        assert_eq!(
            graph
                .stage(FlashMoeGraphStage::RoutingSoftmaxTopK)
                .placement,
            FlashMoeStagePlacement::CpuDeclared
        );
    }
}
