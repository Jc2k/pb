use super::capabilities::{
    FlashMoeCapabilityPlan, FlashMoeGraphStage, FlashMoeStageCapability, FlashMoeStagePlacement,
    FlashMoeUnsupportedCapability,
};
use super::math::softmax_in_place;
use anyhow::{Result, bail};

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertRoute {
    pub expert: usize,
    pub score: f32,
}

impl ExpertRoute {
    pub fn from_pair((expert, score): (usize, f32)) -> Self {
        Self { expert, score }
    }

    pub fn from_scores(routes: &[(usize, f32)]) -> Result<Vec<Self>> {
        routes
            .iter()
            .copied()
            .map(|route| {
                let route = Self::from_pair(route);
                route.validate()?;
                Ok(route)
            })
            .collect()
    }

    fn validate(&self) -> Result<()> {
        if !self.score.is_finite() {
            bail!(
                "expert route score for expert {} is not finite: {}",
                self.expert,
                self.score
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledExpertRoutes {
    pub layer: usize,
    pub routes: Vec<ExpertRoute>,
    pub weights: Vec<f32>,
}

impl ScheduledExpertRoutes {
    pub fn from_routes(
        layer: usize,
        routes: Vec<ExpertRoute>,
        routed_expert_scale: f32,
    ) -> Result<Self> {
        if !(routed_expert_scale.is_finite() && routed_expert_scale > 0.0) {
            bail!("routed expert scale must be positive and finite");
        }
        for route in &routes {
            route.validate()?;
        }
        let mut weights: Vec<f32> = routes.iter().map(|route| route.score).collect();
        softmax_in_place(&mut weights);
        for weight in &mut weights {
            *weight *= routed_expert_scale;
        }
        Ok(Self {
            layer,
            routes,
            weights,
        })
    }

    pub fn from_scores(
        layer: usize,
        routes: &[(usize, f32)],
        routed_expert_scale: f32,
    ) -> Result<Self> {
        Self::from_routes(
            layer,
            ExpertRoute::from_scores(routes)?,
            routed_expert_scale,
        )
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

    #[test]
    fn scheduled_expert_routes_normalize_and_scale_scores() {
        let scheduled =
            ScheduledExpertRoutes::from_scores(12, &[(7, 1.0), (3, 2.0), (9, -1.0)], 0.25).unwrap();
        let mut expected = vec![1.0, 2.0, -1.0];
        softmax_in_place(&mut expected);
        for weight in &mut expected {
            *weight *= 0.25;
        }

        assert_eq!(scheduled.layer, 12);
        assert_eq!(
            scheduled.routes,
            vec![
                ExpertRoute {
                    expert: 7,
                    score: 1.0,
                },
                ExpertRoute {
                    expert: 3,
                    score: 2.0,
                },
                ExpertRoute {
                    expert: 9,
                    score: -1.0,
                },
            ]
        );
        assert_eq!(scheduled.weights.len(), expected.len());
        for (actual, expected) in scheduled.weights.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn scheduled_expert_routes_reject_non_finite_scores() {
        let err = ScheduledExpertRoutes::from_scores(0, &[(2, f32::NAN)], 1.0).unwrap_err();

        assert!(
            err.to_string()
                .contains("expert route score for expert 2 is not finite"),
            "{err:#}"
        );
    }
}
