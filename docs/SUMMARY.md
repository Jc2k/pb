# Summary

[pb](README.md)

# User guide

- [Getting started](user/getting-started.md)
- [Working with pb](user/working-with-pb.md)
- [Configuration and integrations](user/configuration.md)
- [Your data and privacy](user/data-and-privacy.md)

# Architecture

- [How pb fits together](architecture/overview.md)
- [Conversation and delivery workflows](architecture/workflows.md)
- [Security model](architecture/security.md)
- [Local privacy model](architecture/local-privacy.md)
- [Contracts with the user](architecture/user-contracts.md)

# Operations and development

- [Documentation site](contributing/documentation.md)
- [Internal harness](harness.md)
- [Power estimation](power-estimation.md)

# Engineering records

- [Conversational delivery workflow](conversational-delivery-workflow-plan.md)
- [Durable goal mode](goal-mode-plan.md)
  - [G8 · Goal control qualification](benchmarks/goal-mode-g8.md)
- [Task decomposition workflow](multi-task-workflow-plan.md)
  - [Decomposition feasibility probe](benchmarks/multi-task-decomposition.md)
- [Agent handoff and workspaces](agent-handoff-workspace-plan.md)
- [Session environment control plane](session-environment-control-plane.md)
  - [Apple container foundation](apple-container-environment-architecture.md)
- [Small-model reliability](small-model-agent-reliability-plan.md)
  - [Baseline](benchmarks/small-model-agent-baseline.md)
  - [S1 · Prompt budget](benchmarks/small-model-agent-s1.md)
  - [S2 · Focused evidence](benchmarks/small-model-agent-s2.md)
  - [S3 · Progress recovery](benchmarks/small-model-agent-s3.md)
  - [S4 · Action recovery](benchmarks/small-model-agent-s4.md)
  - [S5 · Workflow closure](benchmarks/small-model-agent-s5.md)
  - [S6 · Model evaluation](benchmarks/small-model-agent-s6.md)
  - [DeepSeek V4 Flash field run](benchmarks/deepseek-v4-flash-agent.md)
- [Verified task-completion reliability](task-completion-reliability-plan.md)
  - [Deterministic controller actions](controller-action-elision-plan.md)
    - [E1 · Prompt-rendering screen](benchmarks/controller-action-elision-e1.md)
    - [E2 · Production qualification](benchmarks/controller-action-elision-e2.md)
    - [Final production qualification](benchmarks/deterministic-controller-actions-production.md)
  - [TC1 · Verified completion](benchmarks/task-completion-tc1.md)
  - [TC2 · Useful coding](benchmarks/task-completion-tc2.md)
  - [TC3 · Controller baseline](benchmarks/task-completion-tc3-baseline.md)
  - [Work-Unit Controller v2 qualification](benchmarks/task-completion-work-unit-v2.md)
- [Qwen3-Coder-Next native agent](qwen3-coder-next-agent-plan.md)
  - [Native agent evaluation](benchmarks/qwen3-coder-next-agent.md)
  - [Agent performance follow-on](qwen3-coder-next-agent-follow-on.md)
  - [Prefill qualification](benchmarks/qwen3-coder-next-prefill-qualification.md)
  - [Device-resident prefill graph](qwen3-coder-next-device-resident-prefill-plan.md)
- [Harness reliability](harness-improvement-plan.md)
  - [Open-weight workflow evaluation](harness-workflow-model-evaluation.md)
- [External publication follow-on](external-publication-workflow-follow-on.md)
- [FlashMoe architecture parity](flashmoe-architecture-parity-plan.md)
  - [Resource baseline](benchmarks/harness-r0-baseline.md)
  - [Resource result](benchmarks/harness-r0-after.md)
