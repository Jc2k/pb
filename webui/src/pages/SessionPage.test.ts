/// <reference lib="deno.ns" />
import { ok } from "node:assert/strict";
import { equal } from "node:assert/strict";
import {
  latestPendingDeliveryProposal,
  workflowOutcomeLabel,
  workflowProgressLabel,
  workflowStageLabel,
} from "./SessionPage.tsx";
import type { EventEnvelope } from "../types/index.ts";

Deno.test("delivery proposal remains conversational until an explicit Build turn", () => {
  const events: EventEnvelope[] = [
    {
      version: "v1",
      event: {
        type: "delivery_proposed",
        proposal_id: "proposal-1",
        source_turn_id: "turn-1",
        task_summary: "Implement the agreed change",
      },
    },
  ];
  equal(latestPendingDeliveryProposal(events)?.proposal_id, "proposal-1");

  events.push({
    version: "v1",
    event: {
      type: "conversation_turn_started",
      turn_id: "turn-2",
      intent: "discuss",
      task: "What would that affect?",
    },
  });
  equal(latestPendingDeliveryProposal(events)?.proposal_id, "proposal-1");

  events.push({
    version: "v1",
    event: {
      type: "conversation_turn_started",
      turn_id: "turn-3",
      intent: "deliver",
      task: "Go ahead",
    },
  });
  equal(latestPendingDeliveryProposal(events), undefined);
});

Deno.test("strict workflow stages and outcomes use compact truthful labels", () => {
  equal(workflowStageLabel("planning"), "Planning");
  equal(workflowStageLabel("plan_review"), "Challenging the plan");
  equal(workflowStageLabel("checking"), "Running checks");
  equal(workflowStageLabel("code_review"), "Challenging the code");
  equal(workflowStageLabel("committing"), "Creating reviewed commit");
  equal(workflowOutcomeLabel("no_change"), "No code changes");
  equal(workflowOutcomeLabel("review_failed"), "Needs another pass");
  equal(workflowOutcomeLabel("commit_blocked"), "Needs help");
  equal(workflowOutcomeLabel("cancelled"), "Cancelled — work preserved");
  equal(workflowProgressLabel("ready", "ready"), "Ready");
});

Deno.test("workflow controls preserve work and restore conversation after terminal outcomes", async () => {
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");

  ok(page.includes("`/api/sessions/${sessionId}/cancel`"));
  ok(page.includes("`/api/sessions/${sessionId}/resume`"));
  ok(page.includes('session.workflow?.stage === "blocked"'));
  ok(page.includes("resume from the preserved stage"));
  ok(page.includes('setIntent("discuss")'));
  ok(
    page.includes(
      '!isRunning && (session.status === "completed" || session.status === "failed")',
    ),
  );
  ok(page.includes("<IntentControl intent={intent} onChange={setIntent} />"));
});

Deno.test("paused session composer keeps resume action at intrinsic width", async () => {
  const markup = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");

  ok(markup.includes('className="composer paused-composer"'));
  ok(markup.includes('className="btn btn-warning composer-action"'));
  ok(css.includes(".composer .btn.composer-action"));
  ok(css.includes("width: auto;"));
  ok(css.includes("white-space: nowrap;"));
});

Deno.test("session corrections render as centered plain notices", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");
  const types = await Deno.readTextFile("webui/src/types/index.ts");

  ok(types.includes('type: "correction"'));
  ok(component.includes('case "correction"'));
  ok(component.includes('className="session-correction"'));
  ok(css.includes(".session-correction"));
  ok(css.includes("align-self: center;"));
  ok(!css.includes(".session-correction .bubble"));
});

Deno.test("final assistant messages use profile avatars", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");

  ok(component.includes("function AssistantMessageRow"));
  ok(
    component.includes(
      "<img src={getAvatarForProfile(profile)} alt={profileName(profile)} />",
    ),
  );
  ok(component.includes('case "final"'));
  ok(!component.includes('case "final":\n      const ffd'));
});

Deno.test("session page respects iPhone safe areas and prevents horizontal overflow", async () => {
  const css = await Deno.readTextFile("webui/src/session.css");

  ok(css.includes("max-width: 100vw;"));
  ok(css.includes("overflow-x: hidden;"));
  ok(css.includes("env(safe-area-inset-top)"));
  ok(css.includes("env(safe-area-inset-left)"));
  ok(css.includes("env(safe-area-inset-right)"));
  ok(css.includes(".message-container"));
  ok(css.includes("overflow-wrap: anywhere;"));
  ok(css.includes(".session-header .share-action,"));
});

Deno.test("session metrics show a concise human summary after session summary", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const types = await Deno.readTextFile("webui/src/types/index.ts");

  ok(types.includes("power_summary?: string"));
  ok(component.includes("funEnergySummary(totalRuntimeMs, totalTokens, totalEnergyKwh)"));
  ok(component.includes("used enough electricity to power an LED bulb"));
  ok(component.includes('case "session_metrics"'));
  ok(component.includes('<article className="session-correction" aria-label="Session metrics">'));
  ok(component.includes("{e.timestamp_ms ? <time>{formatEventTime(e.timestamp_ms)}</time> : null}"));
  ok(!component.includes("<strong>Power</strong>"));
  ok(!component.includes("e.power_summary"));
  ok(!component.includes('<i className="bi bi-speedometer2"></i>'));
});

Deno.test("handoff feedback renders as a teammate with expandable evidence", async () => {
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");
  const types = await Deno.readTextFile("webui/src/types/index.ts");

  ok(page.includes('session.status === "completed"'));
  ok(types.includes('SessionStatus = "queued" | "running" | "paused" | "completed" | "failed"'));
  ok(types.includes('type: "team_message"'));
  ok(types.includes('type: "check_result"'));
  ok(types.includes('type: "handoff_summary"'));
  ok(types.includes("handoff_outcome"));
  ok(component.includes("function TeamMessageBubble"));
  ok(component.includes("Trinity Walker"));
  ok(component.includes("What I ran"));
  ok(component.includes("check.command"));
  ok(component.includes("summary.affected_components"));
  ok(component.includes("start.event.focus_root"));
  ok(css.includes(".team-message"));
  ok(!component.toLowerCase().includes("contract verified"));
});
