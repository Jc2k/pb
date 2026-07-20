/// <reference lib="deno.ns" />
import { ok } from "node:assert/strict";
import { equal } from "node:assert/strict";
import {
  latestGoalChangeRequest,
  latestPendingDeliveryProposal,
  latestPendingGoalProposal,
  readyEvidenceLabel,
  workflowOutcomeLabel,
  workflowProgressLabel,
  workflowStageLabel,
} from "./SessionPage.tsx";

import type { EventEnvelope } from "../types/index.ts";

function cssRule(css: string, selector: string): string {
  const start = css.indexOf(`${selector} {`);
  ok(start >= 0, `missing CSS rule for ${selector}`);
  const end = css.indexOf("}", start);
  ok(end > start, `unterminated CSS rule for ${selector}`);
  return css.slice(start, end);
}

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

Deno.test("goal proposal stays read-only until a durable goal starts", () => {
  const events: EventEnvelope[] = [{
    version: "v1",
    event: {
      type: "goal_proposed",
      proposal_id: "goal-proposal-1",
      source_turn_id: "turn-1",
      objective: "Ship goal mode",
      criteria: [{ text: "Persist checkpoints", verifier: "review_required" }],
    },
  }];
  equal(latestPendingGoalProposal(events)?.objective, "Ship goal mode");
  events.push({
    version: "v1",
    event: {
      type: "goal_started",
      goal_id: "goal-1",
      objective: "Ship goal mode",
      plan_sha256: "digest",
    },
  });
  equal(latestPendingGoalProposal(events), undefined);
});

Deno.test("model goal change requests remain pending only until a user path resolves them", () => {
  const events: EventEnvelope[] = [{
    version: "v1",
    event: {
      type: "goal_change_requested",
      goal_id: "goal-1",
      kind: "budget",
      summary: "Need one more checked pass",
    },
  }];
  equal(latestGoalChangeRequest(events)?.kind, "budget");
  events.push({
    version: "v1",
    event: { type: "goal_resumed", goal_id: "goal-1" },
  });
  equal(latestGoalChangeRequest(events), undefined);
});

Deno.test("strict workflow stages and outcomes use compact truthful labels", () => {
  equal(workflowStageLabel("planning"), "Planning");
  equal(workflowStageLabel("plan_review"), "Challenging the plan");
  equal(workflowStageLabel("checking"), "Running checks");
  equal(workflowStageLabel("code_review"), "Challenging the code");
  equal(workflowStageLabel("committing"), "Creating reviewed commit");
  equal(workflowOutcomeLabel("no_change"), "No code changes");
  equal(workflowOutcomeLabel("review_failed"), "Needs another pass");
  equal(workflowOutcomeLabel("contract_unsatisfied"), "Needs another pass");
  equal(workflowOutcomeLabel("commit_blocked"), "Needs help");
  equal(workflowOutcomeLabel("cancelled"), "Cancelled — work preserved");
  equal(workflowProgressLabel("ready", "ready"), "Ready");
  equal(
    readyEvidenceLabel("0123456789abcdef"),
    "Reviewed commit 0123456789ab is ready to publish",
  );
});

Deno.test("workflow controls preserve work and restore conversation after terminal outcomes", async () => {
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");

  ok(page.includes("`/api/sessions/${sessionId}/cancel`"));
  ok(page.includes("`/api/sessions/${sessionId}/resume`"));
  ok(page.includes('session.workflow?.stage === "blocked"'));
  ok(page.includes("resume from the preserved stage"));
  ok(page.includes('setIntent("discuss")'));
  ok(page.includes(": !isRunning &&"));
  ok(
    page.includes(
      '(session.status === "completed" || session.status === "failed")',
    ),
  );
  ok(page.includes("<IntentControl intent={intent} onChange={setIntent} />"));
});

Deno.test("paused session composer keeps resume action at intrinsic width", async () => {
  const markup = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");

  ok(markup.includes('className="composer paused-composer"'));
  ok(markup.includes('className="btn btn-warning composer-action"'));
  const actionRule = cssRule(css, ".composer .btn.composer-action");
  ok(actionRule.includes("flex: 0 0 auto;"));
  ok(actionRule.includes("width: auto;"));
  ok(actionRule.includes("min-width: max-content;"));
  ok(actionRule.includes("margin-left: auto;"));
  ok(actionRule.includes("white-space: nowrap;"));
  const iconRule = cssRule(css, ".composer > .btn.rounded-circle");
  ok(iconRule.includes("flex: 0 0 42px;"));
  ok(!css.includes(".composer .btn {"));
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
  ok(
    component.includes(
      '<img src={getAvatarForProfile("monitor")} alt="Trinity Walker" />',
    ),
  );
  ok(!component.includes("/avatar-monitor.png"));
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
  const preRule = cssRule(css, ".bubble pre");
  ok(preRule.includes("max-width: 100%;"));
  ok(preRule.includes("overflow: auto;"));
  const nestedMessageRule = cssRule(css, ".assistant-message");
  ok(nestedMessageRule.includes("margin-left: 0 !important;"));
  ok(css.includes(".session-header .share-action,"));

  const appCss = await Deno.readTextFile("webui/src/app.css");
  ok(appCss.includes(".diff-block"));
  ok(appCss.includes(".result-pre"));
  ok(appCss.includes("overscroll-behavior-inline: contain;"));
  ok(!appCss.includes("/* diff viewer */ /*"));
});

Deno.test("session metrics expose the canonical estimate and its measurement quality", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const types = await Deno.readTextFile("webui/src/types/index.ts");

  ok(types.includes("power_summary?: string"));
  ok(
    component.includes(
      "funEnergySummary(totalRuntimeMs, totalTokens, totalEnergyJoules)",
    ),
  );
  ok(component.includes("the energy a 10 W LED bulb uses in"));
  ok(component.includes('case "session_metrics"'));
  ok(component.includes('case "llm_invocation"'));
  ok(component.includes("e.prompt_cache.cached_tokens"));
  ok(component.includes("e.prompt_cache.prefilled_tokens"));
  ok(component.includes('e.prompt_cache.source.replaceAll("_", " ")'));
  ok(component.includes("Power-estimate details"));
  ok(component.includes("Measurement coverage"));
  ok(component.includes("Gross device energy"));
  ok(
    component.includes(
      '<article className="session-correction" aria-label="Session metrics">',
    ),
  );
  ok(
    component.includes("? <time>{formatEventTime(e.timestamp_ms)}</time>"),
  );
  ok(!component.includes("<strong>Power</strong>"));
  ok(!component.includes("{e.power_summary"));
  ok(!component.includes('<i className="bi bi-speedometer2"></i>'));
});

Deno.test("session workspace separates user chat from assistant transcript content", async () => {
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");

  ok(page.includes('className="app-shell session-shell"'));
  ok(page.includes("Session details"));
  ok(page.includes('title="Plan"'));
  ok(page.includes('title="Activity"'));
  ok(page.includes("<SessionActivity events={events} />"));
  ok(component.includes("assistant-message assistant-transcript"));
  ok(component.includes("<strong>You</strong>"));
  ok(component.includes("function activityLabel"));
  ok(component.includes('className="transcript-diff"'));
  ok(
    css.includes(
      ".assistant-transcript > .message-container > .thought-bubble",
    ),
  );
  ok(css.includes(".transcript-diff-body"));
  ok(css.includes(".tool-drawer-heading"));
  ok(css.includes(".session-activity-list"));
});

Deno.test("handoff feedback renders as a teammate with expandable evidence", async () => {
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");
  const types = await Deno.readTextFile("webui/src/types/index.ts");

  ok(page.includes('session.status === "completed"'));
  ok(types.includes("export type SessionStatus"));
  for (const status of ["queued", "running", "paused", "completed", "failed"]) {
    ok(types.includes(`"${status}"`));
  }
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
