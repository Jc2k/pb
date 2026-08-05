/// <reference lib="deno.ns" />
import { ok } from "node:assert/strict";
import { equal } from "node:assert/strict";
import {
  mergeEventHistory,
  mergeResetEventHistory,
  readyEvidenceLabel,
  workflowOutcomeLabel,
  workflowProgressLabel,
  workflowRecoveryPresentation,
  workflowStageLabel,
} from "./SessionPage.tsx";

import type { EventEnvelope } from "../types/index.ts";

let testEventIndex = 0;
function eventEnvelopeDefaults(): Pick<
  EventEnvelope,
  "chatter" | "evidence" | "transcript"
> {
  testEventIndex += 1;
  return {
    chatter: [],
    evidence: [],
    transcript: {
      sequence: testEventIndex,
      visibility: "visible",
      kind: "conversation",
      entry_key: `test-event-${testEventIndex}`,
      supersedes: [],
      summary_redundant: false,
    },
  };
}

function cssRule(css: string, selector: string): string {
  const start = css.indexOf(`${selector} {`);
  ok(start >= 0, `missing CSS rule for ${selector}`);
  const end = css.indexOf("}", start);
  ok(end > start, `unterminated CSS rule for ${selector}`);
  return css.slice(start, end);
}

function cssRuleAfter(css: string, selector: string, after: string): string {
  const afterIndex = css.indexOf(after);
  ok(afterIndex >= 0, `missing CSS anchor ${after}`);
  const start = css.indexOf(`${selector} {`, afterIndex);
  ok(start >= 0, `missing CSS rule for ${selector} after ${after}`);
  const end = css.indexOf("}", start);
  ok(end > start, `unterminated CSS rule for ${selector}`);
  return css.slice(start, end);
}

function cssRuleNthAfter(
  css: string,
  selector: string,
  after: string,
  occurrence: number,
): string {
  ok(occurrence >= 1, "CSS occurrence must be >= 1");
  let searchStart = css.indexOf(after);
  ok(searchStart >= 0, `missing CSS anchor ${after}`);
  let start = -1;
  for (let index = 0; index < occurrence; index += 1) {
    start = css.indexOf(`${selector} {`, searchStart);
    ok(start >= 0, `missing CSS rule ${selector} occurrence ${occurrence} after ${after}`);
    searchStart = start + selector.length + 2;
  }
  const end = css.indexOf("}", start);
  ok(end > start, `unterminated CSS rule for ${selector}`);
  return css.slice(start, end);
}

Deno.test("event history merges stream and snapshot data by stable entry key", () => {
  const started: EventEnvelope = {
    ...eventEnvelopeDefaults(),
    version: "v6",
    event: {
      type: "started",
      task: "Review the boundary",
      model: "local-model",
      workspace: "/tmp/project",
      focus_root: "/tmp/project",
      branch: "main",
      attachments: [],
      profile: "build",
    },
  };
  const streamed: EventEnvelope = {
    ...eventEnvelopeDefaults(),
    version: "v6",
    event: {
      type: "user_message",
      message_id: "message-1",
      message: "Done",
    },
  };
  const corrected = {
    ...streamed,
    transcript: { ...streamed.transcript, summary_redundant: true },
  };

  const merged = mergeEventHistory(
    [started, streamed],
    [corrected],
  );

  equal(merged.length, 2);
  equal(merged[0], started);
  equal(merged[1], corrected);
});

Deno.test("history reset uses the event watermark, never the session revision", () => {
  const retained: EventEnvelope = {
    ...eventEnvelopeDefaults(),
    version: "v6",
    event: {
      type: "user_message",
      message_id: "retained",
      message: "retained",
    },
  };
  retained.transcript.sequence = 40;
  const racingLive: EventEnvelope = {
    ...eventEnvelopeDefaults(),
    version: "v6",
    event: {
      type: "user_message",
      message_id: "racing-live",
      message: "racing live event",
    },
  };
  racingLive.transcript.sequence = 41;

  const merged = mergeResetEventHistory(
    [retained],
    [retained, racingLive],
  );

  equal(merged.length, 2);
  equal(merged[1]?.transcript.entry_key, racingLive.transcript.entry_key);
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
  equal(workflowOutcomeLabel("step_limit"), "Needs another pass");
  equal(workflowOutcomeLabel("engine_error"), "Needs another pass");
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
  ok(page.includes("`/api/sessions/${sessionId}/${action}`"));
  ok(page.includes('action: "restart-delivery"'));
  ok(page.includes('session.workflow?.stage === "blocked"'));
  ok(page.includes("Restart with current files"));
  ok(page.includes('setIntent("discuss")'));
  ok(page.includes(": !isRunning &&"));
  ok(
    page.includes(
      '(session.status === "completed" || session.status === "failed")',
    ),
  );
  ok(page.includes("<IntentControl intent={intent} onChange={setIntent} />"));
});

Deno.test("blocked delivery recovery is reason-aware", () => {
  const restart = workflowRecoveryPresentation({
    id: "workflow-1",
    source_turn_id: "turn-1",
    task: "Update the project page",
    stage: "blocked",
    outcome: "commit_blocked",
    policy_sha256: "policy",
    blocked_reason:
      "repository content changed while the read-only PlanReview stage was running",
    blocked_cause: "repository_content_changed",
    recovery: "restart_from_current_files",
  });
  equal(restart.action, "restart-delivery");
  equal(restart.label, "Restart with current files");
  ok(restart.description.includes("previous plan and review remain"));

  const resume = workflowRecoveryPresentation({
    id: "workflow-2",
    source_turn_id: "turn-2",
    task: "Update the project page",
    stage: "blocked",
    outcome: "executor_unavailable",
    policy_sha256: "policy",
    recovery: "resume",
  });
  equal(resume.action, "resume");
  equal(resume.label, "Resume after fixing");
});

Deno.test("running sessions keep a plain-message composer without intent controls", async () => {
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");

  ok(page.includes("`/api/sessions/${sessionId}/message`"));
  ok(page.includes('placeholder="Message the running agent…"'));
  ok(page.includes('aria-label="Send message to running agent"'));
  ok(page.includes("{isRunning"));
  ok(page.includes("body: JSON.stringify({ message })"));
});

Deno.test("paused session composer keeps resume action at intrinsic width", async () => {
  const markup = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");

  ok(markup.includes('className="composer paused-composer"'));
  ok(markup.includes("className={`btn composer-action"));
  const actionRule = cssRule(css, ".composer .btn.composer-action");
  ok(actionRule.includes("flex: 0 0 auto;"));
  ok(actionRule.includes("width: auto;"));
  ok(actionRule.includes("min-width: max-content;"));
  ok(actionRule.includes("margin-left: auto;"));
  ok(actionRule.includes("white-space: nowrap;"));
  const iconRule = cssRule(css, ".composer > .btn.rounded-circle");
  ok(iconRule.includes("flex: 0 0 44px;"));
  ok(!css.includes(".composer .btn {"));
});

Deno.test("session corrections render as direct steward chat with progressive detail", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");
  const types = await Deno.readTextFile("webui/src/types/index.ts");

  ok(types.includes('type: "correction"'));
  ok(types.includes("interface EventChatter"));
  ok(types.includes("chatter: EventChatter[]"));
  ok(types.includes("transcript: TranscriptMetadata"));
  ok(component.includes('case "correction"'));
  ok(component.includes("function CorrectionNotice"));
  ok(component.includes("teamActorPresentation(copy.actor)"));
  ok(component.includes("Correction from ${teammate.name}"));
  ok(component.includes('audience === "team"'));
  ok(!component.includes("trinityCorrectionCopy("));
  ok(!component.includes("normalizedSummary"));
  ok(component.includes("function TechnicalDetailsBubble"));
  ok(component.includes("technical-detail-button"));
  ok(component.includes("window.setTimeout"));
  ok(component.includes("Technical details"));
  ok(component.includes("function WorkflowBlockedNotice"));
  ok(component.includes('audience === "current_user"'));
  ok(!component.includes('fetch("/api/current-user"'));
  ok(component.includes("Task hold message from ${teammate.name}"));
  ok(component.includes("Request from ${teammate.name} to the current user"));
  ok(!component.includes("Delivery not completed"));
  ok(!component.includes("Your next step"));
  ok(!component.includes("terminal-next-step"));
  ok(component.includes('className="action-origin"'));
  ok(component.includes("chat-event-message"));
  ok(component.includes("trinity-message"));
  ok(component.includes("<RichText content={copy.message} />"));
  ok(css.includes(".correction-bubble"));
  ok(css.includes("--trinity-accent:"));
  ok(css.includes(".teammate-message .thought-bubble"));
  ok(css.includes(".teammate-message.chat-event-message"));
  ok(!css.includes(".trinity-message .team-avatar"));
  ok(css.includes(".technical-detail-surface:hover .technical-detail-button"));
  ok(!css.includes(".terminal-next-step"));
});

Deno.test("accepted delivery plans and reviewer prose stay visible in chat", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const helpers = await Deno.readTextFile("webui/src/lib/helpers.ts");

  ok(component.includes("function DeliveryPlanCard"));
  ok(component.includes('"Review invalidated"'));
  ok(component.includes('"Review incomplete"'));
  ok(component.includes("reviewEndedIncomplete"));
  ok(component.includes("What it must achieve"));
  ok(component.includes("Implementation"));
  ok(component.includes("Done when"));
  ok(component.includes('case "workflow_artifact_accepted"'));
  ok(page.includes("workflow={session.workflow ?? undefined}"));
  ok(!component.includes("Notes from this run"));
  ok(!helpers.includes("reasoningEvents:"));
});

Deno.test("session chat groups speakers and reveals useful message times", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");

  ok(page.includes("buildChatPresentation("));
  ok(page.includes('event.pointerType !== "touch"'));
  ok(page.includes('showMessageTimes ? " show-message-times"'));
  ok(component.includes('" speaker-continuation"'));
  ok(component.includes("function MessageTime"));
  ok(css.includes(".chat-time-divider"));
  ok(css.includes(".show-message-times .message-time"));
  ok(css.includes(".speaker-continuation > .bot-avatar"));
});

Deno.test("iPhone transcript keeps chat bubbles instead of card wrappers", async () => {
  const css = await Deno.readTextFile("webui/src/session.css");

  const mobileChatEventRule = cssRule(css, ".chat-event-message");
  ok(mobileChatEventRule.includes("padding: 0;"));
  ok(mobileChatEventRule.includes("border: 0;"));
  ok(mobileChatEventRule.includes("background: transparent;"));

  const mobileChatBubbleMediaRule = cssRuleAfter(
    css,
    ".chat-event-message > .message-container > .thought-bubble",
    "@media (max-width: 575.98px)",
  );
  ok(mobileChatBubbleMediaRule.includes("padding: 0.72rem 0.85rem;"));
  ok(!mobileChatBubbleMediaRule.includes("border: 0;"));

  const mobileUserContainerRule = cssRule(css, ".user-message .message-container");
  ok(mobileUserContainerRule.includes("max-width: min(88%, 28rem);"));

  const mobileUserBubbleRule = cssRuleAfter(
    css,
    ".user-bubble",
    "@media (max-width: 575.98px)",
  );
  const mobileUserWidthRule = cssRuleNthAfter(
    css,
    ".user-bubble",
    "@media (max-width: 575.98px)",
    2,
  );
  ok(mobileUserBubbleRule.includes("border-bottom-right-radius: 6px;"));
  ok(mobileUserWidthRule.includes("max-width: 100%;"));
});

Deno.test("assistant and Trinity prose share safe inline Markdown rendering", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");

  ok(component.includes("parseInlineRichText(content)"));
  ok(component.includes('className="rich-text-inline-code"'));
  ok(component.includes("<RichText content={e.content} />"));
  ok(component.includes("<RichText content={copy.message} />"));
});

Deno.test("final assistant messages use profile avatars", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");

  ok(component.includes("function AssistantMessageRow"));
  ok(
    component.includes(
      "<img src={getAvatarForProfile(profile)} alt={profileName(profile)} />",
    ),
  );
  ok(component.includes("teamActorPresentation(event.actor)"));
  ok(!component.includes("/avatar-monitor.png"));
  ok(component.includes('case "final"'));
  ok(!component.includes('case "final":\n      const ffd'));
});

Deno.test("session page respects iPhone safe areas and prevents horizontal overflow", async () => {
  const css = await Deno.readTextFile("webui/src/session.css");
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");

  ok(css.includes("max-width: 100vw;"));
  ok(css.includes("overflow-x: hidden;"));
  ok(css.includes("env(safe-area-inset-top)"));
  ok(css.includes("env(safe-area-inset-left)"));
  ok(css.includes("env(safe-area-inset-right)"));
  ok(css.includes(".message-container"));
  ok(css.includes("overflow-wrap: anywhere;"));
  ok(
    /\.session-layout\.has-work-drawer\s*\{\s*grid-template-columns:\s*minmax\(0,\s*1fr\);/s
      .test(css),
  );
  ok(css.includes(".session-back-button"));
  ok(page.includes('aria-label="Back to sessions"'));
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

Deno.test("session metrics stay compact while inference details remain available", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");
  const types = await Deno.readTextFile("webui/src/types/index.ts");
  const inferenceDetails = component.slice(
    component.indexOf("function InferenceDetails"),
    component.indexOf("export function MessageBubble"),
  );
  const runtimeDetails = component.slice(
    component.indexOf("function SessionMetricsDetails"),
    component.indexOf("export function MessageBubble"),
  );
  const actionGroup = component.slice(
    component.indexOf("export function ActionGroupBubble"),
    component.indexOf("export function ActionDrawerItem"),
  );
  const actionInferenceDetails = component.slice(
    component.indexOf("function ActionInferenceDetails"),
    component.indexOf("function InferenceDetails"),
  );

  ok(types.includes("power_summary: string"));
  ok(
    component.includes(
      "funEnergySummary(totalRuntimeMs, totalTokens, totalEnergyJoules)",
    ),
  );
  ok(component.includes("the energy a 10 W LED bulb uses in"));
  ok(component.includes('case "session_metrics"'));
  ok(component.includes('case "llm_invocation"'));
  ok(types.includes('"task_partitioning"'));
  ok(types.includes('"workflow_recovery"'));
  ok(inferenceDetails.includes("worked for"));
  ok(!inferenceDetails.includes("used the model"));
  ok(actionInferenceDetails.includes("worked for"));
  ok(actionGroup.includes("Predicted"));
  ok(actionGroup.includes("trinity-prediction-glyph"));
  ok(actionGroup.includes("tool-strip-copy"));
  ok(
    actionGroup.indexOf("<ActionInferenceDetails") >
      actionGroup.indexOf('className="bubble thought-bubble action-bubble"'),
  );
  ok(inferenceDetails.includes("View inference ${event.step} details"));
  ok(component.includes('role="dialog"'));
  ok(inferenceDetails.includes("window.setTimeout"));
  ok(inferenceDetails.includes("event.prompt_cache.cached_tokens"));
  ok(inferenceDetails.includes("event.prompt_cache.prefilled_tokens"));
  ok(inferenceDetails.includes("event.prompt_cache.miss_reason"));
  ok(inferenceDetails.includes("event.prompt_cache.lookup_detail"));
  ok(inferenceDetails.includes("event.prompt_cache.root.reused_tokens"));
  ok(inferenceDetails.includes("event.prompt_cache.root.tokens"));
  ok(inferenceDetails.includes("event.prompt_cache.root.authority_class"));
  ok(types.includes("cache_format_version"));
  ok(inferenceDetails.includes("event.native.refill"));
  ok(
    inferenceDetails.includes(
      "event.native.refill.fresh_suffix_prefill_wall_ms",
    ),
  );
  ok(inferenceDetails.includes("event.native.refill.disk_read_decode_wall_ms"));
  ok(
    inferenceDetails.includes(
      "event.native.refill.cpu_state_validation_allocation_wall_ms",
    ),
  );
  ok(
    inferenceDetails.includes("event.native.refill.persistence_queue_wall_ms"),
  );
  ok(component.includes("Runtime details"));
  ok(runtimeDetails.includes('label="Coverage"'));
  ok(component.includes("Gross device energy"));
  ok(
    component.includes(
      'className="session-correction session-metrics-summary"',
    ),
  );
  ok(component.includes('aria-label="Session runtime summary"'));
  ok(runtimeDetails.includes("View session runtime details"));
  ok(runtimeDetails.includes("<MetricsDialog"));
  ok(!runtimeDetails.includes("<details>"));
  ok(runtimeDetails.includes("event.cache_persistence_completed_checkpoints"));
  ok(runtimeDetails.includes("How pb helped"));
  ok(runtimeDetails.includes("Potential turns avoided"));
  ok(runtimeDetails.includes("not a counterfactual saving"));
  ok(runtimeDetails.includes("harnessEfficiencyStats(evidenceEvents)"));
  ok(inferenceDetails.includes('label="Candidates filtered"'));
  ok(actionInferenceDetails.includes('label="Control collar filtered"'));
  ok(!component.includes("<strong>Power</strong>"));
  ok(!component.includes("{e.power_summary"));
  ok(!component.includes('<i className="bi bi-speedometer2"></i>'));
  const markerRule = cssRule(
    css,
    ".inference-marker,\n.session-metrics-summary",
  );
  ok(markerRule.includes("position: relative;"));
  ok(markerRule.includes("width: fit-content;"));
  const infoRule = cssRule(css, ".inference-info-button");
  ok(infoRule.includes("position: absolute;"));
  ok(infoRule.includes("opacity: 0;"));
  ok(css.includes(".session-metrics-summary:hover .inference-info-button"));
});

Deno.test("terminal Trinity feedback uses server-authored chatter and addresses the current user", async () => {
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");

  ok(component.includes("const teammateFeedback = envelope.chatter.find("));
  ok(component.includes("const userFeedback = envelope.chatter.find("));
  ok(component.includes('audience === "current_user"'));
  ok(component.includes("<RichText content={userFeedback.message} />"));
  ok(!component.includes("currentUsername"));
  ok(!component.includes("Choose **Build** below"));
});

Deno.test("session mutations apply the authoritative stream snapshot response", async () => {
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const mutations = page.slice(
    page.indexOf("const continueSession"),
    page.indexOf("const onChatScroll"),
  );

  ok(!mutations.includes("setSessionRunning"));
  ok(mutations.includes('setFollowUp("");\n      setIntent("discuss");'));
  ok(
    (mutations.match(/parseSessionStreamSnapshotJson/g)?.length ?? 0) >= 7,
  );
  ok(
    (mutations.match(/applySessionSnapshot\(/g)?.length ??
      0) >= 7,
  );
  ok((mutations.match(/snapshot\.warnings/g)?.length ?? 0) >= 7);
  ok(
    page.includes(
      "goalControlsBusy = goalBusy || session?.cancel_requested === true",
    ),
  );
  ok(!page.includes("setSessionRunning"));
  ok(page.includes('const isRunning = session?.status === "running"'));
  ok(!page.includes("session?.running"));
});

Deno.test("committed warnings survive a newer session revision", async () => {
  const source = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const applyStart = source.indexOf("const applySessionSnapshot = (");
  const applyEnd = source.indexOf("const openEvents", applyStart);
  const apply = source.slice(applyStart, applyEnd);
  equal(
    apply.indexOf('setActionError(warnings.join(" "))') <
      apply.indexOf("details.revision < snapshotRevisionRef.current"),
    true,
  );
});

Deno.test("session workspace prioritizes chat and shows work details only when useful", async () => {
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const component = await Deno.readTextFile("webui/src/components/Session.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");
  const team = await Deno.readTextFile("webui/src/lib/team.ts");
  const constants = await Deno.readTextFile("webui/src/lib/constants.ts");

  ok(page.includes('className="app-shell session-shell"'));
  ok(page.includes("showWorkDrawer"));
  ok(page.includes('aria-label="Work details"'));
  ok(page.includes('title="Actions"'));
  ok(!page.includes('title="Plan"'));
  ok(!page.includes("Session details"));
  ok(!page.includes('title="Activity"'));
  ok(!page.includes("<SessionActivity events={events} />"));
  ok(!page.includes("workflow-progress"));
  ok(page.includes("taskPlanningTranscript.attempts.length > 0"));
  ok(page.includes("Branch: {session.branch}"));
  ok(page.includes("const sessionStartMs = session.started_at_ms"));
  ok(component.includes("assistant-message assistant-transcript"));
  ok(component.includes("<strong>You</strong>"));
  ok(!component.includes("Session request"));
  ok(!component.includes("function activityLabel"));
  ok(component.includes('className="action-origin"'));
  ok(component.includes("export function ActionDrawerItem"));
  ok(component.includes('className="drawer-action-detail"'));
  ok(page.includes("buildActionTimeline(events)"));
  ok(team.includes('provenance: "Model"'));
  ok(team.includes('provenance: "Harness"'));
  ok(constants.includes('session_changes: "Review recent work"'));
  ok(constants.includes('run_check: "Run acceptance check"'));
  ok(component.includes("Closed no-change work"));
  ok(component.includes('className="transcript-diff"'));
  ok(
    css.includes(
      ".chat-event-message > .message-container > .thought-bubble",
    ),
  );
  ok(css.includes(".transcript-diff-body"));
  ok(!css.includes(".tool-drawer-heading"));
  ok(!css.includes(".session-activity-list"));
  ok(!css.includes(".workflow-progress"));
  const drawerTitleRule = cssRule(css, ".drawer-item .drawer-action-title");
  ok(drawerTitleRule.includes("grid-template-columns: minmax(0, 1fr) auto;"));
  const drawerDetailRule = cssRule(css, ".drawer-action-detail");
  ok(drawerDetailRule.includes("overflow-wrap: anywhere;"));
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
  ok(component.includes("teamActorPresentation(event.actor)"));
  ok(component.includes("const summary = event.handoff"));
  ok(!component.includes("followingEvents.find"));
  ok(component.includes("What I ran"));
  ok(component.includes("check.command"));
  ok(component.includes("summary.affected_components"));
  ok(page.includes("focusRoot={session.workdir}"));
  ok(component.includes("envelope.evidence.flatMap"));
  ok(!component.includes("evidence_ids"));
  ok(!component.includes('startsWith("check:")'));
  ok(!component.includes('startsWith("commit:")'));
  ok(css.includes(".team-message"));
  ok(!component.toLowerCase().includes("contract verified"));
});

Deno.test("session transport opens first and lets EventSource reconnect with deduplication", async () => {
  const page = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const routeEffect = page.slice(page.indexOf("setEvents([]);"));

  ok(
    routeEffect.indexOf("openEvents(sessionId);") <
      routeEffect.indexOf("void fetchSession();"),
  );
  ok(page.includes("mergeEventHistory(previous, [parsed])"));
  ok(page.includes("mergeResetEventHistory(details.events, previous)"));
  ok(page.includes("if (sourceRef.current !== src) return;"));
  ok(page.includes("new LatestRequest()"));
  ok(page.includes("sessionRequestRef.current.owns(controller)"));
  ok(page.includes("setSession(details)"));
  ok(!page.includes("setSessionRunning"));
  ok(!page.includes("session_effect"));
  ok(page.includes("snapshotRevisionRef.current = details.revision"));
  ok(page.includes('addEventListener("session_snapshot"'));
  ok(page.includes("parsed.reset_history"));
  ok(page.includes("const eventWatermark"));
  ok(!page.includes("envelope.transcript.sequence > details.revision"));
  ok(page.includes('setActionError(warnings.join(" "))'));
  ok(!page.includes("sessionRefreshRequestedRef"));
  ok(!page.includes("latestRefreshEffectRef"));
  equal(page.match(/fetchSession\(\)/g)?.length, 3);
  ok(!page.includes("await fetchSession()"));
  ok(page.includes("session?.pending_delivery_proposal"));
  ok(page.includes("session?.pending_goal_proposal"));
  ok(page.includes("session?.pending_goal_change"));
  ok(!page.includes("latestPendingDeliveryProposal"));
  ok(!page.includes("latestPendingGoalProposal"));
  ok(!page.includes("latestGoalChangeRequest"));
  ok(!page.includes("src.onerror"));
});
