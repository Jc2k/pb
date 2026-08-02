import { ok } from "node:assert/strict";
import { renderToString } from "react-dom/server";
import type { GoalCheckpoint } from "../types/index.ts";
import { GoalModeBanner } from "./GoalModeBanner.tsx";
import { GoalStartSheet } from "./GoalStartSheet.tsx";

function goal(stage: GoalCheckpoint["run"]["stage"]): GoalCheckpoint {
  return {
    sha256: "checkpoint",
    run: {
      id: "goal-1",
      objective: "Ship durable goal mode",
      stage,
      pause_requested: false,
      milestones: [{ id: "m1", status: "running" }],
    },
  } as GoalCheckpoint;
}

Deno.test("goal banner keeps mode, state, objective, progress, and controls textual", () => {
  const html = renderToString(
    <GoalModeBanner
      goal={goal("running_milestone")}
      onDetails={() => {}}
      onPause={() => {}}
      onResume={() => {}}
      onAccept={() => {}}
      onEdit={() => {}}
      onStop={() => {}}
    />,
  );
  ok(html.includes("Goal running"));
  ok(html.includes("Ship durable goal mode"));
  ok(html.includes("goal-mode-count"));
  ok(html.includes("0"));
  ok(html.includes("1"));
  ok(html.includes("Pause"));
  ok(html.includes("Pause and edit"));
  ok(html.includes("Stop goal"));
  ok(html.includes("Details"));
});

Deno.test("goal setup exposes criteria, continuation, three presets, advanced limits, and authority", () => {
  const html = renderToString(
    <GoalStartSheet
      open
      initialObjective="Ship durable goal mode"
      initialCriteria={["Persist checkpoints"]}
      projectName="project"
      onClose={() => {}}
      onStarted={() => {}}
    />,
  );
  for (
    const text of [
      "Done when",
      "How to continue",
      "Compact",
      "Standard",
      "Extended",
      "Advanced limits",
      "No publishing",
      "Plan goal",
    ]
  ) {
    ok(html.includes(text), `missing ${text}`);
  }
});

Deno.test("goal mobile surfaces use safe areas and retain a details route to controls", async () => {
  const css = await Deno.readTextFile(
    new URL("../session.css", import.meta.url),
  );
  ok(css.includes(".goal-mode-banner"));
  ok(css.includes("env(safe-area-inset-top)"));
  ok(css.includes(".goal-mobile-actions"));
  ok(css.includes("@media (max-width: 767px)"));
});
