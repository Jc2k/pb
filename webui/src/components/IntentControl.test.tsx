import { ok } from "node:assert/strict";
import { renderToString } from "react-dom/server";
import { IntentControl } from "./IntentControl.tsx";

Deno.test("intent control presents Discuss, Build, and durable Goal choices", () => {
  const html = renderToString(
    <IntentControl intent="discuss" onChange={() => {}} />,
  );

  ok(html.includes('aria-label="Session intent"'));
  ok(html.includes('aria-pressed="true"'));
  ok(html.includes("Discuss"));
  ok(html.includes("Build"));
  ok(html.includes("Goal"));
  ok(!html.includes("Deliver"));
});

Deno.test("intent control makes active goal mode textual and non-switchable", () => {
  const html = renderToString(
    <IntentControl intent="discuss" activeGoal onChange={() => {}} />,
  );
  ok(html.includes("Goal active"));
  ok(html.includes("disabled"));
});
