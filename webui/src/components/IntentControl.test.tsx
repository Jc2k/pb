import { ok } from "node:assert/strict";
import { renderToString } from "react-dom/server";
import { IntentControl } from "./IntentControl.tsx";

Deno.test("intent control presents conversational Discuss and explicit Build choices", () => {
  const html = renderToString(
    <IntentControl intent="discuss" onChange={() => {}} />,
  );

  ok(html.includes('aria-label="Session intent"'));
  ok(html.includes('aria-pressed="true"'));
  ok(html.includes("Discuss"));
  ok(html.includes("Build"));
  ok(!html.includes("Deliver"));
});
