/// <reference lib="deno.ns" />
import { equal, ok } from "node:assert/strict";
import { nextProjectNotificationPreference } from "./ProjectsPage.tsx";

Deno.test("nextProjectNotificationPreference flips the project notification setting", () => {
  equal(nextProjectNotificationPreference({ notify_on_finish: false }), true);
  equal(nextProjectNotificationPreference({ notify_on_finish: true }), false);
});

Deno.test("project index uses the shared workspace frame", async () => {
  const source = await Deno.readTextFile("webui/src/pages/ProjectsPage.tsx");

  ok(source.includes('<PageShell contentClassName="projects-index-wrap">'));
  ok(source.includes('<span className="chevron" aria-hidden="true">'));
  ok(!source.includes('className="chevron text-decoration-none"'));
});
