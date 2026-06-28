/// <reference lib="deno.ns" />
import { equal } from "node:assert/strict";
import { nextProjectNotificationPreference } from "./ProjectsPage.tsx";

Deno.test("nextProjectNotificationPreference flips the project notification setting", () => {
  equal(nextProjectNotificationPreference({ notify_on_finish: false }), true);
  equal(nextProjectNotificationPreference({ notify_on_finish: true }), false);
});
