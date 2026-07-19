/// <reference lib="deno.ns" />
import { ok } from "node:assert/strict";
import { renderToString } from "react-dom/server";
import { SleepSettingControl } from "./SettingsPage.tsx";

Deno.test("sleep setting reports a live macOS assertion", () => {
  const html = renderToString(
    <SleepSettingControl
      settings={{
        prevent_sleep_while_working: true,
        prevent_sleep_supported: true,
        prevent_sleep_active: true,
      }}
    />,
  );

  ok(html.includes('role="switch"'));
  ok(html.includes('aria-checked="true"'));
  ok(html.includes("Active now"));
  ok(html.includes("The display may still turn off"));
});

Deno.test("sleep setting is disabled when the platform cannot assert idle sleep", () => {
  const html = renderToString(
    <SleepSettingControl
      settings={{
        prevent_sleep_while_working: false,
        prevent_sleep_supported: false,
        prevent_sleep_active: false,
      }}
    />,
  );

  ok(html.includes("available on macOS only"));
  ok(html.includes("disabled"));
});
