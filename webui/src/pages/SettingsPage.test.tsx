/// <reference lib="deno.ns" />
import { ok } from "node:assert/strict";
import { renderToString } from "react-dom/server";
import {
  SleepSettingControl,
  TailscaleAccessControl,
  type TailscaleSettings,
} from "./SettingsPage.tsx";

function tailscale(
  overrides: Partial<TailscaleSettings>,
): TailscaleSettings {
  return {
    state: "available",
    installed: true,
    connected: true,
    enabled: false,
    active: false,
    https_port: 8311,
    backend_target: "http://127.0.0.1:8311",
    direct_lan_access: false,
    ...overrides,
  };
}

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

Deno.test("Tailscale control offers one-click setup when the client is ready", () => {
  const html = renderToString(
    <TailscaleAccessControl status={tailscale({})} />,
  );

  ok(html.includes("Secure remote access"));
  ok(html.includes("Enable secure access"));
  ok(html.includes("tailnet-only HTTPS address"));
});

Deno.test("active Tailscale access exposes its HTTPS address and LAN warning", () => {
  const html = renderToString(
    <TailscaleAccessControl
      status={tailscale({
        state: "active",
        enabled: true,
        active: true,
        url: "https://pb.example.ts.net:8311/",
        direct_lan_access: true,
      })}
    />,
  );

  ok(html.includes("https://pb.example.ts.net:8311/"));
  ok(html.includes("Open pb"));
  ok(html.includes("Disable secure access"));
  ok(html.includes("direct LAN HTTP"));
});

Deno.test("Tailscale conflicts are visible and never offer overwrite", () => {
  const html = renderToString(
    <TailscaleAccessControl status={tailscale({ state: "conflict" })} />,
  );

  ok(html.includes("Port in use"));
  ok(html.includes("left it unchanged"));
  ok(!html.includes("Enable secure access"));
  ok(html.includes("Refresh"));
});

Deno.test("Tailscale authorization sends the user to the supplied approval URL", () => {
  const html = renderToString(
    <TailscaleAccessControl
      status={tailscale({
        state: "authorization_required",
        enabled: true,
        authorization_url: "https://login.tailscale.com/admin/feature/example",
      })}
    />,
  );

  ok(html.includes("Approve secure access"));
  ok(html.includes("Retry setup"));
  ok(html.includes("https://login.tailscale.com/admin/feature/example"));
  ok(html.includes('rel="noreferrer"'));
});
