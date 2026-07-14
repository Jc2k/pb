/// <reference lib="deno.ns" />
import { ok } from "node:assert/strict";
import { renderToString } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { PageShell } from "./PageShell.tsx";

Deno.test("PageShell exposes consistent tablet and phone navigation", () => {
  const html = renderToString(
    <MemoryRouter initialEntries={["/projects"]}>
      <PageShell>
        <h1>Projects</h1>
      </PageShell>
    </MemoryRouter>,
  );

  ok(html.includes("tablet-nav"));
  ok(html.includes("mobile-nav"));
  ok(html.includes('aria-label="Primary navigation"'));
  ok(html.includes('href="/projects"'));
  ok(!html.includes('href="/settings"'));
});

Deno.test("responsive shell accounts for device safe areas and stable route changes", async () => {
  const css = await Deno.readTextFile("webui/src/app.css");
  const sessionCss = await Deno.readTextFile("webui/src/session.css");
  const app = await Deno.readTextFile("webui/src/App.tsx");

  ok(css.includes("env(safe-area-inset-top)"));
  ok(css.includes("env(safe-area-inset-right)"));
  ok(css.includes("env(safe-area-inset-bottom)"));
  ok(css.includes("env(safe-area-inset-left)"));
  ok(css.includes("scrollbar-gutter: stable"));
  ok(css.includes(".mobile-nav"));
  ok(sessionCss.includes("height: 100dvh"));
  ok(app.includes("<RouteReset />"));
});
