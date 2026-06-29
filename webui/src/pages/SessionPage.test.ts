/// <reference lib="deno.ns" />
import { ok } from "node:assert/strict";

Deno.test("paused session composer keeps resume action at intrinsic width", async () => {
  const markup = await Deno.readTextFile("webui/src/pages/SessionPage.tsx");
  const css = await Deno.readTextFile("webui/src/session.css");

  ok(markup.includes('className="composer paused-composer"'));
  ok(markup.includes('className="btn btn-warning composer-action"'));
  ok(css.includes(".composer .btn.composer-action"));
  ok(css.includes("width: auto;"));
  ok(css.includes("white-space: nowrap;"));
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
  ok(css.includes(".session-header .share-action,"));
});
