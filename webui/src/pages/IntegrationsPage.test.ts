/// <reference lib="deno.ns" />
import { equal, ok } from "node:assert/strict";

Deno.test("global integration mutations apply their authoritative response", async () => {
  const source = await Deno.readTextFile(
    "webui/src/pages/IntegrationsPage.tsx",
  );
  const mutations = source.slice(
    source.indexOf("const removeIntegration"),
    source.indexOf("const cancelIntegration"),
  );

  equal(mutations.match(/uniqueInstalledIntegrations/g)?.length, 2);
  equal(mutations.match(/setInstalled\(nextInstalled\)/g)?.length, 2);
  ok(!mutations.includes("fetchInstalledIntegrations"));
});
