import { ok } from "node:assert/strict";
import { renderToString } from "react-dom/server";
import { IntegrationConfigForm, IntegrationList } from "./Integration.tsx";
import type { InstalledIntegration, MarketplaceIntegration } from "../types/index.ts";

Deno.test("IntegrationList renders mockup-style MCP store actions", () => {
  const installed: InstalledIntegration[] = [{
    name: "mcp-sentry",
    kind: "mcp",
    container_image: "ghcr.io/crunchy-pb/mcp-sentry:latest",
    disabled: false,
  }];
  const marketplace: MarketplaceIntegration[] = [{
    name: "figma-assets",
    kind: "mcp",
    description: "Access Figma files, pages, and assets for UI context.",
    icon_url: "/figma.png",
    repo_url: "https://example.com/figma-assets",
    container_image: "ghcr.io/localagent/figma-assets:latest",
  }];

  const html = renderToString(
    <IntegrationList
      marketplace={marketplace}
      installed={installed}
      installedIcon="bi bi-plug"
      emptyText="No MCP servers match your filters."
      onInstall={() => {}}
      onConfigure={() => {}}
      onRemove={() => {}}
    />,
  );

  ok(html.includes("mcp-store-list"));
  ok(html.includes("mcp-sentry"));
  ok(html.includes("aria-label=\"Configure mcp-sentry\""));
  ok(html.includes("aria-label=\"Remove mcp-sentry\""));
  ok(html.includes("aria-label=\"Install figma-assets\""));
});

Deno.test("IntegrationConfigForm blocks an LSP image without a typed package manifest", () => {
  const html = renderToString(
    <IntegrationConfigForm
      pending={{
        kind: "lsp",
        containerImage: "ghcr.io/crunchy-pb/legacy-lsp:latest",
      }}
      schemaResponse={{
        container_image: "ghcr.io/crunchy-pb/legacy-lsp:latest",
        annotation: "uk.unrtd.pb.integration.config-schema",
        lsp_manifest_annotation: "uk.unrtd.pb.integration.lsp-manifest",
      }}
      loading={false}
      onCancel={() => {}}
      onInstall={() => {}}
    />,
  );

  ok(html.includes("typed LSP manifest required"));
  ok(html.includes("disabled"));
});
