import { ok } from "node:assert/strict";
import { renderToString } from "react-dom/server";
import { IntegrationConfigForm, IntegrationList } from "./Integration.tsx";
import type {
  InstalledIntegration,
  MarketplaceIntegration,
} from "../types/index.ts";

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
  ok(html.includes('aria-label="Configure mcp-sentry"'));
  ok(html.includes('aria-label="Remove mcp-sentry"'));
  ok(html.includes('aria-label="Install figma-assets"'));
});

Deno.test("IntegrationList offers an explicit upgrade only when a source is available", () => {
  const installed: InstalledIntegration[] = [{
    name: "lsp-rust-analyzer",
    kind: "lsp",
    container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:abc",
    source_container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer:latest",
    verified_manifest_digest: "sha256:abc",
    disabled: false,
  }];
  const html = renderToString(
    <IntegrationList
      marketplace={[]}
      installed={installed}
      installedIcon="bi bi-code-slash"
      emptyText="Empty"
      onInstall={() => {}}
      onConfigure={() => {}}
      onUpgrade={() => {}}
    />,
  );

  ok(html.includes('aria-label="Configure lsp-rust-analyzer"'));
  ok(html.includes('aria-label="Upgrade lsp-rust-analyzer"'));
});

Deno.test("IntegrationList does not let Configure implicitly upgrade a legacy image", () => {
  const installed: InstalledIntegration[] = [{
    name: "legacy-lsp",
    kind: "lsp",
    container_image: "ghcr.io/crunchy-pb/legacy-lsp:latest",
    disabled: false,
    status: "legacy_unverified",
  }];
  const html = renderToString(
    <IntegrationList
      marketplace={[]}
      installed={installed}
      installedIcon="bi bi-code-slash"
      emptyText="Empty"
      onInstall={() => {}}
      onConfigure={() => {}}
      onUpgrade={() => {}}
    />,
  );

  ok(!html.includes('aria-label="Configure legacy-lsp"'));
  ok(html.includes('aria-label="Upgrade legacy-lsp"'));
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

Deno.test("IntegrationConfigForm keeps an upgrade open and displays submit failures", () => {
  const html = renderToString(
    <IntegrationConfigForm
      pending={{
        kind: "lsp",
        containerImage: "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:new",
        sourceContainerImage: "ghcr.io/crunchy-pb/lsp-rust-analyzer:latest",
        name: "lsp-rust-analyzer",
        installed: true,
        operation: "upgrade",
      }}
      schemaResponse={{
        container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:new",
        source_container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer:latest",
        manifest_digest: "sha256:new",
        annotation: "schema",
        lsp_manifest_annotation: "manifest",
        lsp_manifest: {
          version: 1,
          kind: "lsp",
          server: {
            args: [],
            language_ids: ["rust"],
            workspace_access: "read_only",
            network_access: "none",
            cache_ids: [],
          },
        },
      }}
      loading={false}
      submitError="Image pull failed"
      onCancel={() => {}}
      onInstall={() => {}}
    />,
  );

  ok(html.includes("Image pull failed"));
  ok(html.includes("<h3>Upgrade"));
  ok(html.includes("lsp-rust-analyzer"));
  ok(html.includes(">Upgrade<"));
});
