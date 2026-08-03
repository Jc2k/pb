/// <reference lib="deno.ns" />
import { deepEqual, equal, ok, throws } from "node:assert/strict";
import type {
  IntegrationConfigSchemaResponse,
  IntegrationJsonSchema,
} from "../types/index";
import {
  integrationApiError,
  integrationInstallPayload,
  parseInstalledIntegrationsJson,
  parseIntegrationConfigSchemaResponseJson,
  parseMarketplaceIntegrationsJson,
  schemaPropertyType,
  validateIntegrationConfig,
} from "./integrationConfig.ts";

function rustStructFields(source: string, name: string): string[] {
  const start = source.indexOf(`pub struct ${name}`);
  if (start < 0) throw new Error(`missing Rust struct ${name}`);
  const end = source.indexOf("\n}", start);
  return [
    ...source.slice(start, end).matchAll(/^    pub ([a-z_][a-z0-9_]*):/gm),
  ].map((match) => match[1]);
}

function typescriptInterfaceFields(source: string, name: string): string[] {
  const start = source.indexOf(`export interface ${name}`);
  if (start < 0) throw new Error(`missing TypeScript interface ${name}`);
  const end = source.indexOf("\n}", start);
  return [...source.slice(start, end).matchAll(/^  ([a-z_][a-z0-9_]*)\??:/gm)]
    .map((match) => match[1]);
}

Deno.test("Rust and TypeScript integration response fields stay exact", async () => {
  const [rust, types] = await Promise.all([
    Deno.readTextFile("src/integrations.rs"),
    Deno.readTextFile("webui/src/types/index.ts"),
  ]);
  for (const name of ["MarketplaceIntegration", "InstalledIntegration"]) {
    deepEqual(
      typescriptInterfaceFields(types, name).sort(),
      rustStructFields(rust, name).sort(),
      `${name} fields drifted across the Rust/TypeScript boundary`,
    );
  }
  deepEqual(
    typescriptInterfaceFields(types, "IntegrationConfigSchemaResponse").sort(),
    rustStructFields(rust, "IntegrationConfigSchema").sort(),
    "integration schema response fields drifted across the Rust/TypeScript boundary",
  );
  deepEqual(
    typescriptInterfaceFields(types, "LspPackageManifest").sort(),
    rustStructFields(rust, "LspPackageManifest").sort(),
    "LSP manifest fields drifted across the Rust/TypeScript boundary",
  );
  deepEqual(
    typescriptInterfaceFields(types, "LspPackageServerConfig").sort(),
    rustStructFields(rust, "LspPackageServerConfig").sort(),
    "LSP server fields drifted across the Rust/TypeScript boundary",
  );
});

Deno.test("schemaPropertyType chooses a non-null type from nullable schemas", () => {
  equal(schemaPropertyType({ type: ["null", "string"] }), "string");
  equal(schemaPropertyType({}), "string");
});

Deno.test("validateIntegrationConfig reports required and string constraint errors", () => {
  const schema: IntegrationJsonSchema = {
    required: ["token"],
    properties: {
      token: { type: "string", minLength: 4 },
      mode: { type: "string", enum: ["read", "write"] },
      slug: { type: "string", pattern: "^[a-z-]+$", maxLength: 8 },
    },
  };

  deepEqual(
    validateIntegrationConfig(schema, {
      token: "",
      mode: "admin",
      slug: "Bad Slug",
    }),
    {
      token: "This field is required.",
      mode: "Choose one of the allowed values.",
      slug: "Use the expected format.",
    },
  );

  deepEqual(
    validateIntegrationConfig(schema, {
      token: "abcd",
      mode: "read",
      slug: "pb-web",
    }),
    {},
  );
});

Deno.test("integration responses use exact current runtime contracts", () => {
  deepEqual(
    parseMarketplaceIntegrationsJson(JSON.stringify([{
      name: "mcp-example",
      kind: "mcp",
      description: "",
      icon_url: "https://example.test/icon.png",
      repo_url: "https://example.test/repo",
      container_image: "ghcr.io/crunchy-pb/mcp-example:latest",
    }])),
    [{
      name: "mcp-example",
      kind: "mcp",
      description: "",
      icon_url: "https://example.test/icon.png",
      repo_url: "https://example.test/repo",
      container_image: "ghcr.io/crunchy-pb/mcp-example:latest",
    }],
  );
  deepEqual(
    parseInstalledIntegrationsJson(JSON.stringify([{
      name: "mcp-example",
      kind: "mcp",
      container_image: "ghcr.io/crunchy-pb/mcp-example@sha256:abc",
      env: { TOKEN: "" },
      disabled: false,
      status: "ready",
    }])),
    [{
      name: "mcp-example",
      kind: "mcp",
      container_image: "ghcr.io/crunchy-pb/mcp-example@sha256:abc",
      env: { TOKEN: "" },
      disabled: false,
      status: "ready",
    }],
  );
  const schema = parseIntegrationConfigSchemaResponseJson(JSON.stringify({
    container_image: "ghcr.io/crunchy-pb/lsp-example@sha256:abc",
    source_container_image: "ghcr.io/crunchy-pb/lsp-example:latest",
    manifest_digest: "sha256:abc",
    annotation: "config-schema",
    schema: {
      properties: {
        token: { type: ["string", "null"], default: null },
      },
    },
    lsp_manifest_annotation: "lsp-manifest",
    lsp_manifest: {
      version: 1,
      kind: "lsp",
      server: {
        args: ["--stdio"],
        language_ids: ["rust"],
        initialization_options: null,
        workspace_access: "read_only",
        network_access: "none",
        cache_ids: [],
      },
    },
  }));
  equal(schema.lsp_manifest?.server.language_ids[0], "rust");
  equal(schema.schema?.properties?.token?.default, null);

  throws(
    () =>
      parseInstalledIntegrationsJson(JSON.stringify([{
        name: "old",
        kind: "mcp",
        container_image: "old",
        disabled: false,
      }])),
    /installed integration 0 is missing field env/,
  );
  throws(
    () =>
      parseMarketplaceIntegrationsJson(JSON.stringify([{
        name: "future",
        kind: "mcp",
        description: "future",
        icon_url: "icon",
        repo_url: "repo",
        container_image: "image",
        refresh: true,
      }])),
    /integration marketplace entry 0 contains unknown field refresh/,
  );
  throws(
    () =>
      parseIntegrationConfigSchemaResponseJson(JSON.stringify({
        container_image: "image",
        source_container_image: "source",
        manifest_digest: "digest",
        annotation: "schema",
        schema: null,
        lsp_manifest_annotation: "manifest",
      })),
    /integration schema response is missing field lsp_manifest/,
  );
});

Deno.test("integrationInstallPayload carries typed LSP defaults without pinning a runtime", () => {
  const metadata: IntegrationConfigSchemaResponse = {
    container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:abc",
    source_container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer:latest",
    manifest_digest: "sha256:abc",
    annotation: "uk.unrtd.pb.integration.config-schema",
    schema: null,
    lsp_manifest_annotation: "uk.unrtd.pb.integration.lsp-manifest",
    lsp_manifest: {
      version: 1,
      kind: "lsp",
      server: {
        args: [],
        language_ids: ["rust"],
        initialization_options: { checkOnSave: false },
        workspace_access: "read_only",
        network_access: "none",
        cache_ids: [],
      },
    },
  };

  const payload = integrationInstallPayload(
    {
      kind: "lsp",
      containerImage: metadata.source_container_image!,
      name: "rust-analyzer",
    },
    {},
    metadata,
  );

  equal(payload.lsp_manifest?.server.language_ids[0], "rust");
  equal(
    payload.container_image,
    "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:abc",
  );
  equal(
    payload.source_container_image,
    "ghcr.io/crunchy-pb/lsp-rust-analyzer:latest",
  );
  ok(!("runtime" in payload));
});

Deno.test("integrationInstallPayload rejects metadata fetched for another image", () => {
  let error: unknown;
  try {
    integrationInstallPayload(
      {
        kind: "mcp",
        containerImage: "ghcr.io/crunchy-pb/mcp-current:latest",
        name: "mcp-current",
      },
      {},
      {
        container_image: "ghcr.io/crunchy-pb/mcp-stale@sha256:abc",
        source_container_image: "ghcr.io/crunchy-pb/mcp-stale:latest",
        manifest_digest: "sha256:abc",
        annotation: "schema",
        schema: null,
        lsp_manifest_annotation: "manifest",
        lsp_manifest: null,
      },
    );
  } catch (caught) {
    error = caught;
  }
  ok(error instanceof Error);
  ok(error.message.includes("no longer matches"));
});

Deno.test("integrationInstallPayload keeps the saved source while configuring a pinned image", () => {
  const payload = integrationInstallPayload(
    {
      kind: "lsp",
      containerImage: "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:old",
      sourceContainerImage: "ghcr.io/crunchy-pb/lsp-rust-analyzer:stable",
      operation: "configure",
    },
    {},
    {
      container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:old",
      source_container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:old",
      manifest_digest: "sha256:old",
      annotation: "schema",
      schema: null,
      lsp_manifest_annotation: "manifest",
      lsp_manifest: null,
    },
  );

  equal(
    payload.container_image,
    "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:old",
  );
  equal(
    payload.source_container_image,
    "ghcr.io/crunchy-pb/lsp-rust-analyzer:stable",
  );
});

Deno.test("integrationApiError surfaces structured server failures and safe fallbacks", async () => {
  equal(
    await integrationApiError(
      new Response(JSON.stringify({ error: "registry target is private" }), {
        status: 400,
      }),
      "Could not install",
    ),
    "registry target is private",
  );
  equal(
    await integrationApiError(
      new Response("", { status: 503 }),
      "Could not install",
    ),
    "Could not install (HTTP 503)",
  );
});
