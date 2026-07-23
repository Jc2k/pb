/// <reference lib="deno.ns" />
import { deepEqual, equal, ok } from "node:assert/strict";
import type {
  IntegrationConfigSchemaResponse,
  IntegrationJsonSchema,
} from "../types/index";
import {
  integrationApiError,
  integrationInstallPayload,
  schemaPropertyType,
  validateIntegrationConfig,
} from "./integrationConfig.ts";

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

Deno.test("integrationInstallPayload carries typed LSP defaults without pinning a runtime", () => {
  const metadata: IntegrationConfigSchemaResponse = {
    container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer@sha256:abc",
    source_container_image: "ghcr.io/crunchy-pb/lsp-rust-analyzer:latest",
    manifest_digest: "sha256:abc",
    annotation: "uk.unrtd.pb.integration.config-schema",
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
        lsp_manifest_annotation: "manifest",
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
      lsp_manifest_annotation: "manifest",
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
