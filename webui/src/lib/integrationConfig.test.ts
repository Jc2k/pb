/// <reference lib="deno.ns" />
import { deepEqual, equal, ok } from "node:assert/strict";
import type { IntegrationConfigSchemaResponse, IntegrationJsonSchema } from "../types/index";
import { integrationInstallPayload, schemaPropertyType, validateIntegrationConfig } from "./integrationConfig.ts";

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

  deepEqual(validateIntegrationConfig(schema, { token: "", mode: "admin", slug: "Bad Slug" }), {
    token: "This field is required.",
    mode: "Choose one of the allowed values.",
    slug: "Use the expected format.",
  });

  deepEqual(validateIntegrationConfig(schema, { token: "abcd", mode: "read", slug: "pb-web" }), {});
});

Deno.test("integrationInstallPayload carries typed LSP defaults without pinning a runtime", () => {
  const metadata: IntegrationConfigSchemaResponse = {
    container_image: "ghcr.io/crunchy-pb/rust-analyzer-lsp:latest",
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

  const payload = integrationInstallPayload({
    kind: "lsp",
    containerImage: metadata.container_image,
    name: "rust-analyzer",
  }, {}, metadata);

  equal(payload.lsp_manifest?.server.language_ids[0], "rust");
  ok(!("runtime" in payload));
});
