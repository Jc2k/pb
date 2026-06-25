import { deepEqual, equal } from "node:assert/strict";
import type { IntegrationJsonSchema } from "../types/index";
import { schemaPropertyType, validateIntegrationConfig } from "./integrationConfig.ts";

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
