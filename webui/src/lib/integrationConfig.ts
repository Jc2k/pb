import type {
  IntegrationConfigSchemaResponse,
  IntegrationJsonSchema,
  JsonSchemaProperty,
  PendingIntegrationInstall,
} from "../types/index";

export function integrationInstallPayload(
  pending: PendingIntegrationInstall,
  env: Record<string, string>,
  metadata?: IntegrationConfigSchemaResponse | null,
) {
  return {
    kind: pending.kind,
    container_image: pending.containerImage,
    name: pending.name,
    env,
    lsp_manifest: pending.kind === "lsp"
      ? metadata?.lsp_manifest ?? undefined
      : undefined,
  };
}

export function schemaPropertyType(property: JsonSchemaProperty): string {
  return Array.isArray(property.type) ? property.type.find((item) => item !== "null") || "string" : property.type || "string";
}

export function validateIntegrationConfig(schema: IntegrationJsonSchema | null | undefined, values: Record<string, string>) {
  const errors: Record<string, string> = {};
  if (!schema?.properties) return errors;
  const required = new Set(schema.required || []);
  for (const [key, property] of Object.entries(schema.properties)) {
    const value = values[key] || "";
    if (required.has(key) && !value.trim()) {
      errors[key] = "This field is required.";
      continue;
    }
    if (!value) continue;
    if (property.enum?.length && !property.enum.map(String).includes(value)) errors[key] = "Choose one of the allowed values.";
    if (property.minLength !== undefined && value.length < property.minLength) errors[key] = `Use at least ${property.minLength} characters.`;
    if (property.maxLength !== undefined && value.length > property.maxLength) errors[key] = `Use ${property.maxLength} characters or fewer.`;
    if (property.pattern) {
      try {
        if (!new RegExp(property.pattern).test(value)) errors[key] = "Use the expected format.";
      } catch {
        // Ignore invalid schema patterns from third-party images.
      }
    }
  }
  return errors;
}
