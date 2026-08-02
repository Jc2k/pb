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
  if (
    metadata && metadata.source_container_image !== pending.containerImage
  ) {
    throw new Error(
      "Integration metadata no longer matches the selected container image. Inspect the image again before installing.",
    );
  }
  return {
    kind: pending.kind,
    container_image: metadata?.container_image ?? pending.containerImage,
    source_container_image: pending.sourceContainerImage ??
      metadata?.source_container_image,
    name: pending.name,
    env,
    lsp_manifest: pending.kind === "lsp"
      ? metadata?.lsp_manifest ?? undefined
      : undefined,
  };
}

export async function apiErrorMessage(
  response: Response,
  fallback: string,
): Promise<string> {
  const text = await response.text().catch(() => "");
  if (text) {
    try {
      const decoded = JSON.parse(text) as { error?: unknown };
      if (typeof decoded.error === "string" && decoded.error.trim()) {
        return decoded.error;
      }
    } catch {
      if (text.trim()) return text.trim();
    }
  }
  return response.status ? `${fallback} (HTTP ${response.status})` : fallback;
}

export const integrationApiError = apiErrorMessage;

export function schemaPropertyType(property: JsonSchemaProperty): string {
  return Array.isArray(property.type)
    ? property.type.find((item) => item !== "null") || "string"
    : property.type || "string";
}

export function validateIntegrationConfig(
  schema: IntegrationJsonSchema | null | undefined,
  values: Record<string, string>,
) {
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
    if (property.enum?.length && !property.enum.map(String).includes(value)) {
      errors[key] = "Choose one of the allowed values.";
    }
    if (property.minLength !== undefined && value.length < property.minLength) {
      errors[key] = `Use at least ${property.minLength} characters.`;
    }
    if (property.maxLength !== undefined && value.length > property.maxLength) {
      errors[key] = `Use ${property.maxLength} characters or fewer.`;
    }
    if (property.pattern) {
      try {
        if (!new RegExp(property.pattern).test(value)) {
          errors[key] = "Use the expected format.";
        }
      } catch {
        // Ignore invalid schema patterns from third-party images.
      }
    }
  }
  return errors;
}
