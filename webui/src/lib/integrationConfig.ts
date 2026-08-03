import type {
  InstalledIntegration,
  IntegrationConfigSchemaResponse,
  IntegrationJsonSchema,
  JsonSchemaProperty,
  LspPackageManifest,
  MarketplaceIntegration,
  PendingIntegrationInstall,
} from "../types/index";

function responseJson(text: string, label: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(`${label} is not valid JSON`);
  }
}

function responseRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireResponseKeys(
  value: Record<string, unknown>,
  keys: readonly string[],
  label: string,
): void {
  const missing = keys.find((key) => !Object.hasOwn(value, key));
  if (missing) throw new Error(`${label} is missing field ${missing}`);
}

function exactResponseKeys(
  value: Record<string, unknown>,
  keys: readonly string[],
  label: string,
): void {
  const allowed = new Set(keys);
  const unexpected = Object.keys(value).find((key) => !allowed.has(key));
  if (unexpected) {
    throw new Error(`${label} contains unknown field ${unexpected}`);
  }
}

function responseString(value: unknown, label: string): string {
  if (typeof value !== "string") {
    throw new Error(`${label} must be a string`);
  }
  return value;
}

function responseStringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value.map((entry, index) =>
    responseString(entry, `${label} ${index}`)
  );
}

function integrationKind(value: unknown, label: string): "mcp" | "lsp" {
  if (value !== "mcp" && value !== "lsp") {
    throw new Error(`${label} must be mcp or lsp`);
  }
  return value;
}

function marketplaceIntegration(
  value: unknown,
  label: string,
): MarketplaceIntegration {
  const entry = responseRecord(value, label);
  const fields = [
    "name",
    "kind",
    "description",
    "icon_url",
    "repo_url",
    "container_image",
  ] as const;
  exactResponseKeys(entry, fields, label);
  requireResponseKeys(entry, fields, label);
  return {
    name: responseString(entry.name, `${label} name`),
    kind: integrationKind(entry.kind, `${label} kind`),
    description: responseString(entry.description, `${label} description`),
    icon_url: responseString(entry.icon_url, `${label} icon_url`),
    repo_url: responseString(entry.repo_url, `${label} repo_url`),
    container_image: responseString(
      entry.container_image,
      `${label} container_image`,
    ),
  };
}

function installedIntegration(
  value: unknown,
  label: string,
): InstalledIntegration {
  const entry = responseRecord(value, label);
  const fields = [
    "name",
    "kind",
    "container_image",
    "source_container_image",
    "verified_manifest_digest",
    "env",
    "disabled",
    "status",
  ] as const;
  exactResponseKeys(entry, fields, label);
  requireResponseKeys(
    entry,
    ["name", "kind", "container_image", "env", "disabled", "status"],
    label,
  );
  const environment = responseRecord(entry.env, `${label} env`);
  const env = Object.fromEntries(
    Object.entries(environment).map(([key, value]) => [
      key,
      responseString(value, `${label} env ${key}`),
    ]),
  );
  if (typeof entry.disabled !== "boolean") {
    throw new Error(`${label} disabled must be a boolean`);
  }
  if (
    entry.status !== "ready" && entry.status !== "disabled" &&
    entry.status !== "unavailable" && entry.status !== "legacy_unverified"
  ) {
    throw new Error(`${label} status is invalid`);
  }
  return {
    name: responseString(entry.name, `${label} name`),
    kind: integrationKind(entry.kind, `${label} kind`),
    container_image: responseString(
      entry.container_image,
      `${label} container_image`,
    ),
    ...(entry.source_container_image === undefined ? {} : {
      source_container_image: responseString(
        entry.source_container_image,
        `${label} source_container_image`,
      ),
    }),
    ...(entry.verified_manifest_digest === undefined ? {} : {
      verified_manifest_digest: responseString(
        entry.verified_manifest_digest,
        `${label} verified_manifest_digest`,
      ),
    }),
    env,
    disabled: entry.disabled,
    status: entry.status,
  };
}

function integrationSchema(
  value: unknown,
  label: string,
): IntegrationJsonSchema | null {
  if (value === null) return null;
  const schema = responseRecord(value, label);
  for (const field of ["title", "description"]) {
    if (schema[field] !== undefined && typeof schema[field] !== "string") {
      throw new Error(`${label} ${field} must be a string`);
    }
  }
  if (schema.type !== undefined && typeof schema.type !== "string") {
    throw new Error(`${label} type must be a string`);
  }
  if (schema.required !== undefined) {
    responseStringArray(schema.required, `${label} required`);
  }
  if (schema.properties !== undefined) {
    const properties = responseRecord(schema.properties, `${label} properties`);
    for (const [name, rawProperty] of Object.entries(properties)) {
      const property = responseRecord(rawProperty, `${label} property ${name}`);
      if (
        property.type !== undefined && typeof property.type !== "string" &&
        (!Array.isArray(property.type) ||
          property.type.some((entry) => typeof entry !== "string"))
      ) {
        throw new Error(`${label} property ${name} type is invalid`);
      }
      for (const field of ["title", "description", "pattern"]) {
        if (
          property[field] !== undefined && typeof property[field] !== "string"
        ) {
          throw new Error(
            `${label} property ${name} ${field} must be a string`,
          );
        }
      }
      for (const field of ["minLength", "maxLength"]) {
        if (
          property[field] !== undefined &&
          (!Number.isSafeInteger(property[field]) ||
            (property[field] as number) < 0)
        ) {
          throw new Error(`${label} property ${name} ${field} is invalid`);
        }
      }
      const primitive = (entry: unknown) =>
        entry === null || typeof entry === "string" ||
        typeof entry === "number" ||
        typeof entry === "boolean";
      if (property.default !== undefined && !primitive(property.default)) {
        throw new Error(`${label} property ${name} default is invalid`);
      }
      if (
        property.enum !== undefined &&
        (!Array.isArray(property.enum) ||
          property.enum.some((entry) => !primitive(entry)))
      ) {
        throw new Error(`${label} property ${name} enum is invalid`);
      }
    }
  }
  return schema as IntegrationJsonSchema;
}

function lspPackageManifest(
  value: unknown,
  label: string,
): LspPackageManifest | null {
  if (value === null) return null;
  const manifest = responseRecord(value, label);
  exactResponseKeys(manifest, ["version", "kind", "server"], label);
  requireResponseKeys(manifest, ["version", "kind", "server"], label);
  if (manifest.version !== 1) throw new Error(`${label} version must be 1`);
  if (manifest.kind !== "lsp") throw new Error(`${label} kind must be lsp`);
  const server = responseRecord(manifest.server, `${label} server`);
  const serverFields = [
    "args",
    "language_ids",
    "initialization_options",
    "workspace_access",
    "network_access",
    "cache_ids",
  ] as const;
  exactResponseKeys(server, serverFields, `${label} server`);
  requireResponseKeys(server, serverFields, `${label} server`);
  if (server.workspace_access !== "read_only") {
    throw new Error(`${label} server workspace_access must be read_only`);
  }
  if (server.network_access !== "none") {
    throw new Error(`${label} server network_access must be none`);
  }
  return {
    version: 1,
    kind: "lsp",
    server: {
      args: responseStringArray(server.args, `${label} server args`),
      language_ids: responseStringArray(
        server.language_ids,
        `${label} server language_ids`,
      ),
      initialization_options: server.initialization_options,
      workspace_access: "read_only",
      network_access: "none",
      cache_ids: responseStringArray(
        server.cache_ids,
        `${label} server cache_ids`,
      ),
    },
  };
}

export function parseMarketplaceIntegrationsJson(
  text: string,
): MarketplaceIntegration[] {
  const response = responseJson(text, "integration marketplace response");
  if (!Array.isArray(response)) {
    throw new Error("integration marketplace response must be an array");
  }
  return response.map((entry, index) =>
    marketplaceIntegration(entry, `integration marketplace entry ${index}`)
  );
}

export function parseInstalledIntegrationsJson(
  text: string,
): InstalledIntegration[] {
  const response = responseJson(text, "installed integrations response");
  if (!Array.isArray(response)) {
    throw new Error("installed integrations response must be an array");
  }
  return response.map((entry, index) =>
    installedIntegration(entry, `installed integration ${index}`)
  );
}

export function parseIntegrationConfigSchemaResponseJson(
  text: string,
): IntegrationConfigSchemaResponse {
  const response = responseRecord(
    responseJson(text, "integration schema response"),
    "integration schema response",
  );
  const fields = [
    "container_image",
    "source_container_image",
    "manifest_digest",
    "annotation",
    "schema",
    "lsp_manifest_annotation",
    "lsp_manifest",
  ] as const;
  exactResponseKeys(response, fields, "integration schema response");
  requireResponseKeys(response, fields, "integration schema response");
  return {
    container_image: responseString(
      response.container_image,
      "integration schema response container_image",
    ),
    source_container_image: responseString(
      response.source_container_image,
      "integration schema response source_container_image",
    ),
    manifest_digest: responseString(
      response.manifest_digest,
      "integration schema response manifest_digest",
    ),
    annotation: responseString(
      response.annotation,
      "integration schema response annotation",
    ),
    schema: integrationSchema(
      response.schema,
      "integration schema response schema",
    ),
    lsp_manifest_annotation: responseString(
      response.lsp_manifest_annotation,
      "integration schema response lsp_manifest_annotation",
    ),
    lsp_manifest: lspPackageManifest(
      response.lsp_manifest,
      "integration schema response lsp_manifest",
    ),
  };
}

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
