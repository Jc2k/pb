import { useEffect, useState } from "react";
import type { InstalledIntegration, IntegrationConfigSchemaResponse, MarketplaceIntegration, PendingIntegrationInstall } from "../types";
import { isIntegrationInstalled } from "../lib/helpers";
import { schemaPropertyType, validateIntegrationConfig } from "../lib/integrationConfig";

export function IntegrationList({
  marketplace,
  installed,
  installedIcon,
  emptyText,
  onInstall,
  onRemove,
  onConfigure,
}: {
  marketplace: MarketplaceIntegration[];
  installed: InstalledIntegration[];
  installedIcon: string;
  emptyText: string;
  onInstall: (item: MarketplaceIntegration) => void;
  onRemove?: (item: InstalledIntegration) => void;
  onConfigure: (item: MarketplaceIntegration | InstalledIntegration) => void;
}) {
  const available = marketplace.filter((item) => !isIntegrationInstalled(item, installed));
  const hasItems = installed.length > 0 || available.length > 0;

  return (
    <div className="mcp-store-list" data-testid="mcp-store-list">
      {!hasItems ? (
        <div className="mcp-store-empty text-secondary small">{emptyText}</div>
      ) : (
        <>
          {installed.map((item) => (
            <div key={`installed:${item.kind}:${item.name}`} className="mcp-store-row is-installed">
              <div className="mcp-store-icon"><i className={installedIcon}></i></div>
              <div className="mcp-store-copy">
                <strong>{item.name}</strong>
                <span>{item.container_image}</span>
                <small>{item.disabled ? "Disabled for new sessions" : "Ready for new sessions"}</small>
              </div>
              <div className="mcp-store-actions">
                <button className="mcp-icon-btn" title={`Configure ${item.name}`} aria-label={`Configure ${item.name}`} onClick={() => onConfigure(item)}>
                  <i className="bi bi-gear"></i>
                </button>
                {onRemove && <button className="mcp-icon-btn danger" title={`Remove ${item.name}`} aria-label={`Remove ${item.name}`} onClick={() => onRemove(item)}><i className="bi bi-trash"></i></button>}
              </div>
            </div>
          ))}
          {available.map((item) => (
            <div key={`${item.kind}:${item.name}`} className="mcp-store-row">
              <div className="mcp-store-icon image-icon"><img src={item.icon_url} alt="" /></div>
              <div className="mcp-store-copy">
                <strong>{item.name}</strong>
                <span>{item.container_image}</span>
                <small>{item.description || item.container_image}</small>
              </div>
              <div className="mcp-store-actions">
                <button className="mcp-icon-btn" title={`Configure ${item.name}`} aria-label={`Configure ${item.name}`} onClick={() => onConfigure(item)}>
                  <i className="bi bi-gear"></i>
                </button>
                <button className="mcp-icon-btn add" title={`Install ${item.name}`} aria-label={`Install ${item.name}`} onClick={() => onInstall(item)}><i className="bi bi-plus-lg"></i></button>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

export function IntegrationConfigForm({
  pending,
  schemaResponse,
  loading,
  error,
  onCancel,
  onInstall,
}: {
  pending: PendingIntegrationInstall;
  schemaResponse?: IntegrationConfigSchemaResponse | null;
  loading: boolean;
  error?: string;
  onCancel: () => void;
  onInstall: (env: Record<string, string>) => void;
}) {
  const schema = schemaResponse?.schema || null;
  const [values, setValues] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const next: Record<string, string> = {};
    for (const [key, property] of Object.entries(schema?.properties ?? {})) {
      if (pending.env?.[key] !== undefined) next[key] = pending.env[key];
      else if (property.default !== undefined) next[key] = String(property.default);
    }
    setValues(next);
    setTouched({});
  }, [schemaResponse?.container_image, pending.env]);

  const validationErrors = validateIntegrationConfig(schema, values);
  const fields = Object.entries(schema?.properties ?? {});
  const missingLspManifest = pending.kind === "lsp" && !schemaResponse?.lsp_manifest;
  const canSubmit = !loading && !missingLspManifest && Object.keys(validationErrors).length === 0;

  return (
    <form className="integration-config-sheet" onSubmit={(event) => {
      event.preventDefault();
      const allTouched = Object.fromEntries(fields.map(([key]) => [key, true]));
      setTouched(allTouched);
      const nonEmptyValues: Record<string, string> = Object.fromEntries(
        Object.entries(values).filter(([, value]) => value.trim() !== ""),
      );
      if (canSubmit) onInstall(nonEmptyValues);
    }}>
      <div className="integration-config-head">
        <h3>Configure {pending.name || pending.containerImage}</h3>
        <button type="button" className="mcp-modal-close" aria-label="Close configuration" onClick={onCancel}><i className="bi bi-x-lg"></i></button>
      </div>
      {loading && <div className="alert alert-info py-2 small"><span className="spinner-border spinner-border-sm me-2" />Fetching container schema annotation…</div>}
      {error && pending.kind !== "lsp" && <div className="alert alert-warning py-2 small">Could not fetch the schema annotation, so this integration can be installed without extra fields. {error}</div>}
      {error && pending.kind === "lsp" && <div className="alert alert-danger py-2 small">Could not verify the package's typed LSP manifest, so installation is blocked. {error}</div>}
      {!loading && !error && missingLspManifest && <div className="alert alert-danger py-2 small">This image does not publish the typed LSP manifest required for marketplace installation.</div>}
      {!loading && !error && !missingLspManifest && fields.length === 0 && <div className="alert alert-secondary py-2 small">This container does not publish a configuration schema annotation, so it will be installed without extra environment values.</div>}
      {schema?.description && <p className="small text-secondary">{schema.description}</p>}
      <div className="integration-config-grid">
        {fields.map(([key, property]) => {
          const label = property.title || key;
          const type = schemaPropertyType(property);
          const isInvalid = Boolean(touched[key] && validationErrors[key]);
          return (
            <div className="integration-config-field" key={key}>
              <label className="form-label small fw-semibold" htmlFor={`integration-config-${key}`}>{label}{schema?.required?.includes(key) && <span className="text-danger ms-1">*</span>}</label>
              {property.enum?.length ? (
                <select id={`integration-config-${key}`} className={`form-select ${isInvalid ? "is-invalid" : ""}`} value={values[key] || ""} onBlur={() => setTouched((prev) => ({ ...prev, [key]: true }))} onChange={(event) => setValues((prev) => ({ ...prev, [key]: event.target.value }))}>
                  <option value="">Choose…</option>
                  {property.enum.map((option) => <option key={String(option)} value={String(option)}>{String(option)}</option>)}
                </select>
              ) : (
                <input id={`integration-config-${key}`} className={`form-control ${isInvalid ? "is-invalid" : ""}`} type={type === "integer" || type === "number" ? "number" : type === "boolean" ? "checkbox" : "text"} checked={type === "boolean" ? values[key] === "true" : undefined} value={type === "boolean" ? undefined : values[key] || ""} onBlur={() => setTouched((prev) => ({ ...prev, [key]: true }))} onChange={(event) => setValues((prev) => ({ ...prev, [key]: type === "boolean" ? String(event.target.checked) : event.target.value }))} />
              )}
              {property.description && <div className="form-text">{property.description}</div>}
              {isInvalid && <div className="invalid-feedback d-block">{validationErrors[key]}</div>}
            </div>
          );
        })}
      </div>
      <p className="credentials-note"><i className="bi bi-lock"></i> Credentials are stored locally on your machine.</p>
      <div className="integration-config-actions">
        <button className="btn btn-primary" disabled={!canSubmit}>
          {pending.installed ? "Save configuration" : fields.length ? "Install with config" : "Install"}
        </button>
      </div>
    </form>
  );
}
