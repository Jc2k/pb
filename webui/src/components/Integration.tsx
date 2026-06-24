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
    <div className="project-list session-list list-group mb-4">
      {!hasItems ? (
        <div className="list-group-item text-secondary small">{emptyText}</div>
      ) : (
        <>
          {installed.map((item) => (
            <div key={`installed:${item.kind}:${item.name}`} className="project-row session-row list-group-item py-3 px-4">
              <div className="session-icon"><i className={installedIcon}></i></div>
              <div className="project-main session-main">
                <strong>{item.name} <span className="badge text-bg-light text-uppercase">{item.kind}</span></strong>
                <span>{item.container_image}</span>
              </div>
              <div className="d-flex align-items-center gap-2">
                <span className={`status-pill ${item.disabled ? "status-failed" : "status-completed"}`}>{item.disabled ? "Disabled" : "Configured"}</span>
                <button className="btn btn-sm btn-outline-secondary" title={`Configure ${item.name}`} aria-label={`Configure ${item.name}`} onClick={() => onConfigure(item)}>
                  <i className="bi bi-gear"></i>
                </button>
                {onRemove && <button className="btn btn-sm btn-outline-danger" onClick={() => onRemove(item)}>Remove</button>}
              </div>
            </div>
          ))}
          {available.map((item) => (
            <div key={`${item.kind}:${item.name}`} className="project-row session-row list-group-item py-3 px-4">
              <div className="session-icon"><img src={item.icon_url} alt="" width="24" height="24" style={{ borderRadius: "6px" }} /></div>
              <div className="project-main session-main">
                <strong>{item.name} <span className="badge text-bg-light text-uppercase">{item.kind}</span></strong>
                <span>{item.description || item.container_image}</span>
              </div>
              <div className="d-flex align-items-center gap-2">
                <button className="btn btn-sm btn-outline-secondary" title={`Configure ${item.name}`} aria-label={`Configure ${item.name}`} onClick={() => onConfigure(item)}>
                  <i className="bi bi-gear"></i>
                </button>
                <button className="btn btn-sm btn-primary" onClick={() => onInstall(item)}>Install</button>
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
    for (const [key, property] of Object.entries(schema?.properties || {})) {
      if (pending.env?.[key] !== undefined) next[key] = pending.env[key];
      else if (property.default !== undefined) next[key] = String(property.default);
    }
    setValues(next);
    setTouched({});
  }, [schemaResponse?.container_image, pending.env]);

  const validationErrors = validateIntegrationConfig(schema, values);
  const fields = Object.entries(schema?.properties || {});
  const canSubmit = !loading && Object.keys(validationErrors).length === 0;

  return (
    <form className="card start-card p-3 mb-3 border-primary-subtle" onSubmit={(event) => {
      event.preventDefault();
      const allTouched = Object.fromEntries(fields.map(([key]) => [key, true]));
      setTouched(allTouched);
      if (canSubmit) onInstall(Object.fromEntries(Object.entries(values).filter(([, value]) => value.trim() !== "")));
    }}>
      <div className="d-flex align-items-start justify-content-between gap-3 mb-3">
        <div>
          <h3 className="h6 fw-bold mb-1">Configure {pending.name || pending.containerImage}</h3>
          <p className="text-secondary small mb-0">Values are stored as key/value config and passed as environment variables when the container starts.</p>
        </div>
        <span className="badge text-bg-light text-uppercase">{pending.kind}</span>
      </div>
      {loading && <div className="alert alert-info py-2 small"><span className="spinner-border spinner-border-sm me-2" />Fetching container schema annotation…</div>}
      {error && <div className="alert alert-warning py-2 small">Could not fetch the schema annotation, so this integration can be installed without extra fields. {error}</div>}
      {!loading && !error && fields.length === 0 && <div className="alert alert-secondary py-2 small">This container does not publish a configuration schema annotation, so it will be installed without extra environment values.</div>}
      {schema?.description && <p className="small text-secondary">{schema.description}</p>}
      <div className="row g-3">
        {fields.map(([key, property]) => {
          const label = property.title || key;
          const type = schemaPropertyType(property);
          const isInvalid = Boolean(touched[key] && validationErrors[key]);
          return (
            <div className="col-12 col-md-6" key={key}>
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
      <div className="d-flex justify-content-end gap-2 mt-3">
        <button type="button" className="btn btn-outline-secondary" onClick={onCancel}>Cancel</button>
        <button className="btn btn-primary" disabled={!canSubmit}>
          {pending.installed ? "Save configuration" : fields.length ? "Install with config" : "Install"}
        </button>
      </div>
    </form>
  );
}
