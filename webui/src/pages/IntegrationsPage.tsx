import { useEffect, useState } from "react";
import type { InstalledIntegration, IntegrationConfigSchemaResponse, IntegrationKind, MarketplaceIntegration, PendingIntegrationInstall } from "../types";
import { IntegrationConfigForm, IntegrationList } from "../components/Integration";
import { PageShell } from "../components/PageShell";
import { uniqueInstalledIntegrations, uniqueIntegrations } from "../lib/helpers";

export function IntegrationsPage() {
  const [marketplace, setMarketplace] = useState<MarketplaceIntegration[]>([]);
  const [installed, setInstalled] = useState<InstalledIntegration[]>([]);
  const [pendingInstall, setPendingInstall] = useState<PendingIntegrationInstall | null>(null);
  const [configSchema, setConfigSchema] = useState<IntegrationConfigSchemaResponse | null>(null);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaError, setSchemaError] = useState("");

  useEffect(() => {
    void fetch("/api/integrations/marketplace")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: MarketplaceIntegration[]) => setMarketplace(uniqueIntegrations(entries.filter((entry) => entry.kind === "lsp"))));
    void fetchInstalledIntegrations();
  }, []);

  const fetchInstalledIntegrations = async () => {
    const res = await fetch("/api/integrations/lsp");
    if (res.ok) setInstalled(uniqueInstalledIntegrations((await res.json()) as InstalledIntegration[]));
  };

  const prepareIntegrationInstall = async (containerImage: string, integrationName?: string, installed = false, env?: Record<string, string>) => {
    if (!containerImage.trim()) return;
    const pending = { kind: "lsp" as IntegrationKind, containerImage: containerImage.trim(), name: integrationName, installed, env };
    setPendingInstall(pending);
    setConfigSchema(null);
    setSchemaError("");
    setSchemaLoading(true);
    try {
      const res = await fetch(`/api/integrations/config-schema?image=${encodeURIComponent(pending.containerImage)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setConfigSchema((await res.json()) as IntegrationConfigSchemaResponse);
    } catch (err) {
      setSchemaError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setSchemaLoading(false);
    }
  };

  const removeIntegration = async (item: InstalledIntegration) => {
    if (!window.confirm(`Remove ${item.name} from global configuration?`)) return;
    const res = await fetch(`/api/integrations/lsp/${encodeURIComponent(item.name)}`, {
      method: "DELETE",
    });
    if (res.ok) void fetchInstalledIntegrations();
  };

  const installIntegration = async (env: Record<string, string> = {}) => {
    if (!pendingInstall) return;
    const res = await fetch("/api/integrations/lsp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kind: "lsp",
        container_image: pendingInstall.containerImage,
        name: pendingInstall.name,
        runtime: "docker",
        env,
      }),
    });
    if (res.ok) {
      setPendingInstall(null);
      setConfigSchema(null);
      setSchemaError("");
      void fetchInstalledIntegrations();
    }
  };

  return (
    <PageShell>
      <section className="hero-section">
        <h1>Integrations</h1>
        <p className="text-secondary mb-3">
          Configure global language server containers available to all projects. MCP configuration remains project-scoped.
        </p>
      </section>

      <section className="sessions-section">
        <div className="section-header d-flex align-items-center justify-content-between mb-3">
          <div>
            <h2 className="h6 fw-bold m-0">Language servers</h2>
            <p className="text-secondary small m-0">Install global LSP containers for code intelligence in new sessions.</p>
          </div>
        </div>
        <IntegrationList
          marketplace={marketplace}
          installed={installed}
          installedIcon="bi bi-code-slash"
          emptyText="No marketplace language servers available to install."
          onInstall={(item) => void prepareIntegrationInstall(item.container_image, item.name)}
          onConfigure={(item) => void prepareIntegrationInstall(item.container_image, item.name, "disabled" in item, "disabled" in item ? item.env : undefined)}
          onRemove={(item) => void removeIntegration(item)}
        />
        {pendingInstall && <IntegrationConfigForm pending={pendingInstall} schemaResponse={configSchema} loading={schemaLoading} error={schemaError} onCancel={() => { setPendingInstall(null); setConfigSchema(null); setSchemaError(""); }} onInstall={(env) => void installIntegration(env)} />}
      </section>
    </PageShell>
  );
}
