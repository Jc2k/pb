import { useEffect, useRef, useState } from "react";
import type {
  InstalledIntegration,
  IntegrationConfigSchemaResponse,
  IntegrationKind,
  MarketplaceIntegration,
  PendingIntegrationInstall,
} from "../types";
import {
  IntegrationConfigForm,
  IntegrationList,
} from "../components/Integration";
import { PageShell } from "../components/PageShell";
import {
  uniqueInstalledIntegrations,
  uniqueIntegrations,
} from "../lib/helpers";
import {
  integrationApiError,
  integrationInstallPayload,
} from "../lib/integrationConfig";

export function IntegrationsPage() {
  const [marketplace, setMarketplace] = useState<MarketplaceIntegration[]>([]);
  const [installed, setInstalled] = useState<InstalledIntegration[]>([]);
  const [pendingInstall, setPendingInstall] = useState<
    PendingIntegrationInstall | null
  >(null);
  const [configSchema, setConfigSchema] = useState<
    IntegrationConfigSchemaResponse | null
  >(null);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaError, setSchemaError] = useState("");
  const [submitError, setSubmitError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [pageError, setPageError] = useState("");
  const schemaRequest = useRef<{
    id: number;
    controller?: AbortController;
  }>({ id: 0 });

  const invalidateSchemaRequest = () => {
    schemaRequest.current.controller?.abort();
    schemaRequest.current = { id: schemaRequest.current.id + 1 };
  };

  useEffect(() => () => invalidateSchemaRequest(), []);

  useEffect(() => {
    void fetch("/api/integrations/marketplace")
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(
            await integrationApiError(
              res,
              "Could not load the integration marketplace",
            ),
          );
        }
        return res.json();
      })
      .then((entries: MarketplaceIntegration[]) =>
        setMarketplace(
          uniqueIntegrations(entries.filter((entry) => entry.kind === "lsp")),
        )
      )
      .catch((error) =>
        setPageError(
          error instanceof Error
            ? error.message
            : "Could not load the integration marketplace",
        )
      );
    void fetchInstalledIntegrations();
  }, []);

  const fetchInstalledIntegrations = async () => {
    try {
      const res = await fetch("/api/integrations/lsp");
      if (!res.ok) {
        throw new Error(
          await integrationApiError(
            res,
            "Could not load installed integrations",
          ),
        );
      }
      setInstalled(
        uniqueInstalledIntegrations(
          (await res.json()) as InstalledIntegration[],
        ),
      );
    } catch (error) {
      setPageError(
        error instanceof Error
          ? error.message
          : "Could not load installed integrations",
      );
    }
  };

  const prepareIntegrationInstall = async (
    containerImage: string,
    integrationName?: string,
    installed = false,
    env?: Record<string, string>,
    sourceContainerImage?: string,
    operation: "install" | "configure" | "upgrade" = installed
      ? "configure"
      : "install",
  ) => {
    if (!containerImage.trim()) return;
    const pending = {
      kind: "lsp" as IntegrationKind,
      containerImage: containerImage.trim(),
      sourceContainerImage,
      name: integrationName,
      installed,
      operation,
      env,
    };
    setPendingInstall(pending);
    setConfigSchema(null);
    setSchemaError("");
    setSubmitError("");
    setSchemaLoading(true);
    schemaRequest.current.controller?.abort();
    const requestId = schemaRequest.current.id + 1;
    const controller = new AbortController();
    schemaRequest.current = { id: requestId, controller };
    try {
      const res = await fetch(
        `/api/integrations/config-schema?image=${
          encodeURIComponent(pending.containerImage)
        }`,
        { signal: controller.signal },
      );
      if (!res.ok) {
        throw new Error(
          await integrationApiError(
            res,
            "Could not inspect the integration image",
          ),
        );
      }
      const metadata = (await res.json()) as IntegrationConfigSchemaResponse;
      if (
        schemaRequest.current.id === requestId && !controller.signal.aborted
      ) {
        setConfigSchema(metadata);
      }
    } catch (err) {
      if (
        schemaRequest.current.id === requestId && !controller.signal.aborted
      ) {
        setSchemaError(err instanceof Error ? err.message : "Unknown error");
      }
    } finally {
      if (schemaRequest.current.id === requestId) {
        schemaRequest.current = { id: requestId };
        setSchemaLoading(false);
      }
    }
  };

  const removeIntegration = async (item: InstalledIntegration) => {
    if (!window.confirm(`Remove ${item.name} from global configuration?`)) {
      return;
    }
    setPageError("");
    try {
      const res = await fetch(
        `/api/integrations/lsp/${encodeURIComponent(item.name)}`,
        {
          method: "DELETE",
        },
      );
      if (!res.ok) {
        throw new Error(
          await integrationApiError(res, "Could not remove the integration"),
        );
      }
      void fetchInstalledIntegrations();
    } catch (error) {
      setPageError(
        error instanceof Error
          ? error.message
          : "Could not remove the integration",
      );
    }
  };

  const installIntegration = async (env: Record<string, string> = {}) => {
    if (!pendingInstall) return;
    setSubmitting(true);
    setSubmitError("");
    try {
      const res = await fetch("/api/integrations/lsp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          integrationInstallPayload(pendingInstall, env, configSchema),
        ),
      });
      if (!res.ok) {
        throw new Error(
          await integrationApiError(res, "Could not install the integration"),
        );
      }
      invalidateSchemaRequest();
      setPendingInstall(null);
      setConfigSchema(null);
      setSchemaError("");
      void fetchInstalledIntegrations();
    } catch (error) {
      setSubmitError(
        error instanceof Error
          ? error.message
          : "Could not install the integration",
      );
    } finally {
      setSubmitting(false);
    }
  };

  const cancelIntegration = () => {
    invalidateSchemaRequest();
    setPendingInstall(null);
    setConfigSchema(null);
    setSchemaError("");
    setSubmitError("");
  };

  return (
    <PageShell>
      <section className="hero-section">
        <h1>Integrations</h1>
        <p className="text-secondary mb-3">
          Configure global language server containers available to all projects.
          MCP configuration remains project-scoped.
        </p>
      </section>

      <section className="sessions-section">
        <div className="section-header d-flex align-items-center justify-content-between mb-3">
          <div>
            <h2 className="h6 fw-bold m-0">Language servers</h2>
            <p className="text-secondary small m-0">
              Install global LSP containers for code intelligence in new
              sessions.
            </p>
          </div>
        </div>
        <IntegrationList
          marketplace={marketplace}
          installed={installed}
          installedIcon="bi bi-code-slash"
          emptyText="No marketplace language servers available to install."
          onInstall={(item) =>
            void prepareIntegrationInstall(item.container_image, item.name)}
          onConfigure={(item) =>
            void prepareIntegrationInstall(
              item.container_image,
              item.name,
              "disabled" in item,
              "disabled" in item ? item.env : undefined,
              "source_container_image" in item
                ? item.source_container_image
                : undefined,
            )}
          onUpgrade={(item) =>
            void prepareIntegrationInstall(
              item.source_container_image || item.container_image,
              item.name,
              true,
              item.env,
              item.source_container_image || item.container_image,
              "upgrade",
            )}
          onRemove={(item) => void removeIntegration(item)}
        />
        {pageError && (
          <div className="alert alert-danger mt-3 mb-0">{pageError}</div>
        )}
        {pendingInstall && (
          <IntegrationConfigForm
            pending={pendingInstall}
            schemaResponse={configSchema}
            loading={schemaLoading}
            error={schemaError}
            submitError={submitError}
            submitting={submitting}
            onCancel={cancelIntegration}
            onInstall={(env) => void installIntegration(env)}
          />
        )}
      </section>
    </PageShell>
  );
}
