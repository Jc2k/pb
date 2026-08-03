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
  parseInstalledIntegrationsJson,
  parseIntegrationConfigSchemaResponseJson,
  parseMarketplaceIntegrationsJson,
} from "../lib/integrationConfig";
import { isAbortError, LatestRequest } from "../lib/hooks";

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
  const marketplaceRequest = useRef(new LatestRequest());
  const installedRequest = useRef(new LatestRequest());
  const integrationMutationRequest = useRef(new LatestRequest());

  const invalidateSchemaRequest = () => {
    schemaRequest.current.controller?.abort();
    schemaRequest.current = { id: schemaRequest.current.id + 1 };
  };

  useEffect(() => () => {
    invalidateSchemaRequest();
    marketplaceRequest.current.abort();
    installedRequest.current.abort();
    integrationMutationRequest.current.abort();
  }, []);

  useEffect(() => {
    const controller = marketplaceRequest.current.start();
    void fetch("/api/integrations/marketplace", { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(
            await integrationApiError(
              res,
              "Could not load the integration marketplace",
            ),
          );
        }
        return parseMarketplaceIntegrationsJson(await res.text());
      })
      .then((entries) => {
        if (!marketplaceRequest.current.owns(controller)) return;
        setMarketplace(
          uniqueIntegrations(entries.filter((entry) => entry.kind === "lsp")),
        );
      })
      .catch((error) => {
        if (
          isAbortError(error) || !marketplaceRequest.current.owns(controller)
        ) return;
        setPageError(
          error instanceof Error
            ? error.message
            : "Could not load the integration marketplace",
        );
      });
    void fetchInstalledIntegrations();
    return () => {
      marketplaceRequest.current.abort();
      installedRequest.current.abort();
    };
  }, []);

  const fetchInstalledIntegrations = async () => {
    const controller = installedRequest.current.start();
    try {
      const res = await fetch("/api/integrations/lsp", {
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new Error(
          await integrationApiError(
            res,
            "Could not load installed integrations",
          ),
        );
      }
      const nextInstalled = uniqueInstalledIntegrations(
        parseInstalledIntegrationsJson(await res.text()),
      );
      if (!installedRequest.current.owns(controller)) return;
      setInstalled(nextInstalled);
    } catch (error) {
      if (isAbortError(error) || !installedRequest.current.owns(controller)) {
        return;
      }
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
    integrationMutationRequest.current.abort();
    setSubmitting(false);
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
      const metadata = parseIntegrationConfigSchemaResponseJson(
        await res.text(),
      );
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
    setSubmitting(false);
    installedRequest.current.abort();
    const controller = integrationMutationRequest.current.start();
    try {
      const res = await fetch(
        `/api/integrations/lsp/${encodeURIComponent(item.name)}`,
        {
          method: "DELETE",
          signal: controller.signal,
        },
      );
      if (!res.ok) {
        throw new Error(
          await integrationApiError(res, "Could not remove the integration"),
        );
      }
      const nextInstalled = uniqueInstalledIntegrations(
        parseInstalledIntegrationsJson(await res.text()),
      );
      if (!integrationMutationRequest.current.owns(controller)) return;
      setInstalled(nextInstalled);
    } catch (error) {
      if (
        isAbortError(error) ||
        !integrationMutationRequest.current.owns(controller)
      ) return;
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
    installedRequest.current.abort();
    const controller = integrationMutationRequest.current.start();
    try {
      const res = await fetch("/api/integrations/lsp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          integrationInstallPayload(pendingInstall, env, configSchema),
        ),
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new Error(
          await integrationApiError(res, "Could not install the integration"),
        );
      }
      const nextInstalled = uniqueInstalledIntegrations(
        parseInstalledIntegrationsJson(await res.text()),
      );
      if (!integrationMutationRequest.current.owns(controller)) return;
      setInstalled(nextInstalled);
      invalidateSchemaRequest();
      setPendingInstall(null);
      setConfigSchema(null);
      setSchemaError("");
    } catch (error) {
      if (
        isAbortError(error) ||
        !integrationMutationRequest.current.owns(controller)
      ) return;
      setSubmitError(
        error instanceof Error
          ? error.message
          : "Could not install the integration",
      );
    } finally {
      if (integrationMutationRequest.current.owns(controller)) {
        setSubmitting(false);
      }
    }
  };

  const cancelIntegration = () => {
    integrationMutationRequest.current.abort();
    invalidateSchemaRequest();
    setPendingInstall(null);
    setConfigSchema(null);
    setSchemaError("");
    setSubmitError("");
    setSubmitting(false);
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
