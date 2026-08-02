import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import type {
  ComposerMode,
  InstalledIntegration,
  IntegrationConfigSchemaResponse,
  IntegrationKind,
  MarketplaceIntegration,
  PendingIntegrationInstall,
  ProjectEntry,
  ProjectUsageStats,
  SessionAttachment,
  SessionItem,
} from "../types";
import {
  IntegrationConfigForm,
  IntegrationList,
} from "../components/Integration";
import {
  AttachmentButton,
  ImageAttachments,
  sessionCounts,
  type SessionFilter,
  SessionFilters,
  SessionRows,
  UsageMetrics,
} from "../components/SessionDashboard";
import { PageShell } from "../components/PageShell";
import { IntentControl } from "../components/IntentControl";
import { VoiceInputButton } from "../components/VoiceInputButton";
import { GoalStartSheet } from "../components/GoalStartSheet";
import {
  apiErrorMessage,
  integrationApiError,
  integrationInstallPayload,
} from "../lib/integrationConfig";
import {
  ensureNotificationPermission,
  projectSettingsPath,
  relativeTime,
  sessionBelongsToProject,
  sessionTitle,
  uniqueInstalledIntegrations,
  uniqueIntegrations,
  usageStatsForToday,
} from "../lib/helpers";
import {
  isAbortError,
  LatestRequest,
  useProjectSessionData,
} from "../lib/hooks";

export function ProjectsPage() {
  const {
    projects,
    sessions,
    dataLoading,
    dataError,
    refresh,
  } = useProjectSessionData();

  return (
    <PageShell contentClassName="projects-index-wrap">
      <section className="hero-section">
        <h1>Projects</h1>
        <p className="text-secondary mb-3">
          Choose a registered project to view its sessions and start focused
          project work.
        </p>
      </section>

      <section className="sessions-section">
        <div className="project-list session-list list-group">
          {dataLoading && projects.length === 0
            ? (
              <div className="list-group-item text-secondary small">
                Loading projects…
              </div>
            )
            : dataError && projects.length === 0
            ? (
              <div className="list-group-item text-danger small">
                <span>{dataError}</span>{" "}
                <button
                  type="button"
                  className="btn btn-sm btn-link"
                  onClick={() => void refresh()}
                >
                  Try again
                </button>
              </div>
            )
            : projects.length === 0
            ? (
              <div className="list-group-item text-secondary small">
                No registered projects. Add one with{" "}
                <code>pb projects add</code>.
              </div>
            )
            : (
              projects.map((project) => {
                const projectSessions = sessions.filter((session) =>
                  sessionBelongsToProject(session, project)
                );
                const running = projectSessions.filter((session) =>
                  session.status === "running"
                ).length;
                return (
                  <div
                    key={project.id}
                    className="project-row session-row list-group-item py-3 px-4"
                  >
                    <div className="session-icon">
                      <i className="bi bi-folder2-open"></i>
                    </div>
                    <Link
                      className="project-main session-main text-decoration-none text-reset"
                      to={`/projects/${encodeURIComponent(project.id)}`}
                    >
                      <strong>{project.name}</strong>
                      <span>{project.path}</span>
                    </Link>
                    <span
                      className={`status-pill ${
                        running ? "status-running" : "status-completed"
                      }`}
                    >
                      {dataLoading
                        ? "Loading sessions…"
                        : `${projectSessions.length} session${
                          projectSessions.length === 1 ? "" : "s"
                        }`}
                    </span>
                    <Link
                      className="btn btn-sm btn-icon btn-outline-secondary"
                      to={projectSettingsPath(project.id)}
                      title={`Settings for ${project.name}`}
                      aria-label={`Settings for ${project.name}`}
                    >
                      <i className="bi bi-gear"></i>
                    </Link>
                    <span className="chevron" aria-hidden="true">
                      ›
                    </span>
                  </div>
                );
              })
            )}
        </div>
        {dataError && projects.length > 0 && (
          <div className="alert alert-warning mt-3" role="alert">
            Projects may be out of date: {dataError}{" "}
            <button
              type="button"
              className="btn btn-sm btn-link"
              onClick={() =>
                void refresh()}
            >
              Try again
            </button>
          </div>
        )}
      </section>
    </PageShell>
  );
}

type ProjectDetailsTab = "usage" | "overview" | "snapshot";

export function nextProjectNotificationPreference(
  project: Pick<ProjectEntry, "notify_on_finish">,
): boolean {
  return !project.notify_on_finish;
}

function projectMcpIntegrations(
  entries: InstalledIntegration[],
): InstalledIntegration[] {
  return uniqueInstalledIntegrations(
    entries.filter((entry) => entry.kind === "mcp"),
  );
}

export function ProjectPage() {
  const { projectId: encodedProjectId } = useParams<
    { projectId: string }
  >();
  const navigate = useNavigate();
  const {
    projects,
    sessions,
    projectUsage,
    dataLoading,
    dataError,
    refresh,
  } = useProjectSessionData();
  const [task, setTask] = useState("");
  const [intent, setIntent] = useState<ComposerMode>("discuss");
  const [goalOpen, setGoalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [voiceInputActive, setVoiceInputActive] = useState(false);
  const [images, setImages] = useState<SessionAttachment[]>([]);
  const [filter, setFilter] = useState<SessionFilter>("all");
  const [activeDetailsTab, setActiveDetailsTab] = useState<ProjectDetailsTab>(
    "usage",
  );
  const startRequest = useRef(new LatestRequest());
  const projectId = encodedProjectId
    ? decodeURIComponent(encodedProjectId)
    : "";
  const project = projects.find((entry) => entry.id === projectId);
  const usage: ProjectUsageStats = project
    ? projectUsage[project.id] || {
      tokens: 0,
      runtime_ms: 0,
      tool_calls: 0,
    }
    : { tokens: 0, runtime_ms: 0, tool_calls: 0 };
  const usageLoading = dataLoading && !project;
  const usageError = dataError;
  const projectSessions = useMemo(
    () =>
      project
        ? sessions.filter((session) =>
          sessionBelongsToProject(session, project)
        )
        : [],
    [project, sessions],
  );
  const visibleSessions = filter === "all"
    ? projectSessions
    : projectSessions.filter((session) => session.status === filter);
  const counts = useMemo(() => sessionCounts(projectSessions), [
    projectSessions,
  ]);
  const lastActive = projectSessions[0]?.updated_at_ms
    ? relativeTime(projectSessions[0].updated_at_ms)
    : "No activity yet";
  const todaysUsage = useMemo(() => usageStatsForToday(projectSessions), [
    projectSessions,
  ]);
  const latestBranch = projectSessions[0]?.branch || "Managed automatically";

  useEffect(() => {
    setTask("");
    setIntent("discuss");
    setGoalOpen(false);
    setIsSubmitting(false);
    setSubmitError("");
    setVoiceInputActive(false);
    setImages([]);
    setFilter("all");
    setActiveDetailsTab("usage");
    startRequest.current.abort();
    return () => startRequest.current.abort();
  }, [projectId]);

  const startProjectSession = async () => {
    if (!project || !task.trim()) return;
    if (intent === "goal") {
      setGoalOpen(true);
      return;
    }
    setIsSubmitting(true);
    setSubmitError("");
    const controller = startRequest.current.start();
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task: task.trim(),
          intent,
          project_id: project.id,
          attachments: images,
        }),
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Could not start the session"),
        );
      }
      const data = (await res.json()) as { session_id: string };
      if (!startRequest.current.owns(controller)) return;
      navigate(`/sessions/${data.session_id}`);
    } catch (error) {
      if (isAbortError(error) || !startRequest.current.owns(controller)) return;
      setSubmitError(
        error instanceof Error ? error.message : "Could not start the session",
      );
    } finally {
      if (startRequest.current.owns(controller)) setIsSubmitting(false);
    }
  };

  const usageList = usageLoading
    ? <p className="text-secondary small">Loading usage…</p>
    : usageError
    ? <p className="text-danger small">{usageError}</p>
    : (
      <UsageMetrics
        usage={usage}
        todaysUsage={todaysUsage}
        scopeLabel="Across project sessions"
      />
    );

  return (
    <PageShell
      pageClassName="project-detail-shell"
      contentClassName="project-detail-wrap"
    >
      <div className="project-layout">
        <section className="project-content">
          <Link to="/projects" className="back-link">
            <i className="bi bi-arrow-left"></i> All projects
          </Link>
          <div className="project-heading">
            <div className="title-row">
              <h1>{project?.name || "Project"}</h1>
            </div>
            {project && (
              <Link
                className="btn btn-light icon-btn settings-btn"
                to={projectSettingsPath(project.id)}
                aria-label="Project settings"
              >
                <i className="bi bi-gear"></i>
              </Link>
            )}
          </div>

          {project && dataError && (
            <div className="alert alert-warning" role="alert">
              Project details may be out of date: {dataError}{" "}
              <button
                type="button"
                className="btn btn-sm btn-link"
                onClick={() =>
                  void refresh()}
              >
                Try again
              </button>
            </div>
          )}

          {dataLoading && !project
            ? (
              <div className="card soft-card">
                <div className="card-body text-secondary">
                  Loading project…
                </div>
              </div>
            )
            : dataError && !project
            ? (
              <div className="card soft-card">
                <div className="card-body text-danger">
                  {dataError}{" "}
                  <button
                    type="button"
                    className="btn btn-sm btn-link"
                    onClick={() => void refresh()}
                  >
                    Try again
                  </button>
                </div>
              </div>
            )
            : project
            ? (
              <form
                className="card soft-card composer-card"
                onSubmit={(e) => {
                  e.preventDefault();
                  void startProjectSession();
                }}
              >
                <div className="card-body">
                  <h2>Ask the agent</h2>
                  <p>
                    What would you like the agent to work on in this project?
                  </p>
                  <textarea
                    className="form-control composer-input"
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    rows={3}
                    placeholder="Describe your task or ask a question..."
                    aria-label="Describe your task or ask a question"
                    readOnly={voiceInputActive}
                  />
                  <ImageAttachments images={images} setImages={setImages} />
                  {submitError && (
                    <p className="text-danger small" role="alert">
                      {submitError}
                    </p>
                  )}
                  <div className="composer-actions">
                    <div className="quick-actions">
                      <IntentControl
                        intent={intent}
                        onChange={setIntent}
                        disabled={isSubmitting}
                      />
                      <button
                        className="btn btn-light"
                        type="button"
                        onClick={() => {
                          setIntent("deliver");
                          setTask("fix bug");
                        }}
                      >
                        <i className="bi bi-bug"></i> Fix bug
                      </button>
                      <button
                        className="btn btn-light"
                        type="button"
                        onClick={() => {
                          setIntent("deliver");
                          setTask("add feature");
                        }}
                      >
                        <i className="bi bi-plus-lg"></i> Add feature
                      </button>
                      <button
                        className="btn btn-light"
                        type="button"
                        onClick={() => setTask("more options")}
                      >
                        <span>More</span>
                        <i className="bi bi-chevron-down"></i>
                      </button>
                    </div>
                    <div className="chat-submit-actions">
                      <AttachmentButton setImages={setImages} images={images} />
                      <VoiceInputButton
                        value={task}
                        onValueChange={setTask}
                        disabled={isSubmitting}
                        onActiveChange={setVoiceInputActive}
                      />
                      <button
                        className="btn btn-primary send-btn"
                        type="submit"
                        disabled={!task.trim() || isSubmitting ||
                          voiceInputActive}
                        aria-label="Start project chat"
                      >
                        <i className="bi bi-arrow-up"></i>
                      </button>
                    </div>
                  </div>
                </div>
              </form>
            )
            : (
              <div className="card soft-card">
                <div className="card-body text-secondary">
                  Project not found.
                </div>
              </div>
            )}

          {project && (
            <section className="sessions-section project-sessions-panel">
              <h2>Project sessions</h2>
              {dataLoading
                ? <p className="text-secondary small">Loading sessions…</p>
                : (
                  <>
                    <SessionFilters
                      filter={filter}
                      counts={counts}
                      onFilterChange={setFilter}
                    />
                    <SessionRows
                      sessions={visibleSessions}
                      emptyText="No sessions match this filter."
                      paginationKey={filter}
                      onOpenSession={(session) =>
                        navigate(`/sessions/${session.session_id}`)}
                    />
                  </>
                )}
            </section>
          )}
        </section>

        {project && (
          <aside className="project-aside">
            <div className="details-tabs card soft-card">
              <div className="card-body">
                <div className="details-heading">
                  <h2>Project details</h2>
                  <i className="bi bi-lock"></i>
                </div>
                <div
                  className="tab-nav"
                  role="tablist"
                  aria-label="Project details tabs"
                >
                  {(["usage", "overview", "snapshot"] as ProjectDetailsTab[])
                    .map(
                      (tab) => (
                        <button
                          key={tab}
                          type="button"
                          role="tab"
                          className={`nav-link${
                            activeDetailsTab === tab ? " active" : ""
                          }`}
                          onClick={() => setActiveDetailsTab(tab)}
                        >
                          {tab.charAt(0).toUpperCase() + tab.slice(1)}
                        </button>
                      ),
                    )}
                </div>
                {activeDetailsTab === "usage" && (
                  <div className="info-list usage-list">
                    {usageList}
                    <p className="privacy-note">
                      <i className="bi bi-lock"></i>{" "}
                      All usage is local and private.
                    </p>
                  </div>
                )}
                {activeDetailsTab === "overview" && (
                  <ProjectOverview
                    currentStatus={projectSessions[0]?.status || "queued"}
                    latestBranch={latestBranch}
                    lastActive={lastActive}
                    sessionCount={projectSessions.length}
                  />
                )}
                {activeDetailsTab === "snapshot" && (
                  <ProjectSnapshot
                    project={project}
                    lastSession={projectSessions[0]}
                  />
                )}
              </div>
            </div>
            <section className="card soft-card aside-card desktop-card">
              <div className="card-body">
                <div className="card-title-row">
                  <h2>Local usage</h2>
                  <i className="bi bi-info-circle"></i>
                </div>
                <div className="info-list usage-list">{usageList}</div>
                <p className="privacy-note">
                  <i className="bi bi-lock"></i> All usage is local and private.
                </p>
              </div>
            </section>
            <section className="card soft-card aside-card desktop-card">
              <div className="card-body">
                <h2>Project overview</h2>
                <ProjectOverview
                  currentStatus={projectSessions[0]?.status || "queued"}
                  latestBranch={latestBranch}
                  lastActive={lastActive}
                  sessionCount={projectSessions.length}
                />
              </div>
            </section>
            <section className="card soft-card aside-card desktop-card">
              <div className="card-body">
                <h2>Project snapshot</h2>
                <ProjectSnapshot
                  project={project}
                  lastSession={projectSessions[0]}
                />
              </div>
            </section>
            <p className="device-note">
              <i className="bi bi-lock"></i> All data stays on your device.
            </p>
          </aside>
        )}
      </div>
      <GoalStartSheet
        open={goalOpen}
        initialObjective={task}
        projectId={project?.id}
        onClose={() => setGoalOpen(false)}
        onStarted={(sessionId) => navigate(`/sessions/${sessionId}`)}
      />
    </PageShell>
  );
}

function ProjectOverview(
  { currentStatus, latestBranch, lastActive, sessionCount }: {
    currentStatus: SessionItem["status"];
    latestBranch: string;
    lastActive: string;
    sessionCount: number;
  },
) {
  return (
    <div className="info-list key-value-list">
      <div>
        <span>Current session</span>
        <strong>
          <span className={`state-pill ${currentStatus}`}>{currentStatus}</span>
        </strong>
      </div>
      <div>
        <span>Latest branch</span>
        <strong>
          <i className="bi bi-git"></i> {latestBranch}
        </strong>
      </div>
      <div>
        <span>Last active</span>
        <strong>{lastActive}</strong>
      </div>
      <div>
        <span>Sessions</span>
        <strong>{sessionCount}</strong>
      </div>
    </div>
  );
}

function ProjectSnapshot(
  { project, lastSession }: {
    project?: ProjectEntry;
    lastSession?: SessionItem;
  },
) {
  return (
    <div className="info-list snapshot-list">
      <div className="metric-row">
        <span className="metric-icon blue">
          <i className="bi bi-code-square"></i>
        </span>
        <span>
          <small>Project path</small>
          <strong>{project?.path || "Unknown"}</strong>
        </span>
      </div>
      <div className="metric-row">
        <span className="metric-icon purple">
          <i className="bi bi-box-seam"></i>
        </span>
        <span>
          <small>Workspace</small>
          <strong>{project?.name || "Project"}</strong>
        </span>
      </div>
      <div className="metric-row">
        <span className="metric-icon green">
          <i className="bi bi-git"></i>
        </span>
        <span>
          <small>Last session</small>
          <strong>
            {lastSession ? sessionTitle(lastSession) : "No sessions yet"}
          </strong>
          {lastSession && <em>{relativeTime(lastSession.updated_at_ms)}</em>}
        </span>
      </div>
    </div>
  );
}

export function ProjectSettingsPage() {
  const { projectId: encodedProjectId } = useParams<
    { projectId: string }
  >();
  const {
    projects,
    dataLoading: projectsLoading,
    dataError: projectsDataError,
    refresh: fetchProjects,
  } = useProjectSessionData({ finishNotifications: false });
  const [projectsMutationError, setProjectsMutationError] = useState("");
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
  const [marketplaceError, setMarketplaceError] = useState("");
  const [installedError, setInstalledError] = useState("");
  const [integrationMutationError, setIntegrationMutationError] = useState("");
  const [notificationMutationPending, setNotificationMutationPending] =
    useState(false);
  const notificationRequest = useRef(new LatestRequest());
  const integrationMutationRequest = useRef(new LatestRequest());
  const [integrationSearch, setIntegrationSearch] = useState("");
  const [integrationCategory, setIntegrationCategory] = useState<
    IntegrationKind | "all"
  >("all");
  const schemaRequest = useRef<{
    id: number;
    controller?: AbortController;
  }>({ id: 0 });
  const marketplaceRequest = useRef(new LatestRequest());
  const installedRequest = useRef(new LatestRequest());
  const invalidateSchemaRequest = () => {
    schemaRequest.current.controller?.abort();
    schemaRequest.current = { id: schemaRequest.current.id + 1 };
  };
  useEffect(() => () => {
    invalidateSchemaRequest();
    marketplaceRequest.current.abort();
    installedRequest.current.abort();
    notificationRequest.current.abort();
    integrationMutationRequest.current.abort();
  }, []);
  const projectId = encodedProjectId
    ? decodeURIComponent(encodedProjectId)
    : "";
  const project = projects.find((entry) => entry.id === projectId);
  const projectsError = projectsMutationError || projectsDataError;
  const integrationError = integrationMutationError || installedError ||
    marketplaceError;
  const filteredMarketplace = marketplace.filter((item) => {
    const query = integrationSearch.trim().toLowerCase();
    const matchesCategory = integrationCategory === "all" ||
      item.kind === integrationCategory;
    const matchesSearch = !query ||
      [item.name, item.description, item.container_image].some((value) =>
        value.toLowerCase().includes(query)
      );
    return matchesCategory && matchesSearch;
  });

  const fetchInstalledIntegrations = useCallback(async () => {
    if (!project) return;
    const controller = installedRequest.current.start();
    try {
      const res = await fetch(
        `/api/projects/${encodeURIComponent(project.id)}/integrations`,
        { signal: controller.signal },
      );
      if (!res.ok) {
        throw new Error(
          await integrationApiError(
            res,
            "Could not load installed integrations",
          ),
        );
      }
      const nextInstalled = projectMcpIntegrations(
        (await res.json()) as InstalledIntegration[],
      );
      if (!installedRequest.current.owns(controller)) return;
      setInstalled(nextInstalled);
      setInstalledError("");
    } catch (error) {
      if (isAbortError(error) || !installedRequest.current.owns(controller)) {
        return;
      }
      setInstalledError(
        error instanceof Error
          ? error.message
          : "Could not load installed integrations",
      );
    }
  }, [project?.id, project?.path]);

  useEffect(() => {
    const controller = marketplaceRequest.current.start();
    void fetch("/api/integrations/marketplace", {
      signal: controller.signal,
    })
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
      .then((entries: MarketplaceIntegration[]) => {
        if (!marketplaceRequest.current.owns(controller)) return;
        setMarketplace(
          uniqueIntegrations(entries.filter((entry) => entry.kind === "mcp")),
        );
        setMarketplaceError("");
      })
      .catch((error) => {
        if (
          isAbortError(error) || !marketplaceRequest.current.owns(controller)
        ) {
          return;
        }
        setMarketplaceError(
          error instanceof Error
            ? error.message
            : "Could not load the integration marketplace",
        );
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    invalidateSchemaRequest();
    setPendingInstall(null);
    setConfigSchema(null);
    setSchemaLoading(false);
    setSchemaError("");
    setSubmitError("");
    setSubmitting(false);
    setIntegrationMutationError("");
    setProjectsMutationError("");
    setNotificationMutationPending(false);
    notificationRequest.current.abort();
    integrationMutationRequest.current.abort();
  }, [projectId]);

  useEffect(() => {
    setInstalled([]);
    setInstalledError("");
    setIntegrationMutationError("");
    void fetchInstalledIntegrations();
    return () => installedRequest.current.abort();
  }, [fetchInstalledIntegrations]);

  const toggleProjectNotifications = async () => {
    if (!project || notificationMutationPending) return;
    const notifyOnFinish = nextProjectNotificationPreference(project);
    if (notifyOnFinish) void ensureNotificationPermission();
    setNotificationMutationPending(true);
    const controller = notificationRequest.current.start();
    try {
      const res = await fetch(
        `/api/projects/${encodeURIComponent(project.id)}/notifications`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ notify_on_finish: notifyOnFinish }),
          signal: controller.signal,
        },
      );
      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Could not update notifications"),
        );
      }
      await res.text();
      if (!notificationRequest.current.owns(controller)) return;
      setProjectsMutationError("");
    } catch (error) {
      if (
        isAbortError(error) || !notificationRequest.current.owns(controller)
      ) return;
      setProjectsMutationError(
        error instanceof Error
          ? error.message
          : "Could not update notifications",
      );
    } finally {
      if (notificationRequest.current.owns(controller)) {
        setNotificationMutationPending(false);
      }
    }
  };

  const prepareIntegrationInstall = async (
    kind: IntegrationKind,
    containerImage: string,
    integrationName?: string,
    installed = false,
    env?: Record<string, string>,
    sourceContainerImage?: string,
    operation: "install" | "configure" | "upgrade" = installed
      ? "configure"
      : "install",
  ) => {
    if (!project || !containerImage.trim()) return;
    integrationMutationRequest.current.abort();
    setSubmitting(false);
    const pending = {
      kind,
      containerImage: containerImage.trim(),
      name: integrationName,
      installed,
      sourceContainerImage,
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
    if (!project || !window.confirm(`Remove ${item.name} from this project?`)) {
      return;
    }
    setIntegrationMutationError("");
    setSubmitting(false);
    const controller = integrationMutationRequest.current.start();
    try {
      const res = await fetch(
        `/api/projects/${encodeURIComponent(project.id)}/integrations/${
          encodeURIComponent(item.name)
        }`,
        { method: "DELETE", signal: controller.signal },
      );
      if (!res.ok) {
        throw new Error(
          await integrationApiError(res, "Could not remove the integration"),
        );
      }
      const nextInstalled = projectMcpIntegrations(
        (await res.json()) as InstalledIntegration[],
      );
      if (!integrationMutationRequest.current.owns(controller)) return;
      setInstalled(nextInstalled);
      setInstalledError("");
    } catch (error) {
      if (
        isAbortError(error) ||
        !integrationMutationRequest.current.owns(controller)
      ) return;
      setIntegrationMutationError(
        error instanceof Error
          ? error.message
          : "Could not remove the integration",
      );
    }
  };

  const installIntegration = async (env: Record<string, string> = {}) => {
    if (!project || !pendingInstall) return;
    setSubmitting(true);
    setSubmitError("");
    const controller = integrationMutationRequest.current.start();
    try {
      const res = await fetch(
        `/api/projects/${encodeURIComponent(project.id)}/integrations`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            integrationInstallPayload(pendingInstall, env, configSchema),
          ),
          signal: controller.signal,
        },
      );
      if (!res.ok) {
        throw new Error(
          await integrationApiError(res, "Could not install the integration"),
        );
      }
      const nextInstalled = projectMcpIntegrations(
        (await res.json()) as InstalledIntegration[],
      );
      if (!integrationMutationRequest.current.owns(controller)) return;
      setInstalled(nextInstalled);
      setInstalledError("");
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
      <section className="hero-section project-settings-hero">
        <Link
          to={`/projects/${encodeURIComponent(projectId)}`}
          className="back-link"
        >
          ← Project
        </Link>
        <h1>{project?.name || "Project"} settings</h1>
      </section>

      {projectsLoading && !project
        ? <p className="text-secondary small">Loading project settings…</p>
        : projectsError && !project
        ? (
          <div className="alert alert-danger" role="alert">
            {projectsError}{" "}
            <button
              type="button"
              className="btn btn-sm btn-link"
              onClick={() => void fetchProjects()}
            >
              Try again
            </button>
          </div>
        )
        : !project
        ? <p className="text-secondary small">Project not found.</p>
        : (
          <div className="project-settings-stack">
            {projectsError && (
              <div className="alert alert-warning" role="alert">
                Project settings may be out of date: {projectsError}{" "}
                <button
                  type="button"
                  className="btn btn-sm btn-link"
                  onClick={() => void fetchProjects()}
                >
                  Try again
                </button>
              </div>
            )}
            <section
              className="settings-card notification-card"
              aria-labelledby="project-notifications-title"
            >
              <div>
                <h2 id="project-notifications-title">Notifications</h2>
                <p>
                  Choose whether this project sends browser notifications when
                  sessions complete or fail.
                </p>
              </div>
              <button
                type="button"
                className={`notification-switch ${
                  project.notify_on_finish ? "is-on" : ""
                }`}
                role="switch"
                aria-checked={project.notify_on_finish}
                disabled={notificationMutationPending}
                onClick={() => void toggleProjectNotifications()}
              >
                <span>{project.notify_on_finish ? "On" : "Off"}</span>
                <i></i>
              </button>
            </section>

            <section
              className="settings-card mcp-store-card"
              aria-labelledby="mcp-store-title"
            >
              <div className="mcp-store-header">
                <div>
                  <h2 id="mcp-store-title">MCP store</h2>
                  <p>
                    Install and configure MCP servers scoped to this project.
                  </p>
                </div>
              </div>
              <div className="mcp-store-toolbar">
                <label className="mcp-search-field">
                  <i className="bi bi-search"></i>
                  <input
                    value={integrationSearch}
                    onChange={(event) =>
                      setIntegrationSearch(event.target.value)}
                    placeholder="Search MCP servers..."
                    aria-label="Search MCP servers"
                  />
                </label>
                <select
                  className="mcp-category-select"
                  value={integrationCategory}
                  onChange={(event) =>
                    setIntegrationCategory(
                      event.target.value as IntegrationKind | "all",
                    )}
                  aria-label="Filter MCP servers by category"
                >
                  <option value="all">All categories</option>
                  <option value="mcp">MCP</option>
                </select>
              </div>
              <IntegrationList
                marketplace={filteredMarketplace}
                installed={integrationCategory === "all" ||
                    integrationCategory === "mcp"
                  ? installed
                  : []}
                installedIcon="bi bi-plug"
                emptyText="No MCP servers match your filters."
                onInstall={(item) =>
                  void prepareIntegrationInstall(
                    item.kind,
                    item.container_image,
                    item.name,
                  )}
                onConfigure={(item) =>
                  void prepareIntegrationInstall(
                    item.kind,
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
                    item.kind,
                    item.source_container_image || item.container_image,
                    item.name,
                    true,
                    item.env,
                    item.source_container_image || item.container_image,
                    "upgrade",
                  )}
                onRemove={(item) => void removeIntegration(item)}
              />
              {integrationError && (
                <div className="alert alert-danger mt-3 mb-0">
                  {integrationError}
                </div>
              )}
            </section>
            {pendingInstall && (
              <div className="integration-modal-backdrop">
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
              </div>
            )}
          </div>
        )}
    </PageShell>
  );
}
