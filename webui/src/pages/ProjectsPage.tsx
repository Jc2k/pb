import { useEffect, useMemo, useState } from "react";
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
import { GoalStartSheet } from "../components/GoalStartSheet";
import {
  integrationApiError,
  integrationInstallPayload,
} from "../lib/integrationConfig";
import {
  ensureNotificationPermission,
  projectName,
  projectSettingsPath,
  relativeTime,
  sessionTitle,
  uniqueInstalledIntegrations,
  uniqueIntegrations,
  usageStatsForToday,
} from "../lib/helpers";
import { useProjectFinishNotifications } from "../lib/hooks";

export function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);

  const fetchProjects = () =>
    fetch("/api/projects")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: ProjectEntry[]) => setProjects(entries));

  useEffect(() => {
    void fetchProjects();
    void fetch("/api/sessions")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: SessionItem[]) => setSessions(entries));
  }, []);

  useProjectFinishNotifications(sessions, projects);

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
          {projects.length === 0
            ? (
              <div className="list-group-item text-secondary small">
                No registered projects. Add one with{" "}
                <code>pb projects add</code>.
              </div>
            )
            : (
              projects.map((project) => {
                const projectSessions = sessions.filter((session) =>
                  session.workdir === project.path
                );
                const running = projectSessions.filter((session) =>
                  session.status === "running"
                ).length;
                return (
                  <div
                    key={project.name}
                    className="project-row session-row list-group-item py-3 px-4"
                  >
                    <div className="session-icon">
                      <i className="bi bi-folder2-open"></i>
                    </div>
                    <Link
                      className="project-main session-main text-decoration-none text-reset"
                      to={`/projects/${encodeURIComponent(project.name)}`}
                    >
                      <strong>{project.name}</strong>
                      <span>{project.path}</span>
                    </Link>
                    <span
                      className={`status-pill ${
                        running ? "status-running" : "status-completed"
                      }`}
                    >
                      {projectSessions.length}{" "}
                      session{projectSessions.length === 1 ? "" : "s"}
                    </span>
                    <Link
                      className="btn btn-sm btn-icon btn-outline-secondary"
                      to={projectSettingsPath(project.name)}
                      title={`Settings for ${project.name}`}
                      aria-label={`Settings for ${project.name}`}
                    >
                      <i className="bi bi-gear"></i>
                    </Link>
                    <Link
                      className="chevron text-decoration-none"
                      to={`/projects/${encodeURIComponent(project.name)}`}
                    >
                      ›
                    </Link>
                  </div>
                );
              })
            )}
        </div>
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
export function ProjectPage() {
  const { projectName: encodedProjectName } = useParams<
    { projectName: string }
  >();
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [task, setTask] = useState("");
  const [intent, setIntent] = useState<ComposerMode>("discuss");
  const [goalOpen, setGoalOpen] = useState(false);
  const [branch, setBranch] = useState("main");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [images, setImages] = useState<SessionAttachment[]>([]);
  const [filter, setFilter] = useState<SessionFilter>("all");
  const [activeDetailsTab, setActiveDetailsTab] = useState<ProjectDetailsTab>(
    "usage",
  );
  const [usage, setUsage] = useState<ProjectUsageStats>({
    tokens: 0,
    runtime_ms: 0,
    tool_calls: 0,
  });
  const name = encodedProjectName ? decodeURIComponent(encodedProjectName) : "";
  const project = projects.find((entry) => entry.name === name);
  const projectSessions = useMemo(
    () =>
      project
        ? sessions.filter((session) => session.workdir === project.path)
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
  const defaultBranch = branch.trim() || "main";

  useProjectFinishNotifications(sessions, projects);

  useEffect(() => {
    void fetch("/api/projects")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: ProjectEntry[]) => setProjects(entries));
  }, []);

  useEffect(() => {
    const fetchSessions = () =>
      fetch("/api/sessions")
        .then((res) => (res.ok ? res.json() : []))
        .then((entries: SessionItem[]) => setSessions(entries));
    void fetchSessions();
    const timer = window.setInterval(() => void fetchSessions(), 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!name) return;
    const fetchUsage = () =>
      fetch(`/api/projects/${encodeURIComponent(name)}/usage`)
        .then((
          res,
        ) => (res.ok
          ? res.json()
          : { tokens: 0, runtime_ms: 0, tool_calls: 0 })
        )
        .then((stats: ProjectUsageStats) => setUsage(stats));
    void fetchUsage();
    const timer = window.setInterval(() => void fetchUsage(), 5000);
    return () => window.clearInterval(timer);
  }, [name]);

  const startProjectSession = async () => {
    if (!project || !task.trim()) return;
    if (intent === "goal") {
      setGoalOpen(true);
      return;
    }
    setIsSubmitting(true);
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task: task.trim(),
          intent,
          workdir: project.path,
          branch: defaultBranch,
          attachments: images,
        }),
      });
      if (!res.ok) return;
      const data = (await res.json()) as { session_id: string };
      navigate(`/sessions/${data.session_id}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const usageList = (
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
              <h1>{project?.name || name || "Project"}</h1>
              {project && (
                <label className="branch-picker">
                  <i className="bi bi-git"></i>
                  <select
                    value={branch}
                    onChange={(e) => setBranch(e.target.value)}
                    aria-label="Base branch"
                  >
                    <option>main</option>
                    <option>develop</option>
                    <option>feature/ui-refresh</option>
                  </select>
                  <i className="bi bi-chevron-down"></i>
                </label>
              )}
            </div>
            {project && (
              <Link
                className="btn btn-light icon-btn settings-btn"
                to={projectSettingsPath(project.name)}
                aria-label="Project settings"
              >
                <i className="bi bi-gear"></i>
              </Link>
            )}
          </div>

          {project
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
                  />
                  <ImageAttachments images={images} setImages={setImages} />
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
                      <button
                        className="btn btn-primary send-btn"
                        type="submit"
                        disabled={!task.trim() || isSubmitting}
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

          <section className="sessions-section project-sessions-panel">
            <h2>Project sessions</h2>
            <SessionFilters
              filter={filter}
              counts={counts}
              onFilterChange={setFilter}
            />
            <SessionRows
              sessions={visibleSessions}
              defaultBranch={defaultBranch}
              emptyText="No sessions match this filter."
              onOpenSession={(session) =>
                navigate(`/sessions/${session.session_id}`)}
            />
          </section>
        </section>

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
                {(["usage", "overview", "snapshot"] as ProjectDetailsTab[]).map(
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
                  defaultBranch={defaultBranch}
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
                defaultBranch={defaultBranch}
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
      </div>
      <GoalStartSheet
        open={goalOpen}
        initialObjective={task}
        workdir={project?.path}
        onClose={() => setGoalOpen(false)}
        onStarted={(sessionId) => navigate(`/sessions/${sessionId}`)}
      />
    </PageShell>
  );
}

function ProjectOverview(
  { currentStatus, defaultBranch, lastActive, sessionCount }: {
    currentStatus: SessionItem["status"];
    defaultBranch: string;
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
        <span>Default branch</span>
        <strong>
          <i className="bi bi-git"></i> {defaultBranch}
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
  const { projectName: encodedProjectName } = useParams<
    { projectName: string }
  >();
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
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
  const [integrationError, setIntegrationError] = useState("");
  const [integrationSearch, setIntegrationSearch] = useState("");
  const [integrationCategory, setIntegrationCategory] = useState<
    IntegrationKind | "all"
  >("all");
  const name = encodedProjectName ? decodeURIComponent(encodedProjectName) : "";
  const project = projects.find((entry) => entry.name === name);
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

  const fetchProjects = () =>
    fetch("/api/projects")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: ProjectEntry[]) => setProjects(entries));

  const fetchInstalledIntegrations = async () => {
    if (!name) return;
    try {
      const res = await fetch(
        `/api/projects/${encodeURIComponent(name)}/integrations`,
      );
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
          ((await res.json()) as InstalledIntegration[]).filter((entry) =>
            entry.kind === "mcp"
          ),
        ),
      );
    } catch (error) {
      setIntegrationError(
        error instanceof Error
          ? error.message
          : "Could not load installed integrations",
      );
    }
  };

  useEffect(() => {
    void fetchProjects();
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
          uniqueIntegrations(entries.filter((entry) => entry.kind === "mcp")),
        )
      )
      .catch((error) =>
        setIntegrationError(
          error instanceof Error
            ? error.message
            : "Could not load the integration marketplace",
        )
      );
  }, []);

  useEffect(() => {
    void fetchInstalledIntegrations();
  }, [name]);

  const toggleProjectNotifications = async () => {
    if (!project) return;
    const notifyOnFinish = nextProjectNotificationPreference(project);
    if (notifyOnFinish) void ensureNotificationPermission();
    const res = await fetch(
      `/api/projects/${encodeURIComponent(project.name)}/notifications`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notify_on_finish: notifyOnFinish }),
      },
    );
    if (res.ok) void fetchProjects();
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
    try {
      const res = await fetch(
        `/api/integrations/config-schema?image=${
          encodeURIComponent(pending.containerImage)
        }`,
      );
      if (!res.ok) {
        throw new Error(
          await integrationApiError(
            res,
            "Could not inspect the integration image",
          ),
        );
      }
      setConfigSchema((await res.json()) as IntegrationConfigSchemaResponse);
    } catch (err) {
      setSchemaError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setSchemaLoading(false);
    }
  };

  const removeIntegration = async (item: InstalledIntegration) => {
    if (!project || !window.confirm(`Remove ${item.name} from this project?`)) {
      return;
    }
    setIntegrationError("");
    try {
      const res = await fetch(
        `/api/projects/${encodeURIComponent(project.name)}/integrations/${
          encodeURIComponent(item.name)
        }`,
        { method: "DELETE" },
      );
      if (!res.ok) {
        throw new Error(
          await integrationApiError(res, "Could not remove the integration"),
        );
      }
      void fetchInstalledIntegrations();
    } catch (error) {
      setIntegrationError(
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
    try {
      const res = await fetch(
        `/api/projects/${encodeURIComponent(project.name)}/integrations`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            integrationInstallPayload(pendingInstall, env, configSchema),
          ),
        },
      );
      if (!res.ok) {
        throw new Error(
          await integrationApiError(res, "Could not install the integration"),
        );
      }
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

  return (
    <PageShell>
      <section className="hero-section project-settings-hero">
        <Link
          to={`/projects/${encodeURIComponent(name)}`}
          className="back-link"
        >
          ← Project
        </Link>
        <h1>{project?.name || name || "Project"} settings</h1>
      </section>

      {project && (
        <div className="project-settings-stack">
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
                <p>Install and configure MCP servers scoped to this project.</p>
              </div>
            </div>
            <div className="mcp-store-toolbar">
              <label className="mcp-search-field">
                <i className="bi bi-search"></i>
                <input
                  value={integrationSearch}
                  onChange={(event) => setIntegrationSearch(event.target.value)}
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
                onCancel={() => {
                  setPendingInstall(null);
                  setConfigSchema(null);
                  setSchemaError("");
                  setSubmitError("");
                }}
                onInstall={(env) => void installIntegration(env)}
              />
            </div>
          )}
        </div>
      )}
    </PageShell>
  );
}
