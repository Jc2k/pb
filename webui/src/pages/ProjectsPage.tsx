import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import type { InstalledIntegration, IntegrationConfigSchemaResponse, IntegrationKind, MarketplaceIntegration, PendingIntegrationInstall, ProjectEntry, SessionItem } from "../types";
import { IntegrationConfigForm, IntegrationList } from "../components/Integration";
import { PageShell } from "../components/PageShell";
import { SessionCard } from "../components/Session";
import { ensureNotificationPermission, uniqueInstalledIntegrations, uniqueIntegrations, useProjectFinishNotifications } from "../lib/helpers";

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

  const toggleProjectNotifications = async (project: ProjectEntry) => {
    if (!project.notify_on_finish && !(await ensureNotificationPermission())) return;
    const res = await fetch(`/api/projects/${encodeURIComponent(project.name)}/notifications`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notify_on_finish: !project.notify_on_finish }),
    });
    if (res.ok) void fetchProjects();
  };

  return (
    <PageShell>
      <section className="hero-section">
        <h1>Projects</h1>
        <p className="text-secondary mb-3">
          Choose a registered project to view its sessions and start focused project work.
        </p>
      </section>

      <section className="sessions-section">
        <div className="project-list session-list list-group">
          {projects.length === 0 ? (
            <div className="list-group-item text-secondary small">
              No registered projects. Add one with <code>pb projects add</code>.
            </div>
          ) : (
            projects.map((project) => {
              const projectSessions = sessions.filter((session) => session.workdir === project.path);
              const running = projectSessions.filter((session) => session.status === "running").length;
              return (
                <div
                  key={project.name}
                  className="project-row session-row list-group-item py-3 px-4"
                >
                  <div className="session-icon"><i className="bi bi-folder2-open"></i></div>
                  <Link
                    className="project-main session-main text-decoration-none text-reset"
                    to={`/projects/${encodeURIComponent(project.name)}`}
                  >
                    <strong>{project.name}</strong>
                    <span>{project.path}</span>
                  </Link>
                  <span className={`status-pill ${running ? "status-running" : "status-completed"}`}>
                    {projectSessions.length} session{projectSessions.length === 1 ? "" : "s"}
                  </span>
                  <button
                    type="button"
                    className={`btn btn-sm btn-icon ${project.notify_on_finish ? "btn-primary" : "btn-outline-secondary"}`}
                    title={project.notify_on_finish ? "Disable finish notifications" : "Notify me when sessions complete or fail"}
                    aria-label={project.notify_on_finish ? "Disable finish notifications" : "Enable finish notifications"}
                    onClick={(event) => {
                      event.preventDefault();
                      void toggleProjectNotifications(project);
                    }}
                  >
                    <i className={`bi ${project.notify_on_finish ? "bi-alarm-fill" : "bi-alarm"}`}></i>
                  </button>
                  <Link className="chevron text-decoration-none" to={`/projects/${encodeURIComponent(project.name)}`}>›</Link>
                </div>
              );
            })
          )}
        </div>
      </section>
    </PageShell>
  );
}

export function ProjectPage() {
  const { projectName: encodedProjectName } = useParams<{ projectName: string }>();
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectEntry[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [task, setTask] = useState("");
  const [branch, setBranch] = useState("main");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [marketplace, setMarketplace] = useState<MarketplaceIntegration[]>([]);
  const [installed, setInstalled] = useState<InstalledIntegration[]>([]);
  const [pendingInstall, setPendingInstall] = useState<PendingIntegrationInstall | null>(null);
  const [configSchema, setConfigSchema] = useState<IntegrationConfigSchemaResponse | null>(null);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaError, setSchemaError] = useState("");
  const name = encodedProjectName ? decodeURIComponent(encodedProjectName) : "";
  const project = projects.find((entry) => entry.name === name);
  const projectSessions = project ? sessions.filter((session) => session.workdir === project.path) : [];

  useProjectFinishNotifications(sessions, projects);

  useEffect(() => {
    void fetch("/api/projects")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: ProjectEntry[]) => setProjects(entries));
    void fetch("/api/integrations/marketplace")
      .then((res) => (res.ok ? res.json() : []))
      .then((entries: MarketplaceIntegration[]) => setMarketplace(uniqueIntegrations(entries.filter((entry) => entry.kind === "mcp"))));
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

  const fetchInstalledIntegrations = async () => {
    if (!name) return;
    const res = await fetch(`/api/projects/${encodeURIComponent(name)}/integrations`);
    if (res.ok) setInstalled(uniqueInstalledIntegrations(((await res.json()) as InstalledIntegration[]).filter((entry) => entry.kind === "mcp")));
  };

  useEffect(() => {
    void fetchInstalledIntegrations();
  }, [name]);

  const prepareIntegrationInstall = async (kind: IntegrationKind, containerImage: string, integrationName?: string, installed = false, env?: Record<string, string>) => {
    if (!project || !containerImage.trim()) return;
    const pending = { kind, containerImage: containerImage.trim(), name: integrationName, installed, env };
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
    if (!project || !window.confirm(`Remove ${item.name} from this project?`)) return;
    const res = await fetch(`/api/projects/${encodeURIComponent(project.name)}/integrations/${encodeURIComponent(item.name)}`, {
      method: "DELETE",
    });
    if (res.ok) void fetchInstalledIntegrations();
  };

  const installIntegration = async (env: Record<string, string> = {}) => {
    if (!project || !pendingInstall) return;
    const res = await fetch(`/api/projects/${encodeURIComponent(project.name)}/integrations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kind: pendingInstall.kind,
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

  const startProjectSession = async () => {
    if (!project || !task.trim()) return;
    setIsSubmitting(true);
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: task.trim(), workdir: project.path, branch: branch.trim() || "main" }),
      });
      if (!res.ok) return;
      const data = (await res.json()) as { session_id: string };
      navigate(`/sessions/${data.session_id}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <PageShell>
      <section className="hero-section">
        <Link to="/projects" className="text-decoration-none small fw-medium text-blue">← All projects</Link>
        <h1>{project?.name || name || "Project"}</h1>
        <p className="text-secondary mb-3">{project?.path || "Project not found"}</p>

        {project && (
          <form className="start-card card" onSubmit={(e) => { e.preventDefault(); void startProjectSession(); }}>
            <div className="task-editor position-relative">
              <textarea
                className="form-control"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                placeholder={`Ask the agent to work in ${project.name}…`}
                rows={4}
              />
            </div>
            <div className="session-controls row g-3 align-items-end p-3">
              <div className="col-12 col-md-8">
                <label className="form-label small fw-semibold">Base branch</label>
                <select className="form-select" value={branch} onChange={(e) => setBranch(e.target.value)}>
                  <option>main</option>
                  <option>develop</option>
                  <option>feature/ui-refresh</option>
                </select>
              </div>
              <div className="col-12 col-md-4 d-grid">
                <button className="btn btn-primary start-button" type="submit" disabled={!task.trim() || isSubmitting}>▷ Start project chat</button>
              </div>
            </div>
          </form>
        )}
      </section>

      {project && (
        <section className="sessions-section">
          <div className="section-header d-flex align-items-center justify-content-between mb-3">
            <div>
              <h2 className="h6 fw-bold m-0">Integrations</h2>
              <p className="text-secondary small m-0">Install project-scoped MCP containers for new sessions.</p>
            </div>
          </div>
          <IntegrationList
            marketplace={marketplace}
            installed={installed}
            installedIcon="bi bi-plug"
            emptyText="No marketplace integrations available to install."
            onInstall={(item) => void prepareIntegrationInstall(item.kind, item.container_image, item.name)}
            onConfigure={(item) => void prepareIntegrationInstall(item.kind, item.container_image, item.name, "disabled" in item, "disabled" in item ? item.env : undefined)}
            onRemove={(item) => void removeIntegration(item)}
          />
          {pendingInstall && (
            <IntegrationConfigForm
              pending={pendingInstall}
              schemaResponse={configSchema}
              loading={schemaLoading}
              error={schemaError}
              onCancel={() => { setPendingInstall(null); setConfigSchema(null); setSchemaError(""); }}
              onInstall={(env) => void installIntegration(env)}
            />
          )}
        </section>
      )}

      <section className="sessions-section">
        <div className="section-header d-flex align-items-center justify-content-between mb-3">
          <h2 className="h6 fw-bold m-0">Project sessions</h2>
        </div>
        <div className="session-list list-group">
          {projectSessions.length === 0 ? (
            <div className="list-group-item text-secondary small">No sessions for this project yet</div>
          ) : (
            projectSessions.map((session) => (
              <SessionCard key={session.session_id} session={session} onClick={() => navigate(`/sessions/${session.session_id}`)} />
            ))
          )}
        </div>
      </section>
    </PageShell>
  );
}
