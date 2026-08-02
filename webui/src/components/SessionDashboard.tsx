import { useState } from "react";
import type {
  ProjectUsageStats,
  SessionAttachment,
  SessionItem,
} from "../types";
import { projectName, relativeTime, sessionTitle } from "../lib/helpers";
import { formatEnergy } from "../lib/energy";

export type SessionFilter = "all" | SessionItem["status"];

export const SESSION_ROW_BATCH_SIZE = 6;

export function sessionCounts(
  sessions: SessionItem[],
): Record<SessionFilter, number> {
  return {
    all: sessions.length,
    running: sessions.filter((session) => session.status === "running").length,
    queued: sessions.filter((session) => session.status === "queued").length,
    paused: sessions.filter((session) => session.status === "paused").length,
    completed:
      sessions.filter((session) => session.status === "completed").length,
    failed: sessions.filter((session) => session.status === "failed").length,
  };
}

export function formatUsageValue(value: number, suffix = ""): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m${suffix}`;
  if (value >= 1_000) return `${Math.round(value / 1_000)}k${suffix}`;
  return `${Math.round(value)}${suffix}`;
}

export function formatRuntime(ms: number): string {
  const hours = Math.floor(ms / 3_600_000);
  const minutes = Math.round((ms % 3_600_000) / 60_000);
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m`;
  return `${Math.round(ms / 1000)}s`;
}

export function statusIcon(status: SessionItem["status"]): string {
  if (status === "running") return "bi bi-record-circle";
  if (status === "queued") return "bi bi-clock";
  if (status === "paused") return "bi bi-pause-fill";
  if (status === "failed") return "bi bi-x-circle";
  return "bi bi-check-lg";
}

export function statusLabel(session: SessionItem): string {
  if (session.status === "paused" && session.pending_question) {
    return "Needs answer";
  }
  if (session.status === "paused") return "Paused";
  return session.status.charAt(0).toUpperCase() + session.status.slice(1);
}

export function iconTone(status: SessionItem["status"], index: number): string {
  if (status === "running") return "green";
  if (status === "queued") return "blue";
  if (status === "paused") return "amber";
  if (status === "failed") return "red";
  return ["black", "green", "purple"][index % 3];
}

export function UsageMetrics(
  { usage, todaysUsage, scopeLabel }: {
    usage: ProjectUsageStats;
    todaysUsage: ProjectUsageStats;
    scopeLabel: string;
  },
) {
  return (
    <>
      <div className="metric-row">
        <span className="metric-icon purple">
          <i className="bi bi-file-earmark-text"></i>
        </span>
        <span>
          <small>Tokens</small>
          <strong>{formatUsageValue(usage.tokens)}</strong>
          <em className="today-usage">
            Today: {formatUsageValue(todaysUsage.tokens)}
          </em>
          <em>{scopeLabel}</em>
        </span>
      </div>
      <div className="metric-row">
        <span className="metric-icon green">
          <i className="bi bi-lightning-charge"></i>
        </span>
        <span>
          <small>Estimated task energy</small>
          <strong>
            {formatEnergy(
              usage.energy_joules ?? (usage.energy_kwh == null
                ? undefined
                : usage.energy_kwh * 3_600_000),
            )}
          </strong>
          <em className="today-usage">
            Today: {formatEnergy(
              todaysUsage.energy_joules ??
                (todaysUsage.energy_kwh == null
                  ? undefined
                  : todaysUsage.energy_kwh * 3_600_000),
            )}
          </em>
          <em>
            Whole-device estimate; display and idle baseline excluded when
            measured
          </em>
        </span>
      </div>
      <div className="metric-row">
        <span className="metric-icon blue">
          <i className="bi bi-clock"></i>
        </span>
        <span>
          <small>Runtime</small>
          <strong>{formatRuntime(usage.runtime_ms)}</strong>
          <em className="today-usage">
            Today: {formatRuntime(todaysUsage.runtime_ms)}
          </em>
          <em>Task wall time</em>
        </span>
      </div>
      <div className="metric-row">
        <span className="metric-icon orange">
          <i className="bi bi-bezier2"></i>
        </span>
        <span>
          <small>Tool calls</small>
          <strong>{usage.tool_calls}</strong>
          <em className="today-usage">Today: {todaysUsage.tool_calls}</em>
          <em>{scopeLabel}</em>
        </span>
      </div>
    </>
  );
}

export function SessionFilters(
  { filter, counts, onFilterChange }: {
    filter: SessionFilter;
    counts: Record<SessionFilter, number>;
    onFilterChange: (filter: SessionFilter) => void;
  },
) {
  return (
    <div
      className="status-filter"
      role="group"
      aria-label="Filter sessions by state"
    >
      {([
        "all",
        "running",
        "queued",
        "paused",
        "completed",
        "failed",
      ] as SessionFilter[]).map((item) => (
        <button
          key={item}
          type="button"
          className={`filter-chip${filter === item ? " active" : ""}`}
          onClick={() => onFilterChange(item)}
        >
          {item !== "all" && (
            <i
              className={`${
                statusIcon(item as SessionItem["status"])
              } status-${item}`}
            >
            </i>
          )} {item.charAt(0).toUpperCase() + item.slice(1)}{" "}
          <span>{counts[item]}</span>
        </button>
      ))}
    </div>
  );
}

export function SessionRows({
  sessions,
  emptyText,
  onOpenSession,
  paginationKey = "default",
}: {
  sessions: SessionItem[];
  emptyText: string;
  onOpenSession: (session: SessionItem) => void;
  paginationKey?: string;
}) {
  const [pagination, setPagination] = useState({
    key: paginationKey,
    count: SESSION_ROW_BATCH_SIZE,
  });
  const visibleCount = pagination.key === paginationKey
    ? pagination.count
    : SESSION_ROW_BATCH_SIZE;
  const visibleSessions = sessions.slice(0, visibleCount);
  const remaining = Math.max(0, sessions.length - visibleSessions.length);

  return (
    <div className="session-list card soft-card">
      {sessions.length === 0
        ? <div className="empty-session-row">{emptyText}</div>
        : (
          <>
            {visibleSessions.map((session, index) => (
              <button
                key={session.session_id}
                className="session-row project-session-row"
                type="button"
                onClick={() => onOpenSession(session)}
              >
                <span
                  className={`session-icon ${iconTone(session.status, index)}`}
                >
                  <i className={statusIcon(session.status)}></i>
                </span>
                <span className="session-main">
                  <strong>{sessionTitle(session)}</strong>
                  <small>
                    {projectName(session.workdir)} <i className="bi bi-git"></i>
                    {" "}
                    {session.branch || "Managed workspace"}
                  </small>
                </span>
                <span className={`state-pill ${session.status}`}>
                  <i className={statusIcon(session.status)}></i>{" "}
                  {statusLabel(session)}
                </span>
                <time>{relativeTime(session.updated_at_ms)}</time>
                <i className="bi bi-chevron-right row-chevron"></i>
              </button>
            ))}
            {remaining > 0
              ? (
                <button
                  className="session-list-more"
                  type="button"
                  onClick={() =>
                    setPagination({
                      key: paginationKey,
                      count: visibleCount + SESSION_ROW_BATCH_SIZE,
                    })}
                >
                  <span>
                    Show {Math.min(SESSION_ROW_BATCH_SIZE, remaining)}{" "}
                    more sessions
                  </span>
                  <small>{remaining} remaining</small>
                </button>
              )
              : null}
          </>
        )}
    </div>
  );
}

export function AttachmentButton(
  { images, setImages }: {
    images: SessionAttachment[];
    setImages: (images: SessionAttachment[]) => void;
  },
) {
  const onFiles = async (files: FileList | null) => {
    if (!files) return;
    const loaded = await Promise.all(
      Array.from(files).filter((file) => file.type.startsWith("image/")).map(
        async (file) => ({
          name: file.name,
          mime: file.type || "application/octet-stream",
          base64: await fileToBase64(file),
        }),
      ),
    );
    setImages([...images, ...loaded]);
  };
  return (
    <label
      className="btn btn-light attach-btn"
      aria-label="Attach images"
      title="Attach images"
    >
      <i className="bi bi-paperclip" aria-hidden="true"></i>
      <input
        className="visually-hidden"
        type="file"
        accept="image/*"
        multiple
        onChange={(e) => void onFiles(e.target.files)}
      />
    </label>
  );
}

export function ImageAttachments(
  { images, setImages }: {
    images: SessionAttachment[];
    setImages: (images: SessionAttachment[]) => void;
  },
) {
  if (images.length === 0) return null;
  return (
    <div className="attachment-row small text-secondary mt-2">
      {images.map((image, index) => (
        <span
          key={`${image.name}-${index}`}
          className="badge text-bg-light ms-2"
        >
          {image.name}
          <button
            type="button"
            className="btn-close btn-close-sm ms-2"
            aria-label={`Remove ${image.name}`}
            onClick={() => setImages(images.filter((_, i) => i !== index))}
          >
          </button>
        </span>
      ))}
    </div>
  );
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve(String(reader.result || "").split(",")[1] || "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}
