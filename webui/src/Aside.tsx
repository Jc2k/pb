import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

interface CurrentUser {
  username: string;
  real_name?: string | null;
}

function initialsForName(name: string): string {
  const parts = name
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  if (parts.length === 0) {
    return "?";
  }

  return parts
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");
}

export function Aside() {
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null);
  const [avatarFailed, setAvatarFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    fetch("/api/current-user", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`current user request failed: ${response.status}`);
        }
        return response.json() as Promise<CurrentUser>;
      })
      .then(setCurrentUser)
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        console.debug("Could not load current user info", error);
      });

    return () => controller.abort();
  }, []);

  const displayName =
    currentUser?.real_name || currentUser?.username || "Local user";
  const userSubtitle = currentUser?.username
    ? `@${currentUser.username}`
    : "Local workspace";
  const initials = useMemo(() => initialsForName(displayName), [displayName]);

  return (
    <aside className="sidebar d-none d-lg-flex flex-column">
      <div className="brand d-flex align-items-center gap-2 px-3 py-3">
        <div className="brand-mark">&gt;_</div>
        <strong>LocalAgent</strong>
      </div>

      <nav className="nav nav-pills flex-column gap-1 px-2">
        <Link className="nav-link active" to="/">
          <i className="bi bi-house-door"></i> Home
        </Link>
        <Link className="nav-link" to="/sessions">
          <i className="bi bi-chat-square-text"></i> Sessions
        </Link>
        <Link className="nav-link" to="/projects">
          <i className="bi bi-folder2-open"></i> Projects
        </Link>
        <Link className="nav-link" to="/integrations">
          <i className="bi bi-plug"></i> Integrations
        </Link>
        <Link className="nav-link" to="/settings">
          <i className="bi bi-gear"></i> Settings
        </Link>
      </nav>

      <div className="mt-auto user-menu p-3 d-flex align-items-center gap-2">
        <div className="avatar-sm" aria-hidden="true">
          {avatarFailed ? (
            <span>{initials}</span>
          ) : (
            <img
              src="/api/current-user.png"
              alt=""
              onError={() => setAvatarFailed(true)}
            />
          )}
        </div>
        <div>
          <strong>{displayName}</strong>
          <small className="d-block text-secondary">{userSubtitle}</small>
        </div>
      </div>
    </aside>
  );
}
