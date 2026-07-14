import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import { initialsForName } from "./lib/user";

export const primaryNavigation = [
  { to: "/", label: "Home", icon: "bi bi-house-door", end: true },
  {
    to: "/projects",
    label: "Projects",
    icon: "bi bi-folder2-open",
    end: false,
  },
  {
    to: "/integrations",
    label: "Integrations",
    icon: "bi bi-plug",
    end: false,
  },
] as const;

export function PrimaryNavigationLinks(
  { className = "nav-link" }: { className?: string },
) {
  return (
    <>
      {primaryNavigation.map((item) => (
        <NavLink
          key={item.to}
          className={({ isActive }) =>
            `${className}${isActive ? " active" : ""}`}
          to={item.to}
          end={item.end}
        >
          <i className={item.icon} aria-hidden="true"></i>
          <span>{item.label}</span>
        </NavLink>
      ))}
    </>
  );
}

interface CurrentUser {
  username: string;
  real_name?: string | null;
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
    <aside className="sidebar d-none d-xl-flex flex-column">
      <div className="brand d-flex align-items-center gap-2 px-3 py-3">
        <div className="brand-mark">&gt;_</div>
        <strong>LocalAgent</strong>
      </div>

      <nav
        className="nav nav-pills flex-column gap-1 px-2"
        aria-label="Primary navigation"
      >
        <PrimaryNavigationLinks />
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
