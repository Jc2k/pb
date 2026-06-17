import { Link } from "react-router-dom";

export function Aside() {
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
        <Link className="nav-link" to="/settings">
          <i className="bi bi-gear"></i> Settings
        </Link>
      </nav>

      <div className="mt-auto user-menu p-3 d-flex align-items-center gap-2">
        <div className="avatar-sm">JD</div>
        <div>
          <strong>Jane Doe</strong>
          <small className="d-block text-secondary">Local workspace</small>
        </div>
      </div>
    </aside>
  );
}
