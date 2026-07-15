import type React from "react";
import { Link } from "react-router-dom";
import { Aside, PrimaryNavigationLinks } from "../Aside";

export function PageShell(
  { children, pageClassName = "", contentClassName = "" }: {
    children: React.ReactNode;
    pageClassName?: string;
    contentClassName?: string;
  },
) {
  return (
    <div className="app-shell">
      <Aside />
      <section className={`main-panel ${pageClassName}`.trim()}>
        <header className="mobile-topbar d-flex d-xl-none align-items-center justify-content-between">
          <Link
            className="brand compact text-decoration-none"
            to="/"
            aria-label="LocalAgent home"
          >
            <div className="brand-mark">&gt;_</div>
            <strong>LocalAgent</strong>
          </Link>
          <nav
            className="tablet-nav d-none d-md-flex"
            aria-label="Primary navigation"
          >
            <PrimaryNavigationLinks className="tablet-nav-link" />
          </nav>
        </header>
        <main className={`content-wrap workspace-wrap ${contentClassName}`.trim()}>
          {children}
        </main>
        <nav
          className="mobile-nav d-flex d-md-none"
          aria-label="Primary navigation"
        >
          <PrimaryNavigationLinks className="mobile-nav-link" />
        </nav>
      </section>
    </div>
  );
}
