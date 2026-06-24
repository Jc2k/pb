import type React from "react";
import { Aside } from "../Aside";

export function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="app-shell">
      <Aside />
      <section className="main-panel">
        <header className="mobile-topbar d-lg-none d-flex align-items-center justify-content-between px-3 py-2">
          <div className="brand compact d-flex align-items-center gap-2">
            <div className="brand-mark">&gt;_</div>
            <strong>LocalAgent</strong>
          </div>
        </header>
        <div className="content-wrap">{children}</div>
      </section>
    </div>
  );
}
