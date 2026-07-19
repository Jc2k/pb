import "./session.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { HomePage } from "./pages/HomePage";
import { IntegrationsPage } from "./pages/IntegrationsPage";
import { ProjectsPage, ProjectPage, ProjectSettingsPage } from "./pages/ProjectsPage";
import { SessionPage } from "./pages/SessionPage";
import { SettingsPage } from "./pages/SettingsPage";
import { RouteReset } from "./components/RouteReset";

export default function App() {
  return (
    <BrowserRouter>
      <RouteReset />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/sessions/:sessionId" element={<SessionPage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/integrations" element={<IntegrationsPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/projects/:projectName/settings" element={<ProjectSettingsPage />} />
        <Route path="/projects/:projectName" element={<ProjectPage />} />
      </Routes>
    </BrowserRouter>
  );
}
