import "./session.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { HomePage } from "./pages/HomePage";
import { IntegrationsPage } from "./pages/IntegrationsPage";
import {
  ProjectPage,
  ProjectSettingsPage,
  ProjectsPage,
} from "./pages/ProjectsPage";
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
        <Route
          path="/projects/:projectId/settings"
          element={<ProjectSettingsPage />}
        />
        <Route path="/projects/:projectId" element={<ProjectPage />} />
      </Routes>
    </BrowserRouter>
  );
}
