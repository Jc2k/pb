import "./session.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { HomePage } from "./pages/HomePage";
import { IntegrationsPage } from "./pages/IntegrationsPage";
import { ProjectsPage, ProjectPage } from "./pages/ProjectsPage";
import { SessionPage } from "./pages/SessionPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/sessions/:sessionId" element={<SessionPage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/integrations" element={<IntegrationsPage />} />
        <Route path="/projects/:projectName" element={<ProjectPage />} />
      </Routes>
    </BrowserRouter>
  );
}
