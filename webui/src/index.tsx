import "bootstrap/dist/css/bootstrap.min.css";
import "./app.css";
import { createRoot } from "react-dom/client";
import App from "./App";

const root = document.getElementById("root")!;
createRoot(root).render(<App />);
