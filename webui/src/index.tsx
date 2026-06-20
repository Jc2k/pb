import "./app.css";
import { createRoot } from "react-dom/client";
import App from "./App";

const root = document.getElementById("root")!;
createRoot(root).render(<App />);


if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    void navigator.serviceWorker.register("/pb-sw.js");
  });
}
