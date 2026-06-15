import { defineConfig } from "@rsbuild/core";
import { pluginReact } from "@rsbuild/plugin-react";

export default defineConfig({
  plugins: [pluginReact()],
  source: {
    entry: { index: "./webui/src/index.tsx" },
  },
  html: {
    template: "./webui/src/index.html",
  },
  output: {
    distPath: {
      root: "webui/dist",
    },
    assetPrefixes: {
      js: "/",
      css: "/",
      image: "/",
      font: "/",
      other: "/",
    },
  },
});
