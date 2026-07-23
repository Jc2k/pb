#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / "pb-lsp.json").read_text())
schema = json.loads((ROOT / "config-schema.json").read_text())
containerfile = (ROOT / "Containerfile").read_text()

assert manifest["version"] == 1
assert manifest["kind"] == "lsp"
server = manifest["server"]
assert server["language_ids"] == ["rust"]
assert server["workspace_access"] == "read_only"
assert server["network_access"] == "none"
assert server["cache_ids"] == []
options = server["initialization_options"]
assert options["checkOnSave"] is False
assert options["cargo"]["buildScripts"]["enable"] is False
assert options["cargo"]["noDeps"] is True
assert options["procMacro"]["enable"] is False
assert schema["type"] == "object"
assert schema["properties"] == {}
assert "uk.unrtd.pb.integration.lsp-manifest" in containerfile
assert "uk.unrtd.pb.integration.config-schema" in containerfile
assert containerfile.count("@sha256:") == 2

print("package metadata is valid")
