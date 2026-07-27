#!/usr/bin/env python3
"""Minimal pull-diagnostic LSP fixture for exact overlay/version tests."""

import json
import pathlib
import sys


LOG_PATH = pathlib.Path(sys.argv[1])
DOCUMENTS = {}


def read_message():
    content_length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, value = line.decode("ascii").split(":", 1)
        if name.lower() == "content-length":
            content_length = int(value.strip())
    if content_length is None:
        return None
    return json.loads(sys.stdin.buffer.read(content_length))


def send(message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def record(event, uri, version, text):
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"event": event, "uri": uri, "version": version, "text": text},
                separators=(",", ":"),
            )
            + "\n"
        )


while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    params = message.get("params") or {}
    request_id = message.get("id")
    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "capabilities": {
                        "textDocumentSync": 2,
                        "diagnosticProvider": {"interFileDependencies": True},
                    }
                },
            }
        )
    elif method == "textDocument/didOpen":
        document = params["textDocument"]
        DOCUMENTS[document["uri"]] = (document["version"], document["text"])
        record("open", document["uri"], document["version"], document["text"])
    elif method == "textDocument/didChange":
        document = params["textDocument"]
        text = params["contentChanges"][0]["text"]
        DOCUMENTS[document["uri"]] = (document["version"], text)
        record("change", document["uri"], document["version"], text)
    elif method == "textDocument/diagnostic":
        uri = params["textDocument"]["uri"]
        version, text = DOCUMENTS[uri]
        record("diagnostic", uri, version, text)
        items = []
        if "TYPE_ERROR" in text:
            items.append(
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 10},
                    },
                    "severity": 1,
                    "code": "E0308",
                    "source": "fake-lsp",
                    "message": "fixture semantic error",
                }
            )
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"kind": "full", "items": items},
            }
        )
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
    elif method == "exit":
        break
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
