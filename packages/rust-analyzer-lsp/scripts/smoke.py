#!/usr/bin/env python3
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = (ROOT / "fixtures" / "syntax-error").resolve()
IMAGE = os.environ.get("IMAGE", "pb/rust-analyzer-lsp:dev")
RUNTIME = os.environ.get("CONTAINER_RUNTIME", "docker")
NAME = f"pb-ra-smoke-{uuid.uuid4().hex[:12]}"
NETWORK = f"{NAME}-network"
MAX_FRAME = 1024 * 1024


def run_runtime(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [RUNTIME, *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def send(process: subprocess.Popen[bytes], message: dict) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode()
    process.stdin.write(f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
    process.stdin.flush()


def read_exact(stream, length: int, deadline: float) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out reading LSP frame")
        ready, _, _ = select.select([stream], [], [], remaining)
        if not ready:
            raise TimeoutError("timed out reading LSP frame")
        chunk = os.read(stream.fileno(), length - len(chunks))
        if not chunk:
            raise EOFError("language server closed stdout")
        chunks.extend(chunk)
    return bytes(chunks)


def receive(process: subprocess.Popen[bytes], deadline: float) -> dict:
    header = bytearray()
    while b"\r\n\r\n" not in header:
        if len(header) > 8192:
            raise RuntimeError("oversized LSP header")
        header.extend(read_exact(process.stdout, 1, deadline))
    content_length = None
    for line in header.decode().split("\r\n"):
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
    if content_length is None or content_length < 0 or content_length > MAX_FRAME:
        raise RuntimeError(f"invalid LSP content length: {content_length}")
    return json.loads(read_exact(process.stdout, content_length, deadline))


def answer_server_request(process: subprocess.Popen[bytes], message: dict) -> bool:
    if "id" not in message or "method" not in message:
        return False
    if message["method"] == "workspace/configuration":
        count = len(message.get("params", {}).get("items", []))
        result = [None] * count
    elif message["method"] == "workspace/workspaceFolders":
        result = [{"uri": "file:///workspace", "name": "syntax-error"}]
    else:
        result = None
    send(process, {"jsonrpc": "2.0", "id": message["id"], "result": result})
    return True


def drain_stderr(process: subprocess.Popen[bytes], output: bytearray) -> None:
    while True:
        chunk = process.stderr.read(4096)
        if not chunk:
            return
        output.extend(chunk)
        if len(output) > 64 * 1024:
            del output[: len(output) - 64 * 1024]


def main() -> None:
    run_runtime("network", "create", "--internal", NETWORK)
    args = [
        RUNTIME,
        "run",
        "-i",
        "--name",
        NAME,
        "--volume",
        f"{FIXTURE}:/workspace:ro",
        "--workdir",
        "/workspace",
        "--network",
        NETWORK,
        "--tmpfs",
        "/tmp",
        "--read-only",
        IMAGE,
    ]
    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        run_runtime("network", "rm", NETWORK, check=False)
        raise
    stderr = bytearray()
    stderr_thread = threading.Thread(target=drain_stderr, args=(process, stderr), daemon=True)
    stderr_thread.start()
    try:
        options = json.loads((ROOT / "pb-lsp.json").read_text())["server"]["initialization_options"]
        send(process, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": "file:///workspace",
                "workspaceFolders": [{"uri": "file:///workspace", "name": "syntax-error"}],
                "capabilities": {
                    "textDocument": {
                        "synchronization": {"didSave": True, "dynamicRegistration": False},
                        "publishDiagnostics": {},
                    },
                    "workspace": {"workspaceFolders": True, "configuration": True},
                },
                "initializationOptions": options,
            },
        })
        deadline = time.monotonic() + 30
        while True:
            message = receive(process, deadline)
            if message.get("id") == 1 and "result" in message:
                break
            answer_server_request(process, message)

        send(process, {"jsonrpc": "2.0", "method": "initialized", "params": {}})
        source = (FIXTURE / "src" / "lib.rs").read_text()
        send(process, {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///workspace/src/lib.rs",
                    "languageId": "rust",
                    "version": 1,
                    "text": source,
                }
            },
        })

        diagnostics = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            message = receive(process, deadline)
            if answer_server_request(process, message):
                continue
            if message.get("method") == "textDocument/publishDiagnostics":
                params = message.get("params", {})
                if params.get("uri") == "file:///workspace/src/lib.rs" and params.get("diagnostics"):
                    diagnostics = params["diagnostics"]
                    break
        if not diagnostics:
            raise RuntimeError("rust-analyzer did not publish the expected syntax diagnostic")

        send(process, {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": None})
        deadline = time.monotonic() + 10
        while True:
            message = receive(process, deadline)
            if message.get("id") == 2:
                break
            answer_server_request(process, message)
        send(process, {"jsonrpc": "2.0", "method": "exit", "params": None})
        process.stdin.close()
        if process.wait(timeout=10) != 0:
            raise RuntimeError("rust-analyzer exited unsuccessfully")
        print(f"LSP smoke passed with {len(diagnostics)} syntax diagnostic(s)")
    except Exception:
        process.kill()
        process.wait(timeout=5)
        sys.stderr.write(stderr.decode(errors="replace"))
        raise
    finally:
        run_runtime("rm", "-f", NAME, check=False)
        run_runtime("network", "rm", NETWORK, check=False)


if __name__ == "__main__":
    main()
