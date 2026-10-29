"""Small authenticated HTTP service around the serialized GPU engine."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from .config import MprfRuntimeConfiguration
from .contracts import decode_keyframe_packet, validated_session_id
from .engine import MprfOnlineEngine


class MprfServiceServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        *,
        engine: MprfOnlineEngine,
        configuration: MprfRuntimeConfiguration,
        auth_token: Optional[str],
    ) -> None:
        super().__init__(server_address, MprfRequestHandler)
        self.engine = engine
        self.configuration = configuration
        self.auth_token = auth_token
        self.inference_gate = threading.BoundedSemaphore(1)


class MprfRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "T3-MPRF/1"

    @property
    def mprf_server(self) -> MprfServiceServer:
        return self.server

    def log_message(self, format_string, *args) -> None:
        # Keep one service log line per HTTP transaction without dumping
        # images, tokens or request bodies.
        print(
            f"[mprf-http] {self.client_address[0]} "
            + format_string % args,
            flush=True,
        )

    def _authorized(self) -> bool:
        expected = self.mprf_server.auth_token
        if expected is None:
            return True
        supplied = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not supplied.startswith(prefix):
            return False
        return hmac.compare_digest(supplied[len(prefix):], expected)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client timeouts never cancel or corrupt the serialized inference.
            pass

    def _error(self, status: int, message: str) -> None:
        # Several failures are decided before a request body is consumed.
        # Closing this connection prevents unread NPZ bytes from being parsed
        # as the next HTTP request on a persistent socket.
        self.close_connection = True
        self._send_json(
            status,
            {
                "status": "error",
                "error": str(message),
            },
        )

    def _read_body(self, maximum_bytes: int) -> bytes:
        header = self.headers.get("Content-Length")
        if header is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(header)
        except ValueError as exception:
            raise ValueError("Content-Length is invalid") from exception
        if length <= 0 or length > int(maximum_bytes):
            raise ValueError(
                f"request body must be within 1..{int(maximum_bytes)} bytes"
            )
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("request body ended before Content-Length")
        return body

    def do_GET(self) -> None:
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "unauthorized")
            return
        path = urlsplit(self.path).path
        if path != "/health":
            self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        self._send_json(HTTPStatus.OK, self.mprf_server.engine.health())

    def do_POST(self) -> None:
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "unauthorized")
            return
        path = urlsplit(self.path).path
        if path == "/v1/keyframes":
            self._post_keyframe()
        elif path == "/v1/reset":
            self._post_reset()
        else:
            self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def _post_keyframe(self) -> None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
        if content_type != "application/x-npz":
            self._error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Content-Type must be application/x-npz",
            )
            return
        if not self.mprf_server.inference_gate.acquire(blocking=False):
            self._error(
                HTTPStatus.TOO_MANY_REQUESTS,
                "inference is busy; retry this keyframe later",
            )
            return
        try:
            body = self._read_body(
                self.mprf_server.configuration.maximum_request_bytes
            )
            packet = decode_keyframe_packet(
                body,
                maximum_points=(
                    self.mprf_server.configuration.maximum_request_points
                ),
            )
            response = self.mprf_server.engine.process(packet)
        except ValueError as exception:
            self._error(HTTPStatus.BAD_REQUEST, str(exception))
            return
        except Exception as exception:
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"{type(exception).__name__}: {exception}",
            )
            return
        finally:
            self.mprf_server.inference_gate.release()
        self._send_json(HTTPStatus.OK, response)

    def _post_reset(self) -> None:
        try:
            body = self._read_body(16 * 1024)
            request = json.loads(body.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("reset body must be a JSON object")
            session_id = validated_session_id(request.get("session_id", ""))
            response = self.mprf_server.engine.reset(session_id)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exception:
            self._error(HTTPStatus.BAD_REQUEST, str(exception))
            return
        self._send_json(HTTPStatus.OK, response)


def build_argument_parser() -> argparse.ArgumentParser:
    defaults = MprfRuntimeConfiguration()
    parser = argparse.ArgumentParser(
        description="Run the isolated online MPRF loop-candidate service"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument(
        "--mprf-source-root", type=Path, default=defaults.mprf_source_root
    )
    parser.add_argument("--model-root", type=Path, default=defaults.model_root)
    parser.add_argument("--state-root", type=Path, default=defaults.state_root)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--auth-token-env",
        default="MPRF_SERVICE_TOKEN",
        help="environment variable containing the optional bearer token",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="load every model, print health JSON and exit",
    )
    return parser


def main(argv=None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if not 1 <= int(arguments.port) <= 65535:
        raise SystemExit("--port must be in 1..65535")
    token = os.environ.get(str(arguments.auth_token_env), "").strip() or None
    host = str(arguments.host).strip()
    if host not in {"127.0.0.1", "localhost", "::1"} and token is None:
        raise SystemExit(
            "a non-loopback --host requires MPRF_SERVICE_TOKEN (or the "
            "variable selected by --auth-token-env)"
        )
    configuration = MprfRuntimeConfiguration(
        mprf_source_root=arguments.mprf_source_root,
        model_root=arguments.model_root,
        state_root=arguments.state_root,
        device=arguments.device,
        use_amp=not arguments.no_amp,
    )
    engine = MprfOnlineEngine(configuration)
    if arguments.verify_only:
        print(json.dumps(engine.health(), ensure_ascii=False, indent=2))
        return 0
    server = MprfServiceServer(
        (host, int(arguments.port)),
        engine=engine,
        configuration=configuration,
        auth_token=token,
    )
    print(
        f"[mprf] ready at http://{host}:{int(arguments.port)}; "
        f"device={configuration.device}; auth={'on' if token else 'loopback-only'}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
