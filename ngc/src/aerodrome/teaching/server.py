"""Serve the identical static textbook and a bounded, same-origin local API."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
import json
import mimetypes
from pathlib import Path
import secrets
from threading import Lock
from urllib.parse import unquote, urlsplit
from .experiment import MANIFEST, VelocityConfig, run_velocity


class TeachingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, site, port=8765, runner=run_velocity):
        self.site = Path(site).resolve()
        if not (self.site / "index.html").is_file():
            raise ValueError("build website first: site must contain index.html")
        manifest = json.loads((self.site / "teaching-manifest.json").read_text())
        if manifest != MANIFEST:
            raise ValueError("textbook/core version mismatch; rebuild matching website")
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.lock = Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="teaching")
        self.runner = runner
        super().__init__(("127.0.0.1", port), Handler)

    def server_close(self):
        super().server_close()
        self.pool.shutdown(wait=True, cancel_futures=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, code, body, content_type="application/json", cookie=False):
        if not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if cookie:
            self.send_header("Set-Cookie", f"aerodrome={self.server.token}; HttpOnly; SameSite=Strict; Path=/api/v1")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self, api=False, write=False):
        host = self.headers.get("Host", "")
        if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
            self.respond(403, {"error": "invalid host"})
            return False
        if api:
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
                valid = secrets.compare_digest(cookie["aerodrome"].value, self.server.token)
            except (KeyError, ValueError):
                valid = False
            if not valid or (write and self.headers.get("Origin") != "http://"+host):
                self.respond(403, {"error": "open the local textbook before using its API"})
                return False
        return True

    def do_GET(self):
        route = unquote(urlsplit(self.path).path)
        if not self.authorized(api=route.startswith("/api/")):
            return
        if route == "/api/v1/hello":
            return self.respond(200, MANIFEST)
        if route.startswith("/api/v1/jobs/"):
            with self.server.lock:
                job = self.server.jobs.get(route.rsplit("/", 1)[1])
            if job is None:
                return self.respond(404, {"error": "unknown job"})
            if not job.done():
                return self.respond(200, {"status": "running" if job.running() else "queued"})
            try:
                return self.respond(200, {"status": "complete", "result": job.result()})
            except Exception:
                return self.respond(200, {"status": "failed", "error": "simulation failed; inspect local runtime"})
        if route.startswith("/api/"):
            return self.respond(404, {"error": "unknown endpoint"})
        if route == "/":
            self.send_response(302)
            self.send_header("Location", "/Aerodrome/")
            self.end_headers()
            return
        if not route.startswith("/Aerodrome/"):
            return self.respond(404, {"error": "not found"})
        file = (self.server.site / route[len("/Aerodrome/"):]).resolve()
        if file.is_dir():
            file = file / "index.html"
        if not file.is_relative_to(self.server.site) or not file.is_file():
            return self.respond(404, {"error": "not found"})
        content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        self.respond(200, file.read_bytes(), content_type, cookie=content_type == "text/html")

    def do_POST(self):
        if not self.authorized(api=True, write=True):
            return
        if self.path != "/api/v1/jobs":
            return self.respond(404, {"error": "unknown endpoint"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384 or self.headers.get("Content-Type") != "application/json":
                raise ValueError("JSON body required, maximum 16 KiB")
            payload = json.loads(self.rfile.read(length))
            if payload.get("manifest") != MANIFEST:
                return self.respond(409, {"error": "version mismatch"})
            if payload.get("experiment") != "rigid-velocity-v1":
                raise ValueError("unknown experiment")
            config = VelocityConfig.model_validate(payload["config"]).model_dump()
        except (ValueError, KeyError, TypeError, AttributeError):
            return self.respond(400, {"error": "invalid experiment configuration"})
        with self.server.lock:
            if sum(not job.done() for job in self.server.jobs.values()) >= 2:
                return self.respond(429, {"error": "local queue is full"})
            if len(self.server.jobs) >= 32:
                oldest = next(key for key, job in self.server.jobs.items() if job.done())
                del self.server.jobs[oldest]
            job_id = secrets.token_hex(16)
            self.server.jobs[job_id] = self.server.pool.submit(self.server.runner, config)
        self.respond(202, {"job_id": job_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with TeachingServer(args.site, args.port) as server:
        print(f"Probabilistic Aviation: http://127.0.0.1:{server.server_port}/Aerodrome/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
