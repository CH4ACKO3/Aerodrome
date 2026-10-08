"""Exercise the local HTTP service with the real JAX experiment runner."""
import json
from threading import Thread
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pytest

from aerodrome.teaching.experiment import MANIFEST
from aerodrome.teaching.server import TeachingServer


def test_local_experiment_submission_results_and_replay(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("local manual")
    (site / "teaching-manifest.json").write_text(json.dumps(MANIFEST))
    (tmp_path / "outside.txt").write_text("outside the published site")
    with TeachingServer(site, 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/Aerodrome/", timeout=5) as response:
                cookie = response.headers["Set-Cookie"].split(";", 1)[0]
                assert response.read() == b"local manual"

            def request(path, payload=None, **headers):
                body = None if payload is None else json.dumps(payload).encode()
                return urlopen(Request(base + path, data=body, headers={
                    "Cookie": cookie, "Origin": base, "Content-Type": "application/json", **headers,
                }), timeout=5)

            with request("/api/v1/hello") as response:
                manifest = json.load(response)
            payload = dict(manifest=manifest, experiment="rigid-velocity-v1", config={})
            # Exercise the actual HTTP boundary, not the server's validation helpers.
            with pytest.raises(HTTPError) as denied:
                request("/api/v1/jobs", payload, Cookie="")
            assert denied.value.code == 403
            with pytest.raises(HTTPError) as outside:
                request("/Aerodrome/%2e%2e/outside.txt")
            assert outside.value.code == 404
            with request("/api/v1/jobs", payload) as response:
                job_id = json.load(response)["job_id"]
            deadline = time.monotonic() + 30
            while True:
                with request("/api/v1/jobs/" + job_id) as response:
                    job = json.load(response)
                if job["status"] in ("complete", "failed"):
                    break
                if time.monotonic() >= deadline:
                    pytest.fail(f"experiment did not finish: {job}")
                time.sleep(.02)
            assert job["status"] == "complete", job
            result = job["result"]
            expected = [0.]
            for _ in range(250):
                expected.append(expected[-1] + .02*np.clip(2 - expected[-1], -1, 1))
            np.testing.assert_allclose(result["velocity_m_s"], expected, atol=3e-6)
            np.testing.assert_allclose(result["time_s"], np.arange(251)*.02)
            frames = result["render"]["frames"]
            assert len(frames) == 251 and frames[-1]["tick"] == 500
            np.testing.assert_allclose(frames[-1]["poses"][0]["position_ned_m"][0],
                                       .01*sum(a+b for a,b in zip(expected, expected[1:])), atol=3e-5)
        finally:
            server.shutdown()
            thread.join(timeout=5)
