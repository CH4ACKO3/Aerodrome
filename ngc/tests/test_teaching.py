import json
from pathlib import Path
from threading import Thread
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pytest
from aerodrome.teaching.experiment import MANIFEST, VelocityConfig, run_velocity
from aerodrome.teaching.server import TeachingServer


def test_feedback_matches_sampled_equation():
    result = run_velocity({})
    expected = [0.]
    for _ in range(250):
        expected.append(expected[-1]+.02*np.clip(2-expected[-1], -1, 1))
    np.testing.assert_allclose(result["velocity_m_s"], expected, atol=3e-6)
    assert result["time_s"][-1] == 5
    assert len(result["force_N"]) == 250


@pytest.mark.parametrize("config", [{"gain_per_s":float("nan")},{"mass_kg":0},{"max_force_N":1e9},{"python":"print(1)"}])
def test_config_limits(config):
    with pytest.raises(ValueError):
        VelocityConfig.model_validate(config)


def test_http_contract(tmp_path):
    (tmp_path/"index.html").write_text("textbook")
    (tmp_path/"teaching-manifest.json").write_text(json.dumps(MANIFEST))
    with TeachingServer(tmp_path, 0, runner=lambda c: {"config":c}) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base+"/Aerodrome/") as response:
                cookie = response.headers["Set-Cookie"].split(";",1)[0]
                assert response.read() == b"textbook"
            def request(path, data=None, **headers):
                body = None if data is None else json.dumps(data).encode()
                return urlopen(Request(base+path, data=body, headers={"Cookie":cookie,"Origin":base,"Content-Type":"application/json",**headers}))
            with request("/api/v1/hello") as response:
                assert json.load(response) == MANIFEST
            for headers in ({"Cookie":""},{"Host":"evil.test"}):
                with pytest.raises(HTTPError) as error:
                    request("/api/v1/hello", **headers)
                assert error.value.code == 403
            payload = dict(manifest=MANIFEST, experiment="rigid-velocity-v1", config={})
            with pytest.raises(HTTPError) as error:
                request("/api/v1/jobs",payload,Origin="https://evil.test")
            assert error.value.code == 403
            with pytest.raises(HTTPError) as error:
                request("/api/v1/jobs",{**payload,"manifest":{}})
            assert error.value.code == 409
            with request("/api/v1/jobs", payload) as response:
                assert response.status == 202
                job_id = json.load(response)["job_id"]
            for _ in range(100):
                with request("/api/v1/jobs/"+job_id) as response:
                    job = json.load(response)
                if job["status"] == "complete":
                    break
                time.sleep(.01)
            assert job["result"]["config"]["mass_kg"] == 1000
            with pytest.raises(HTTPError) as error:
                request("/Aerodrome/%2e%2e/secret")
            assert error.value.code == 404
        finally:
            server.shutdown()
            thread.join()


def test_site_version_mismatch(tmp_path):
    (tmp_path/"index.html").write_text("textbook")
    (tmp_path/"teaching-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="version mismatch"):
        TeachingServer(tmp_path, 0)
