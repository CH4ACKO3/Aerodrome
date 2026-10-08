"""One environment check exercises the full installed native dependency stack."""
import json
from pathlib import Path
import subprocess
import sys


def test_runtime_after_imports_and_native_execution(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/check_runtime.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["py_gil_disabled_build"] == 1
    assert not any(report["gil_enabled_by_stage"].values())
    assert report["devices"]
