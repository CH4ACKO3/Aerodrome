"""Run the optional Hydra frontend against the real Python 3.14 argparse."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

pytest.importorskip("hydra")


def cli(tmp_path,*args):
    env = dict(os.environ,HYDRA_FULL_ERROR="1",PYTHONPATH=str(Path(__file__).parents[2]/"src"))
    return subprocess.run([sys.executable,"-m","aerodrome.configuration.hydra_cli",*args],
                          cwd=tmp_path,env=env,text=True,capture_output=True,timeout=120)


def test_multirun_chdir_and_provenance(tmp_path):
    result = cli(tmp_path,"-m","seed=1,2","runtime.steps=2","renderer=headless",
                 "renderer.parameters.every_steps=1","hydra.job.chdir=true")
    assert result.returncode==0,result.stdout+result.stderr
    runs = list((tmp_path/"artifacts/config_runs").iterdir())
    assert len(runs)==2
    assert {json.loads((p/"resolved.json").read_text())["seed"] for p in runs}=={1,2}
    for p in runs:
        status = json.loads((p/"status.json").read_text())
        assert status["status"]=="complete" and status["rendered_frames"]==2
        context = json.loads((p/"hydra-context.json").read_text())
        assert context["choices"]["renderer"]=="headless"
        assert json.loads((Path(context["output_dir"])/"aerodrome-run.json").read_text())["run_dir"]==str(p)


def test_external_config_interpolation_and_relative_assets(tmp_path):
    from hashlib import sha256
    payload = b"test asset"
    (tmp_path/"table.bin").write_bytes(payload)
    (tmp_path/"custom.yaml").write_text(
        "defaults:\n  - aerodrome_experiment\n  - _self_\n"
        "seed: 7\nname: experiment_${seed}\nruntime: {steps: 1}\n"
        "assets:\n  table:\n    path: table.bin\n    sha256: '"+sha256(payload).hexdigest()+"'\n"
        "    source: test\n    license: CC0\n")
    result = cli(tmp_path,"--config-dir",str(tmp_path),"--config-name","custom","hydra.job.chdir=true")
    assert result.returncode==0,result.stdout+result.stderr
    run = next((tmp_path/"artifacts/config_runs").iterdir())
    assert json.loads((run/"resolved.json").read_text())["name"]=="experiment_7"
    assert json.loads((run/"assets.json").read_text())["table"]["path"]==str(tmp_path/"table.bin")
