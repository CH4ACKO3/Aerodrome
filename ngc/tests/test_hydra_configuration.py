"""Run the optional Hydra frontend against the real Python 3.14 argparse."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

pytest.importorskip("hydra")
from aerodrome.configuration.hydra_cli import compatible_parser


def cli(tmp_path,*args):
    env = dict(os.environ,HYDRA_FULL_ERROR="1",PYTHONPATH=str(Path(__file__).parents[1]/"src"))
    return subprocess.run([sys.executable,"-m","aerodrome.configuration.hydra_cli",*args],
                          cwd=tmp_path,env=env,text=True,capture_output=True,timeout=120)


def test_compatibility_is_local():
    import hydra._internal.utils as utils
    original = argparse.ArgumentParser._check_help
    factory = utils.get_args_parser
    with ThreadPoolExecutor(max_workers=2) as pool:
        parsers = list(pool.map(lambda _:compatible_parser(),range(4)))
    assert argparse.ArgumentParser._check_help is original
    assert utils.get_args_parser is factory and utils.argparse is argparse
    for parser in parsers:
        assert "shell-completion" in parser.format_help()
        assert parser.parse_args(["-m","seed=1,2"]).multirun
    with pytest.raises(ValueError):
        argparse.ArgumentParser().add_argument("--invalid",help="%(missing)s")


@pytest.mark.parametrize("args,expected",[
    (["--help"],"scenario: f16, rigid"),
    (["--hydra-help"],"shell-completion"),
    (["--shell-completion","--help"],"powered by Hydra"),
    (["--cfg","job","--resolve","scenario=f16","environment=earth"],"f16_longitudinal"),
    (["--info","defaults-tree"],"scenario: rigid"),
    (["--shell-completion","install=bash"],"complete"),
    (["action=validate"],"mass_kg: 1000.0"),
])
def test_native_commands(tmp_path,args,expected):
    result = cli(tmp_path,*args)
    assert result.returncode==0,result.stdout+result.stderr
    assert expected in result.stdout


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
        assert not json.loads((p/"provenance.json").read_text())["gil_enabled"]


def test_validation_rejects_coercion_and_unknown_fields(tmp_path):
    for override in ("runtime.steps=true","+runtime.typo=1"):
        result = cli(tmp_path,"action=validate",override)
        assert result.returncode!=0
        assert "ValidationError" in result.stderr


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
