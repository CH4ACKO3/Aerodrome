"""Check platform build payloads and ensure sdist excludes generated output."""
from hashlib import sha256
import json
from pathlib import Path
import tarfile
import zipfile


def inspect_archives(root):
    report = {}
    baseline = None
    for platform in ("windows","linux","macos"):
        folder = root/platform/"dist"
        wheel, = folder.glob("*.whl")
        source, = folder.glob("*.tar.gz")
        with zipfile.ZipFile(wheel) as archive:
            payload = {name:sha256(archive.read(name)).hexdigest() for name in sorted(archive.namelist())}
        required = ("aerodrome/configuration/conf/experiment.yaml",
                    "aerodrome/models/data/f16/longitudinal.npz",
                    "aerodrome/models/data/f16/manifest.json",
                    "aerodrome/models/data/f16/LICENSE",
                    "aerodrome/adapters/gymnasium.py",
                    "aerodrome/configuration/hydra_cli.py")
        if any(name not in payload for name in required):
            raise ValueError(f"{platform}: missing package code or data")
        if baseline is None:
            baseline = payload
        elif payload!=baseline:
            raise ValueError(f"{platform}: wheel payload differs across platforms")
        with tarfile.open(source) as archive:
            names = archive.getnames()
        if any(set(Path(name).parts)&{"artifacts","__pycache__",".pytest_cache",".venv"} for name in names):
            raise ValueError(f"{platform}: source archive contains generated output")
        if not any(name.endswith("/pyproject.toml") for name in names):
            raise ValueError(f"{platform}: source archive has no build configuration")
        report[platform] = dict(wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),
                                sdist_sha256=sha256(source.read_bytes()).hexdigest(),
                                wheel_bytes=wheel.stat().st_size,sdist_bytes=source.stat().st_size,
                                wheel_files=len(payload),sdist_files=len(names))
    return dict(status="passed",identical_wheel_payloads=True,platforms=report)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    report = json.dumps(inspect_archives(args.root),indent=2)
    if args.output:
        args.output.write_text(report,encoding="utf-8")
    print(report)
