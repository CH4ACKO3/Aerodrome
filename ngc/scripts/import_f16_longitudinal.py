"""Offline converter only: Python + numpy + h5py. Runtime does not need h5py."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import h5py
import numpy as np

SOURCE_HASH = "b9ac8d21cfb749c0e9897766d5f7b3cdcfdafc86ef8e948c064aafdc2e453f07"
COMMIT = "d019742f2fe7f5f25c521bc7e846519b28b759b3"


def convert(source, destination):
    source,destination = Path(source),Path(destination)
    raw = source/"F16AeroData.h5"
    if hashlib.sha256(raw.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("F16 source hash mismatch")
    destination.mkdir(parents=True,exist_ok=True)
    arrays = {}
    with h5py.File(raw,"r") as f:
        for key in ("alpha1","alpha2","dh1"):
            arrays[key] = np.asarray(f[key])
        beta = np.asarray(f["beta1"])
        zero = int(np.flatnonzero(beta == 0)[0])
        for name in ("Cx","Cz","Cm"):
            original = np.asarray(f["_"+name])
            table = original.reshape((len(arrays["alpha1"]),len(beta),len(arrays["dh1"])),order="F")
            arrays[name] = table[:,zero,:]
            # Independently check MATLAB column-major scalar indexing.
            for a in range(table.shape[0]):
                for e in range(table.shape[2]):
                    assert arrays[name][a,e] == original[a+table.shape[0]*(zero+len(beta)*e)]
        for name in ("Cx_lef","Cz_lef","Cm_lef"):
            arrays[name] = np.asarray(f["_"+name]).reshape((len(arrays["alpha2"]),len(beta)),order="F")[:,zero]
        for name in ("Cxq","Czq","Cmq","deltaCxq_lef","deltaCzq_lef","deltaCmq_lef","deltaCm","eta_el"):
            arrays[name] = np.asarray(f["_"+name])
    target = destination/"longitudinal.npz"
    np.savez(target,**arrays)
    shutil.copyfile(source/"LICENSE",destination/"LICENSE")
    manifest = dict(source="https://github.com/isrlab/F16-Model-Matlab",commit=COMMIT,
                    source_sha256=SOURCE_HASH,npz_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    extraction="MATLAB reshape(order=F), exact beta=0 grid slice; no resampling",
                    axis_units="degrees",tables={k:list(v.shape) for k,v in arrays.items()})
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("destination",type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.source,args.destination),indent=2))
