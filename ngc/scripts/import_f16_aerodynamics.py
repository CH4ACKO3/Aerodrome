"""Offline full-table converter: numpy + h5py; neither HDF5 nor h5py at runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from import_f16_longitudinal import COMMIT, SOURCE_HASH


# Channels sharing axes are interpolated together, without resampling.
GROUPS = {
    "longitudinal": (("alpha1", "beta1", "dh1"), ("Cx", "Cz", "Cm")),
    "lateral": (("alpha1", "beta1", "dh2"), ("Cl", "Cn")),
    "side_controls": (("alpha1", "beta1"),
                      ("Cy", "Cy_a20", "Cl_a20", "Cn_a20", "Cy_r30", "Cl_r30", "Cn_r30")),
    "lef": (("alpha2", "beta1"),
            ("Cx_lef", "Cy_lef", "Cz_lef", "Cl_lef", "Cm_lef", "Cn_lef",
             "Cy_a20_lef", "Cl_a20_lef", "Cn_a20_lef")),
    "rates": (("alpha1",), ("Cxq", "Czq", "Cmq", "Cyp", "Cyr", "Clp", "Clr", "Cnp", "Cnr",
                            "deltaClbeta", "deltaCnbeta", "deltaCm")),
    "lef_rates": (("alpha2",), ("deltaCxq_lef", "deltaCzq_lef", "deltaCmq_lef",
                               "deltaCyp_lef", "deltaCyr_lef", "deltaClp_lef", "deltaClr_lef",
                               "deltaCnp_lef", "deltaCnr_lef")),
    "elevator": (("dh1",), ("eta_el",)),
}


def convert(source, destination):
    source, destination = Path(source), Path(destination)
    raw = source / "F16AeroData.h5"
    if hashlib.sha256(raw.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("F16 source hash mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw, "r") as data:
        arrays = {key: np.asarray(data[key]) for key in ("alpha1", "alpha2", "beta1", "dh1", "dh2")}
        for group, (axes, channels) in GROUPS.items():
            shape = tuple(len(arrays[axis]) for axis in axes)
            arrays[group] = np.stack([np.asarray(data["_" + name]).reshape(shape, order="F")
                                      for name in channels], axis=-1)
    target = destination / "aerodynamics.npz"
    np.savez_compressed(target, **arrays)
    shutil.copyfile(source / "LICENSE", destination / "LICENSE")
    manifest = dict(source="https://github.com/isrlab/F16-Model-Matlab", commit=COMMIT,
                    source_sha256=SOURCE_HASH, npz_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    extraction="MATLAB reshape(order=F); all 43 tables, grouped channels; no resampling",
                    axis_units="degrees",
                    groups={key: dict(axes=axes, channels=channels) for key, (axes, channels) in GROUPS.items()})
    (destination / "aerodynamics.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
