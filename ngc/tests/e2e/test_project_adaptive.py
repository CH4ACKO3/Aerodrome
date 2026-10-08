"""真实故障注入、离线数据生成、在线估计和闭环恢复的整体比较。"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_adaptive_control_recovers_unseen_efficacy(tmp_path):
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root / "projects/adaptive_control/run.py"), "--output", str(tmp_path)],
                   cwd=root, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / "metrics.json").read_text())
    rows = result["evaluation"]
    assert len(rows) == 24
    for efficacy in (0.62, 0.82):
        for seed in (101, 102):
            group = {r["method"]: r for r in rows if r["efficacy_after"] == efficacy and r["seed"] == seed}
            assert group["online"]["late_rmse_m"] < 0.01
            assert group["learned"]["late_rmse_m"] < group["fixed"]["late_rmse_m"]
            assert group["online"]["recovery_s"] is not None
            with np.load(tmp_path / group["online"]["trace"]) as saved:
                trace = saved["trajectory"]
                # 实际作用而非画图伪造：输入通道变化与动力学速度增量一致。
                np.testing.assert_allclose(trace[:, 6], trace[:, 7] * trace[:, 5], atol=1e-12)
                np.testing.assert_allclose(np.diff(trace[:, 2]), 0.05 * (trace[:-1, 6] - 1.2), atol=1e-12)
    assert all(r["late_rmse_m"] < 0.02 for r in rows if r["efficacy_after"] == 1)
    with np.load(tmp_path / "training.npz") as training:
        assert not np.any(np.isin(training["efficacy"], [0.62, 0.82]))
