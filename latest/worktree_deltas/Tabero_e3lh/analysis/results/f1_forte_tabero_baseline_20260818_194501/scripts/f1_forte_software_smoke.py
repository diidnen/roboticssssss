#!/usr/bin/env python3
"""F1-D: FORTE software-only smoke. External wrapper. Does not patch FORTE source.

SYNTHETIC unit tests are labeled SYNTHETIC_SOFTWARE_TEST_ONLY.
This is NOT a paper-demo reproduction.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import traceback
from pathlib import Path

OUT = Path(os.environ.get("F1_OUT", Path(__file__).resolve().parents[1]))
FORTE_ROOT = Path("/home/exouser/FORTE")
sys.path.insert(0, str(FORTE_ROOT))

report: dict = {
    "label": "SOFTWARE_SMOKE",
    "python": sys.version,
    "cwd": os.getcwd(),
    "imports": {},
    "force_estimator": {},
    "slip_detector": {},
    "controller": {},
    "offline_replay": {},
    "hardware": {},
    "synthetic_tests": [],
    "errors": [],
}


def rec(ok: bool, **kw):
    return {"ok": ok, **kw}


def main():
    import numpy as np

    # --- imports ---
    modules = [
        "forte",
        "forte.sensing",
        "forte.sensing.sensor",
        "forte.sensing.sensor_buffer",
        "forte.sensing.utils",
        "forte.runtime",
        "forte.runtime.force_and_slip",
        "forte.runtime.sys_utils",
        "forte_gripper",
        "forte_gripper.FORTE_gripper",
        "forte_gripper.command",
        "forte_gripper.bulk_gripper",
    ]
    for name in modules:
        try:
            __import__(name)
            report["imports"][name] = rec(True)
        except Exception as e:
            report["imports"][name] = rec(False, error=repr(e))
            report["errors"].append({"import": name, "error": repr(e)})

    from forte.runtime.force_and_slip import (
        NUM_CHANNELS,
        SENSOR_HZ,
        BUFFER_SIZE,
        compute_slip_indicator,
        compute_window_psd,
        load_model,
        update_psd_history,
        welch,
    )
    from forte.runtime.sys_utils import sensor2force_feature

    # --- hardware presence ---
    serial_devs = []
    for p in ["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyACM1", "/dev/ttyUSB1"]:
        serial_devs.append({"path": p, "exists": os.path.exists(p)})
    report["hardware"] = {
        "serial_devices": serial_devs,
        "any_forte_serial": any(d["exists"] for d in serial_devs),
        "note": "No FORTE sensor (ttyACM0) or Dynamixel U2D2 (ttyUSB0) on this machine.",
    }

    # --- offline replay ---
    repo_data = []
    for pat in ["*.h5", "*.hdf5", "*sensor_log*.csv", "*slip_log*.csv", "*force_log*.csv", "*.npz"]:
        repo_data.extend([str(p.relative_to(FORTE_ROOT)) for p in FORTE_ROOT.rglob(pat) if ".venv" not in str(p) and ".git" not in str(p)])
    report["offline_replay"] = {
        "available": False,
        "status": "OFFLINE_SENSOR_REPLAY_NOT_AVAILABLE",
        "files_found_in_repo": repo_data,
        "note": "Official repo ships SVR_ckpt.pkl but no recorded tactile traces.",
    }

    # --- force model ---
    model_path = FORTE_ROOT / "models" / "SVR_ckpt.pkl"
    model = None
    try:
        model = load_model(str(model_path))
        info = {
            "ok": model is not None,
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "type": type(model).__name__,
            "module": type(model).__module__,
        }
        for attr in ["kernel", "C", "gamma", "epsilon", "n_features_in_", "n_support_", "support_"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if attr == "support_":
                    info["n_support_vectors"] = int(np.asarray(val).shape[0])
                else:
                    info[attr] = val if not hasattr(val, "tolist") else (val if np.ndim(val) == 0 else str(val))
        if hasattr(model, "get_params"):
            params = model.get_params()
            info["get_params"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in params.items()}
        # sklearn Pipeline?
        if hasattr(model, "steps"):
            info["pipeline_steps"] = [s[0] for s in model.steps]
            last = model.steps[-1][1]
            info["final_estimator_type"] = type(last).__name__
            if hasattr(last, "kernel"):
                info["final_kernel"] = last.kernel
            if hasattr(last, "n_features_in_"):
                info["final_n_features"] = int(last.n_features_in_)
        report["force_estimator"] = info
    except Exception as e:
        report["force_estimator"] = rec(False, error=repr(e), trace=traceback.format_exc())
        report["errors"].append({"force_estimator": repr(e)})

    report["force_estimator"].update(
        {
            "input_feature_fn": "sensor2force_feature",
            "input_dim": 24,
            "feature_layout": "concat(last_6, mean_5k_6, mean_10k_6, mean_20k_6) at 2 kHz => 0 / 2.5s / 5s / 10s windows",
            "window_frames": 20000,
            "loop_hz_code": 100,
            "output": "scalar predicted force (paper: Newtons, 0-8 N range)",
            "checkpoint": "models/SVR_ckpt.pkl",
        }
    )

    # --- slip detector code constants ---
    src = inspect.getsource(sys.modules["forte.runtime.force_and_slip"].slip_predictor)
    report["slip_detector"] = {
        "loaded": True,
        "input": "6-channel filtered analog tactile, 2000 Hz",
        "method": "Welch PSD max in [10,50] Hz then moving variance; left=ch0-2, right=ch3-5",
        "nperseg": 400,
        "noverlap_frac": 0.99,
        "var_window": 15,
        "band_hz": [10, 50],
        "monotonic": True,
        "monotonic_delta_db": 0.1,
        "threshold_condition_in_code": False,
        "threshold_db_unused_because_flag_false": -72.0,
        "single_finger_suppress_alpha": 0.6,
        "slip_var_threshold_T": 2,
        "loop_sleep_s": 0.002,
        "loop_hz_nominal": 500,
        "output": "[slip in {0,1}, avgL, avgR]",
        "latency_paper_ms": "<100",
        "source_excerpt_present": "nperseg = 400" in src,
    }

    # --- controller instantiate ---
    try:
        from forte_gripper import FORTE_gripper

        gripper = FORTE_gripper(str(FORTE_ROOT / "configs/actuator/FORTE_gripper.yaml"))
        report["controller"]["instantiate"] = rec(True, note="unexpected: serial opened")
        try:
            gripper.shutdown()
        except Exception:
            pass
    except Exception as e:
        report["controller"]["instantiate"] = rec(
            False,
            expected=True,
            error=repr(e),
            note="REQUIRES_PHYSICAL_FORTE_HARDWARE (Dynamixel /dev/ttyUSB0)",
        )

    try:
        from omegaconf import OmegaConf
        from forte.sensing import FORTE_sensor

        cfg = OmegaConf.load(str(FORTE_ROOT / "configs/sensor/FORTE_sensor.yaml"))
        sensor = FORTE_sensor(cfg.FORTE)
        report["controller"]["sensor_init"] = rec(True, note="object constructed; start() would open serial")
        # do not call start() — it opens /dev/ttyACM0
        del sensor
    except Exception as e:
        report["controller"]["sensor_init"] = rec(False, error=repr(e))

    report["controller"]["logic_source"] = "examples/gripper_showcase_vis_realtime.py::gripper_control_worker"
    report["controller"]["is_direct_force_servo"] = False
    report["controller"]["actuation"] = "Dynamixel impedance (position + current); slip reaction is a position increment, not a Newton command"

    # --- SYNTHETIC tests ---
    def add_test(name, ok, **kw):
        report["synthetic_tests"].append({"name": name, "ok": bool(ok), "label": "SYNTHETIC_SOFTWARE_TEST_ONLY", **kw})

    # feature dim
    chunk = np.zeros((20000, 6), dtype=float)
    chunk[-1] = np.linspace(0.01, 0.06, 6)
    feat = sensor2force_feature(chunk)
    add_test("feature_dim_24", feat.shape == (24,), shape=list(feat.shape))

    # model predict on zeros
    if model is not None:
        try:
            y0 = float(model.predict(feat.reshape(1, -1))[0])
            add_test("svr_predict_zeroish_window", True, predicted=y0)
            feat2 = feat.copy()
            feat2[:6] += 0.2
            y1 = float(model.predict(feat2.reshape(1, -1))[0])
            add_test("svr_predict_perturbed", True, predicted=y1, delta=y1 - y0)
        except Exception as e:
            add_test("svr_predict_zeroish_window", False, error=repr(e))

    # welch shape
    x = np.sin(2 * np.pi * 20 * np.arange(400) / 2000.0)
    f, Pxx = welch(x, fs=2000.0, nperseg=400, noverlap=int(0.99 * 400))
    add_test("welch_runs", len(f) == len(Pxx) and len(f) > 10, n_freq=int(len(f)))

    psd = compute_window_psd(x, 2000.0, 400, int(0.99 * 400), 10, 50)
    add_test("window_psd_finite", np.isfinite(psd), psd_db=float(psd))

    # quiet window -> no slip
    from collections import deque

    hist = [deque(maxlen=15) for _ in range(6)]
    quiet = np.zeros((400, 6))
    for _ in range(20):
        update_psd_history(quiet, 2000.0, 400, int(0.99 * 400), 10, 50, hist)
    slip, avgL, avgR = compute_slip_indicator(hist, 15, True, False, -72.0, 2)
    add_test("quiet_no_slip", slip == 0, slip=int(slip), avgL=float(avgL), avgR=float(avgR))

    # synthetic rising 20 Hz burst on all channels (NOT real FORTE data)
    hist2 = [deque(maxlen=15) for _ in range(6)]
    t = np.arange(400) / 2000.0
    for k in range(20):
        amp = 0.02 * (k + 1)
        win = (amp * np.sin(2 * np.pi * 20 * t))[:, None] * np.ones((1, 6))
        update_psd_history(win, 2000.0, 400, int(0.99 * 400), 10, 50, hist2)
    slip2, avgL2, avgR2 = compute_slip_indicator(hist2, 15, True, False, -72.0, 2)
    add_test(
        "synthetic_20hz_burst_slip_indicator_ran",
        True,
        slip=int(slip2),
        avgL=float(avgL2),
        avgR=float(avgR2),
        note="SYNTHETIC_SOFTWARE_TEST_ONLY; not claimed as FORTE demo reproduction",
    )

    report["synthetic_all_ok"] = all(t["ok"] for t in report["synthetic_tests"])
    report["force_estimator_loaded"] = bool(model is not None)
    report["slip_detector_loaded"] = True
    report["offline_replay_available"] = False
    report["real_hardware_available"] = False

    (OUT / "FORTE_SOFTWARE_SMOKE.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ["force_estimator_loaded", "slip_detector_loaded", "offline_replay", "hardware", "synthetic_all_ok"]}, indent=2, default=str))
    print("WROTE", OUT / "FORTE_SOFTWARE_SMOKE.json")


if __name__ == "__main__":
    main()
