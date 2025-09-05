#!/usr/bin/env python3
"""
UDP receiver that listens for Blender puppet joint targets and applies them to MuJoCo.

Usage (remember to use the project venv):
  Edit the CONFIG section below as needed, then run:
  source .venv/bin/activate && python deployment/udp_mujoco_receiver.py

Message schema from Blender:
  { "t": float_unix_time, "seq": int, "joints": [ {"name": str, "value": float_radians} ] }
"""

import os
import json
import socket
import time
from typing import Dict

import mujoco as mj
import mujoco.viewer as viewer
import numpy as np


# -----------------------------
# CONFIG (edit here)
# -----------------------------
MODEL_PATH = "g1_description/scene_mjx_alt.xml"
UDP_IP = "127.0.0.1"
UDP_PORT = 31001
LOOP_HZ = 500.0
USE_VIEWER = True
USE_EGL = False

# Elastic band to keep robot suspended
USE_ELASTIC = True
ELASTIC_BODY = "torso_link"
ELASTIC_KP = 1200.0
ELASTIC_KD = 400.0
# If True, apply elastic force only along world-Z; otherwise XYZ
ELASTIC_ONLY_Z = True
# If not None, override target Z height in meters; otherwise use initial body height
ELASTIC_TARGET_Z = None
# If set (meters), lift target Z above initial by this offset when ELASTIC_TARGET_Z is None
ELASTIC_Z_OFFSET = 0.15
# Clamp elastic force magnitude (Newton) to avoid violent kicks (None to disable)
ELASTIC_MAX_FORCE = 1000.0

if USE_EGL:
    os.environ["MUJOCO_GL"] = "egl"


def build_name_to_qpos(model: mj.MjModel) -> Dict[str, int]:
    name_to_qpos: Dict[str, int] = {}
    for jid in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            continue
        addr = int(model.jnt_qposadr[jid])
        name_to_qpos[name] = addr
    return name_to_qpos


def main():
    # Prepare UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.0)  # non-blocking
    print(f"[INFO] Listening on {UDP_IP}:{UDP_PORT}")

    # Shared state for latest joint targets
    latest_targets: Dict[str, float] = {}

    def pump_udp_once():
        while True:
            try:
                data_raw, _ = sock.recvfrom(65536)
                msg = json.loads(data_raw.decode("utf-8"))
                for j in msg.get("joints", []):
                    n = j.get("name")
                    v = float(j.get("value", 0.0))
                    if isinstance(n, str):
                        latest_targets[n] = v
            except BlockingIOError:
                break
            except Exception:
                # Ignore malformed packets
                break

    if USE_VIEWER:
        # Viewer path: install a control callback that applies latest UDP targets
        def load_callback(model=None, data=None):
            m = mj.MjModel.from_xml_path(MODEL_PATH)
            d = mj.MjData(m)
            name_to_qpos = build_name_to_qpos(m)
            # Elastic band state
            band_bid = None
            target_pos = None
            use_elastic = bool(USE_ELASTIC)
            kp = float(ELASTIC_KP)
            kd = float(ELASTIC_KD)
            if use_elastic:
                try:
                    band_bid = m.body(ELASTIC_BODY).id
                except Exception:
                    band_bid = None

            def control_cb(_m=None, _d=None):
                pump_udp_once()
                for name, val in latest_targets.items():
                    idx = name_to_qpos.get(name)
                    if idx is not None:
                        d.qpos[idx] = val
                # Apply elastic band in world frame to hold body near initial position
                if use_elastic and band_bid is not None:
                    nonlocal target_pos
                    if target_pos is None:
                        target_pos = d.xpos[band_bid].copy()
                        if ELASTIC_TARGET_Z is not None:
                            target_pos[2] = float(ELASTIC_TARGET_Z)
                        elif ELASTIC_Z_OFFSET is not None:
                            target_pos[2] = target_pos[2] + float(ELASTIC_Z_OFFSET)
                    # Position error in world frame
                    err = target_pos - d.xpos[band_bid]
                    if ELASTIC_ONLY_Z:
                        err[0] = 0.0
                        err[1] = 0.0
                    # Approximate body linear velocity from composite velocity (first 3)
                    try:
                        linvel = np.asarray(d.cvel[band_bid][:3], dtype=float)
                    except Exception:
                        lv = d.qvel[:3] if d.qvel.shape[0] >= 3 else [0.0, 0.0, 0.0]
                        linvel = np.asarray(lv, dtype=float)
                    if ELASTIC_ONLY_Z:
                        linvel[0] = 0.0
                        linvel[1] = 0.0
                    force = kp * err - kd * linvel
                    if ELASTIC_MAX_FORCE is not None:
                        mag = np.linalg.norm(force)
                        if mag > ELASTIC_MAX_FORCE and mag > 0:
                            force = force * (ELASTIC_MAX_FORCE / mag)
                    d.xfrc_applied[band_bid, :3] = force

            mj.set_mjcb_control(control_cb)
            return m, d

        print("[INFO] Launching viewer...")
        viewer.launch(loader=load_callback)
        return

    # Headless path: manual stepping loop
    model = mj.MjModel.from_xml_path(MODEL_PATH)
    data = mj.MjData(model)
    name_to_qpos = build_name_to_qpos(model)
    # Elastic band init
    band_bid = None
    target_pos = None
    use_elastic = bool(USE_ELASTIC)
    kp = float(ELASTIC_KP)
    kd = float(ELASTIC_KD)
    if use_elastic:
        try:
            band_bid = model.body(ELASTIC_BODY).id
        except Exception:
            band_bid = None

    dt = 1.0 / LOOP_HZ
    last = time.time()
    while True:
        pump_udp_once()
        for name, val in latest_targets.items():
            idx = name_to_qpos.get(name)
            if idx is not None:
                data.qpos[idx] = val
        # Apply elastic band in world frame
        if use_elastic and band_bid is not None:
            if target_pos is None:
                target_pos = data.xpos[band_bid].copy()
                if ELASTIC_TARGET_Z is not None:
                    target_pos[2] = float(ELASTIC_TARGET_Z)
                elif ELASTIC_Z_OFFSET is not None:
                    target_pos[2] = target_pos[2] + float(ELASTIC_Z_OFFSET)
            err = target_pos - data.xpos[band_bid]
            if ELASTIC_ONLY_Z:
                err[0] = 0.0
                err[1] = 0.0
            try:
                linvel = np.asarray(data.cvel[band_bid][:3], dtype=float)
            except Exception:
                lv = data.qvel[:3] if data.qvel.shape[0] >= 3 else [0.0, 0.0, 0.0]
                linvel = np.asarray(lv, dtype=float)
            if ELASTIC_ONLY_Z:
                linvel[0] = 0.0
                linvel[1] = 0.0
            force = kp * err - kd * linvel
            if ELASTIC_MAX_FORCE is not None:
                mag = np.linalg.norm(force)
                if mag > ELASTIC_MAX_FORCE and mag > 0:
                    force = force * (ELASTIC_MAX_FORCE / mag)
            data.xfrc_applied[band_bid, :3] = force
        now = time.time()
        steps = int((now - last) / dt) if now - last > 0 else 1
        for _ in range(max(1, steps)):
            mj.mj_step(model, data)
        last = now


if __name__ == "__main__":
    main()


