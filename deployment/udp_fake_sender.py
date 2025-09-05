#!/usr/bin/env python3
"""
Send synthetic joint targets over UDP to drive the MuJoCo receiver without Blender.

Message schema (matches receiver):
  { "t": float_unix_time, "seq": int, "joints": [ {"name": str, "value": float_radians} ] }

Examples:
  python deployment/udp_fake_sender.py --model g1_description/g1_mjx_alt.xml --ip 127.0.0.1 --port 31001 --hz 60 --duration 10
  python deployment/udp_fake_sender.py --model g1_description/g1_mjx_alt.xml --joints left_elbow_joint,right_elbow_joint --speed 0.5 --amp_scale 0.8
"""

import argparse
import json
import math
import socket
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple


def _parse_vec3(text: Optional[str]) -> Tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 1.0)
    vals = [float(x) for x in text.strip().split()]
    if len(vals) == 3:
        return (vals[0], vals[1], vals[2])
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    return (0.0, 0.0, 1.0)


def _parse_range(text: Optional[str]) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    vals = [float(x) for x in text.strip().split()]
    if len(vals) >= 2:
        return (vals[0], vals[1])
    return None


def load_mjcf_joint_specs(mjcf_path: str) -> Dict[str, Dict]:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    specs: Dict[str, Dict] = {}

    # Defaults per class (simplified: joint axis/range only)
    class_defaults: Dict[str, Dict[str, Tuple]] = {}
    default_elem = root.find('default')
    if default_elem is not None:
        for d in default_elem.findall('default'):
            cls = d.get('class')
            if not cls:
                continue
            entry: Dict[str, Tuple] = {}
            j = d.find('joint')
            if j is not None:
                if j.get('axis'):
                    entry['axis'] = _parse_vec3(j.get('axis'))
                if j.get('range'):
                    entry['range'] = _parse_range(j.get('range'))
            if entry:
                class_defaults[cls] = entry

    def resolve_joint(j_elem: ET.Element) -> Tuple[str, str, Tuple[float, float, float], Optional[Tuple[float, float]]]:
        name = j_elem.get('name', 'unnamed_joint')
        jtype = j_elem.get('type', 'hinge')
        jclass = j_elem.get('class')
        axis = _parse_vec3(j_elem.get('axis')) if j_elem.get('axis') else None
        rng = _parse_range(j_elem.get('range')) if j_elem.get('range') else None
        if (axis is None or rng is None) and jclass and jclass in class_defaults:
            d = class_defaults[jclass]
            if axis is None and 'axis' in d:
                axis = d['axis']
            if rng is None and 'range' in d:
                rng = d['range']
        if axis is None:
            axis = (0.0, 0.0, 1.0)
        return name, jtype, axis, rng

    def walk_body(body_elem: ET.Element):
        j = body_elem.find('joint')
        if j is not None:
            name, jtype, axis, rng = resolve_joint(j)
            specs[name] = {"type": jtype, "axis": axis, "range": rng}
        fj = body_elem.find('freejoint')
        if fj is not None:
            name = fj.get('name', 'unnamed_freejoint')
            specs[name] = {"type": "free", "axis": None, "range": None}
        for child in body_elem.findall('body'):
            walk_body(child)

    worldbody = root.find('worldbody')
    if worldbody is not None:
        for b in worldbody.findall('body'):
            walk_body(b)

    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="g1_description/g1_mjx_alt.xml", help="Path to MJCF xml (radians)")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31001)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run (omit or set <0 for infinite)")
    parser.add_argument("--speed", type=float, default=1.0, help="Sine wave angular speed multiplier")
    parser.add_argument("--amp_scale", type=float, default=0.8, help="Fraction of joint range to use (0..1)")
    parser.add_argument("--zeros", action="store_true", help="Send 0.0 for all selected joints instead of sine")
    default_arm_joints = ",".join([
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ])
    parser.add_argument("--joints", type=str, default=default_arm_joints, help="Comma-separated joint names to send; default arms only")
    args = parser.parse_args()

    specs = load_mjcf_joint_specs(args.model)
    all_joint_names = [n for n, s in specs.items() if s.get("type") in {"hinge", "slide"}]

    if args.joints.strip():
        selected = [j.strip() for j in args.joints.split(",") if j.strip()]
        joint_names = [j for j in selected if j in specs]
    else:
        joint_names = all_joint_names

    if not joint_names:
        print("[ERROR] No joints selected.")
        return

    phases: Dict[str, float] = {}
    for i, name in enumerate(joint_names):
        phases[name] = (i / max(1, len(joint_names))) * 2.0 * math.pi

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[INFO] Sending {len(joint_names)} joints to {args.ip}:{args.port} @ {args.hz} Hz")
    t0 = time.time()
    seq = 0
    dt = 1.0 / args.hz

    while True:
        now = time.time()
        elapsed = now - t0
        if args.duration > 0 and elapsed >= args.duration:
            break

        joints: List[Dict] = []
        for name in joint_names:
            if args.zeros:
                val = 0.0
            else:
                spec = specs[name]
                rng = spec.get("range")
                if rng is None:
                    # use default +-30 deg if no range
                    center = 0.0
                    halfspan = math.radians(30.0)
                else:
                    center = 0.5 * (rng[0] + rng[1])
                    halfspan = 0.5 * (rng[1] - rng[0])
                amp = halfspan * max(0.0, min(1.0, args.amp_scale))
                val = center + amp * math.sin(phases[name] + args.speed * elapsed)
            joints.append({"name": name, "value": float(val)})

        payload = {"t": now, "seq": int(seq), "joints": joints}
        sock.sendto(json.dumps(payload).encode("utf-8"), (args.ip, args.port))
        seq += 1
        # Simple rate control
        to_sleep = dt - (time.time() - now)
        if to_sleep > 0:
            time.sleep(to_sleep)

    print("[OK] Finished sending.")


if __name__ == "__main__":
    main()


