#!/usr/bin/env python3
"""
Blender Puppet Panel
Creates bone constraints and sliders from MJCF joint ranges and optionally streams
the slider targets over UDP for external consumers (e.g., MuJoCo) to drive the robot.

Usage (headless):
  blender -b -P blender_scripts/puppet_panel.py -- <input.glb> <model.xml> [--stream] [--ip 127.0.0.1] [--port 31001] [--hz 60] [--save /path/out.blend]

Usage (interactive UI):
  blender -P blender_scripts/puppet_panel.py -- <input.glb> <model.xml>
  Then use the N-Panel → "Puppet (MJCF)" to view sliders and start streaming.

Assumptions:
  - Armature bone names match MJCF joint names (as produced by glTF armature builder).
  - MJCF compiler angle units are radians (common in provided model).
"""

import bpy
import sys, os, math, json, socket, time
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List, Optional


# -----------------------------
# CLI argument parsing
# -----------------------------

def parse_args(argv: List[str]) -> Tuple[Optional[str], Optional[str], Dict[str, str]]:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    in_glb = None
    in_mjcf = None
    opts: Dict[str, str] = {"--ip": "127.0.0.1", "--port": "31001", "--hz": "60", "--stream": "0"}

    if len(argv) >= 2:
        in_glb = os.path.abspath(argv[0])
        in_mjcf = os.path.abspath(argv[1])
        i = 2
        while i < len(argv):
            if argv[i] in {"--ip", "--port", "--hz", "--save"}:
                if i + 1 < len(argv):
                    opts[argv[i]] = argv[i + 1]
                    i += 2
                else:
                    i += 1
            elif argv[i] == "--stream":
                opts["--stream"] = "1"
                i += 1
            else:
                i += 1
    return in_glb, in_mjcf, opts


# -----------------------------
# Scene utilities
# -----------------------------

def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_gltf(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".gltf", ".glb"}:
        raise SystemExit(f"Input must be .gltf or .glb, got: {ext}")
    bpy.ops.import_scene.gltf(filepath=path)


def get_first_armature() -> Optional[bpy.types.Object]:
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


# -----------------------------
# MJCF parsing (minimal, no SciPy deps)
# -----------------------------

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
    """
    Returns a dict: joint_name -> {"type": str, "axis": (x,y,z), "range": (min,max) in radians or None}
    Assumes <compiler angle="radian">; will not convert units.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    specs: Dict[str, Dict] = {}

    # Collect defaults by class for inherited attributes (simple, only axis/range relevant here)
    class_defaults: Dict[str, Dict[str, Dict]] = {}
    default_elem = root.find('default')
    if default_elem is not None:
        for d in default_elem.findall('default'):
            cls = d.get('class')
            if not cls:
                continue
            class_defaults[cls] = {}
            joint_d = d.find('joint')
            if joint_d is not None:
                class_defaults[cls]['axis'] = _parse_vec3(joint_d.get('axis')) if joint_d.get('axis') else None
                class_defaults[cls]['range'] = _parse_range(joint_d.get('range')) if joint_d.get('range') else None

    def resolve_joint_attrs(joint_elem: ET.Element) -> Tuple[str, str, Tuple[float, float, float], Optional[Tuple[float, float]]]:
        jname = joint_elem.get('name', 'unnamed_joint')
        jtype = joint_elem.get('type', 'hinge')
        jclass = joint_elem.get('class')
        axis = _parse_vec3(joint_elem.get('axis')) if joint_elem.get('axis') else None
        rng = _parse_range(joint_elem.get('range')) if joint_elem.get('range') else None
        if (axis is None or rng is None) and jclass and jclass in class_defaults:
            defaults = class_defaults[jclass]
            if axis is None and 'axis' in defaults and defaults['axis'] is not None:
                axis = defaults['axis']
            if rng is None and 'range' in defaults and defaults['range'] is not None:
                rng = defaults['range']
        if axis is None:
            axis = (0.0, 0.0, 1.0)
        return jname, jtype, axis, rng

    worldbody = root.find('worldbody')
    def walk_body(body_elem: ET.Element):
        # joint
        j = body_elem.find('joint')
        if j is not None:
            jname, jtype, axis, rng = resolve_joint_attrs(j)
            specs[jname] = {"type": jtype, "axis": axis, "range": rng}
        # freejoint
        fj = body_elem.find('freejoint')
        if fj is not None:
            jname = fj.get('name', 'unnamed_freejoint')
            specs[jname] = {"type": "free", "axis": None, "range": None}
        # recurse
        for child in body_elem.findall('body'):
            walk_body(child)

    if worldbody is not None:
        for body in worldbody.findall('body'):
            walk_body(body)

    return specs


# -----------------------------
# Bone setup: constraints, sliders, drivers
# -----------------------------

def choose_axis_channel(axis: Tuple[float, float, float]) -> Tuple[int, float]:
    """Return (channel_index, sign) mapping for a hinge axis.
    Picks the dominant axis among X(0), Y(1), Z(2) with sign from the component.
    """
    ax = [abs(axis[0]), abs(axis[1]), abs(axis[2])]
    ch = int(ax.index(max(ax)))
    sgn = 1.0
    comp = [axis[0], axis[1], axis[2]][ch]
    if comp < 0:
        sgn = -1.0
    return ch, sgn


def ensure_joint_setup(arm: bpy.types.Object, joint_specs: Dict[str, Dict]) -> Tuple[int, int]:
    total = 0
    edited = 0
    for pbone in arm.pose.bones:
        bname = pbone.name
        if bname not in joint_specs:
            continue
        spec = joint_specs[bname]
        if spec.get("type") not in {"hinge", "slide"}:  # handle hinge; slide unsupported here
            continue
        total += 1

        # Rotation mode
        try:
            pbone.rotation_mode = 'XYZ'
        except Exception:
            pass

        axis = spec.get("axis", (0.0, 1.0, 0.0))
        ch, sgn = choose_axis_channel(axis)
        rng = spec.get("range")  # radians or None

        # Custom properties for UI/driver
        if "target_deg" not in pbone:
            pbone["target_deg"] = 0.0
        # Store metadata
        min_deg = -180.0
        max_deg = 180.0
        if rng is not None:
            min_deg = math.degrees(rng[0])
            max_deg = math.degrees(rng[1])
        pbone["_min_deg"] = float(min_deg)
        pbone["_max_deg"] = float(max_deg)
        pbone["_channel"] = float(ch)
        pbone["_sign"] = float(sgn)

        # Limit Rotation constraint
        con = None
        for c in pbone.constraints:
            if c.type == 'LIMIT_ROTATION' and getattr(c, 'name', '').startswith('MJCF Limit'):
                con = c
                break
        if con is None:
            con = pbone.constraints.new('LIMIT_ROTATION')
            con.name = 'MJCF Limit'
        con.owner_space = 'LOCAL'

        # Initialize zero limits on all, then set driven axis from MJCF
        con.use_limit_x = True; con.min_x = 0.0; con.max_x = 0.0
        con.use_limit_y = True; con.min_y = 0.0; con.max_y = 0.0
        con.use_limit_z = True; con.min_z = 0.0; con.max_z = 0.0

        if rng is not None:
            a = math.radians(min_deg) * sgn
            b = math.radians(max_deg) * sgn
            lo = min(a, b)
            hi = max(a, b)
        else:
            lo = -math.pi
            hi = math.pi

        if ch == 0:
            con.min_x = lo; con.max_x = hi
        elif ch == 1:
            con.min_y = lo; con.max_y = hi
        else:
            con.min_z = lo; con.max_z = hi

        # Driver: rotation_euler[ch] = radians(target_deg) * sign
        # Remove existing driver on that channel to avoid duplicates
        try:
            fcurve = pbone.driver_add("rotation_euler", ch)
        except TypeError:
            # If driver exists, Blender returns the existing fcurve; continue
            fcurve = pbone.driver_add("rotation_euler", ch)
        drv = fcurve.driver
        drv.type = 'SCRIPTED'
        drv.expression = "radians(val) * sgn"
        # Clear existing variables
        while drv.variables:
            drv.variables.remove(drv.variables[0])
        var = drv.variables.new()
        var.name = "val"
        var.targets[0].id = arm
        var.targets[0].data_path = f'pose.bones["{bname}"]["target_deg"]'
        var2 = drv.variables.new()
        var2.name = "sgn"
        var2.targets[0].id = arm
        var2.targets[0].data_path = f'pose.bones["{bname}"]["_sign"]'

        edited += 1
    return total, edited


# -----------------------------
# UDP streaming via timer
# -----------------------------

class UdpStreamer:
    def __init__(self, arm: bpy.types.Object, ip: str = "127.0.0.1", port: int = 31001, hz: float = 60.0):
        self.arm = arm
        self.ip = ip
        self.port = int(port)
        self.hz = float(hz)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self._registered = False

    def _collect(self) -> Dict:
        payload = {"t": time.time(), "seq": int(self.seq), "joints": []}
        for pbone in self.arm.pose.bones:
            if "target_deg" not in pbone:
                continue
            name = pbone.name
            min_deg = float(pbone.get("_min_deg", -180.0))
            max_deg = float(pbone.get("_max_deg", 180.0))
            val_deg = float(pbone.get("target_deg", 0.0))
            # Clamp to range
            if val_deg < min_deg:
                val_deg = min_deg
                pbone["target_deg"] = val_deg
            elif val_deg > max_deg:
                val_deg = max_deg
                pbone["target_deg"] = val_deg
            sgn = float(pbone.get("_sign", 1.0))
            val_rad = math.radians(val_deg) * sgn
            payload["joints"].append({"name": name, "value": val_rad})
        return payload

    def _tick(self):
        try:
            payload = self._collect()
            self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.ip, self.port))
            self.seq += 1
        except Exception as e:
            print(f"[UDP] Error: {e}")
        return 1.0 / self.hz

    def start(self):
        if not self._registered:
            bpy.app.timers.register(self._tick, first_interval=1.0 / self.hz, persistent=True)
            self._registered = True

    def stop(self):
        # There is no direct unregister by handle; simplest is to set hz to a very low value
        self.hz = 1e9
        self._registered = False


# Keep a module-level streamer for UI buttons
_STREAMER: Optional[UdpStreamer] = None


# -----------------------------
# UI Panel for interactive control
# -----------------------------

class VIEW3D_PT_puppet_panel(bpy.types.Panel):
    bl_label = "Puppet (MJCF)"
    bl_idname = "VIEW3D_PT_puppet_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Puppet (MJCF)'

    def draw(self, context):
        layout = self.layout
        arm = get_first_armature()
        if arm is None:
            layout.label(text="No armature found.")
            return
        col = layout.column(align=True)
        col.label(text=f"Armature: {arm.name}")
        col.separator()
        # Draw sliders
        for pbone in arm.pose.bones:
            if "target_deg" in pbone:
                row = col.row(align=True)
                row.prop(pbone, '["target_deg"]', text=pbone.name, slider=True)
                # Show range summary
                min_deg = pbone.get("_min_deg", None)
                max_deg = pbone.get("_max_deg", None)
                if min_deg is not None and max_deg is not None:
                    row = col.row(align=True)
                    row.label(text=f"[{min_deg:.1f}, {max_deg:.1f}] deg")
        col.separator()
        col.operator("puppet.start_stream", text="Start Streaming", icon='PLAY')
        col.operator("puppet.stop_stream", text="Stop Streaming", icon='PAUSE')


class PUPPET_OT_start_stream(bpy.types.Operator):
    bl_idname = "puppet.start_stream"
    bl_label = "Start Puppet UDP Streaming"

    ip: bpy.props.StringProperty(name="IP", default="127.0.0.1")
    port: bpy.props.IntProperty(name="Port", default=31001, min=1, max=65535)
    hz: bpy.props.FloatProperty(name="Hz", default=60.0, min=1.0, max=240.0)

    def execute(self, context):
        global _STREAMER
        arm = get_first_armature()
        if arm is None:
            self.report({'ERROR'}, "No armature found")
            return {'CANCELLED'}
        _STREAMER = UdpStreamer(arm, self.ip, self.port, self.hz)
        _STREAMER.start()
        self.report({'INFO'}, f"Streaming to {self.ip}:{self.port} @ {self.hz} Hz")
        return {'FINISHED'}


class PUPPET_OT_stop_stream(bpy.types.Operator):
    bl_idname = "puppet.stop_stream"
    bl_label = "Stop Puppet UDP Streaming"

    def execute(self, context):
        global _STREAMER
        if _STREAMER is not None:
            _STREAMER.stop()
            _STREAMER = None
            self.report({'INFO'}, "Streaming stopped")
            return {'FINISHED'}
        self.report({'WARNING'}, "Streamer was not running")
        return {'CANCELLED'}


def register_ui():
    bpy.utils.register_class(VIEW3D_PT_puppet_panel)
    bpy.utils.register_class(PUPPET_OT_start_stream)
    bpy.utils.register_class(PUPPET_OT_stop_stream)


def unregister_ui():
    bpy.utils.unregister_class(VIEW3D_PT_puppet_panel)
    bpy.utils.unregister_class(PUPPET_OT_start_stream)
    bpy.utils.unregister_class(PUPPET_OT_stop_stream)


# -----------------------------
# Main
# -----------------------------

def main():
    in_glb, in_mjcf, opts = parse_args(sys.argv)
    is_headless = not bpy.app.background is False  # Blender sets background True in -b

    if in_glb:
        clean_scene()
        import_gltf(in_glb)

    arm = get_first_armature()
    if arm is None:
        print("[WARN] No armature in scene. If running interactively, import your GLB first.")

    joint_specs = {}
    if in_mjcf and os.path.isfile(in_mjcf):
        print(f"[INFO] Loading MJCF joints from: {in_mjcf}")
        joint_specs = load_mjcf_joint_specs(in_mjcf)
        print(f"[INFO] Parsed joints: {len(joint_specs)}")
    else:
        print("[WARN] No MJCF provided; will create sliders for any existing bones without limits.")

    if arm is not None:
        total, edited = ensure_joint_setup(arm, joint_specs)
        print(f"[INFO] Bones matched to joints: {total}")
        print(f"[INFO] Constraints/drivers created: {edited}")

    # Optional streaming in headless mode
    if opts.get("--stream", "0") == "1" and arm is not None:
        ip = opts.get("--ip", "127.0.0.1")
        port = int(opts.get("--port", "31001"))
        hz = float(opts.get("--hz", "60"))
        print(f"[INFO] Starting UDP streaming to {ip}:{port} @ {hz} Hz")
        streamer = UdpStreamer(arm, ip, port, hz)
        streamer.start()
        # In background mode, run for a short period to emit data, then exit
        if bpy.app.background:
            # Run for ~5 seconds
            t0 = time.time()
            while time.time() - t0 < 5.0:
                time.sleep(1.0 / hz)
            streamer.stop()

    # Save blend if requested
    out_path = opts.get("--save")
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_path))
        print(f"[OK] Saved .blend to: {out_path}")


if __name__ == "__main__":
    try:
        register_ui()
    except Exception:
        pass
    main()


