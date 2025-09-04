# Run with: blender -b -P blender_scripts/add_constraint_test.py -- output/test_full_rig.glb output/output.blend
import bpy
import sys, os, math

def parse_args(argv):
    """
    Expect:
      blender ... -P add_constraint_test.py -- <input_gltf_or_glb> [output_blend]
    """
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    if not argv:
        raise SystemExit("Usage: blender -b -P add_constraint_test.py -- <input.gltf|.glb> [output.blend]")
    in_path = os.path.abspath(argv[0])
    out_path = os.path.abspath(argv[1]) if len(argv) > 1 else None
    return in_path, out_path

def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_gltf(path):
    # Supports .gltf and .glb
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".gltf", ".glb"}:
        raise SystemExit(f"Input must be .gltf or .glb, got: {ext}")
    bpy.ops.import_scene.gltf(filepath=path)

def add_limit_rotation_to_all_bones(y_min_deg=-100.0, y_max_deg=+100.0):
    """
    For every Armature object's pose bones:
      - set rotation_mode to 'XYZ'
      - add a Limit Rotation constraint that:
        - clamps X to [0, 0] (no rotation)
        - clamps Y to [y_min_deg, y_max_deg] 
        - clamps Z to [0, 0] (no rotation)
      - operates in LOCAL space
    """
    y_min_rad = math.radians(y_min_deg)
    y_max_rad = math.radians(y_max_deg)
    zero_rad = math.radians(0.0)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']

    total_bones = 0
    edited_bones = 0

    for arm in armatures:
        # Ensure we’re operating on the armature’s pose bones
        for pbone in arm.pose.bones:
            total_bones += 1

            # Ensure Euler so X means X as expected
            try:
                pbone.rotation_mode = 'XYZ'
            except Exception:
                pass

            # Reuse existing constraint if present
            existing = None
            for c in pbone.constraints:
                if c.type == 'LIMIT_ROTATION' and getattr(c, "name", "").startswith("Limit All Axes"):
                    existing = c
                    break

            con = existing or pbone.constraints.new('LIMIT_ROTATION')
            if not existing:
                con.name = "Limit All Axes (Y ±100°)"

            # Limit X axis to 0 (no rotation)
            con.use_limit_x = True
            con.min_x = zero_rad
            con.max_x = zero_rad

            # Limit Y axis to specified range
            con.use_limit_y = True
            con.min_y = y_min_rad
            con.max_y = y_max_rad

            # Limit Z axis to 0 (no rotation)
            con.use_limit_z = True
            con.min_z = zero_rad
            con.max_z = zero_rad

            # Local (bone) space is usually what you want for per-joint limits
            con.owner_space = 'LOCAL'
            edited_bones += 1

    return armatures, total_bones, edited_bones

def main():
    in_path, out_path = parse_args(sys.argv)
    clean_scene()
    import_gltf(in_path)
    arms, total_bones, edited_bones = add_limit_rotation_to_all_bones(-100.0, 100.0)

    print(f"[INFO] Armatures found: {len(arms)}")
    print(f"[INFO] Pose bones seen: {total_bones}")
    print(f"[INFO] Constraints applied/updated: {edited_bones}")

    # Save a .blend if requested
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=out_path)
        print(f"[OK] Saved .blend to: {out_path}")
    else:
        print("[OK] Done (no .blend saved).")

if __name__ == "__main__":
    main()
