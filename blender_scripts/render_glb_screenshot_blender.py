import sys
import argparse
import math
from pathlib import Path

import bpy
from mathutils import Vector, Matrix


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender headless GLB renderer (single or multi view)")
    parser.add_argument("glb_file", type=str, help="Path to the GLB file")
    parser.add_argument("-o", "--output", type=str, help="Output PNG path (default: glb with .png)")
    parser.add_argument("-w", "--width", type=int, default=1024, help="Image width (single) or per-view width (multi)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (single) or per-view height (multi)")
    parser.add_argument("--margin", type=float, default=1.3, help="Frame margin multiplier")
    parser.add_argument("--fov", type=float, default=45.0, help="Horizontal FOV in degrees")
    parser.add_argument("--single-view", action="store_true", help="Render only a single 3/4 view")
    parser.add_argument("--separate-views", action="store_true", help="Save individual view PNGs and skip compositing")
    return parser.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.world = bpy.data.worlds.new("World") if not scene.world else scene.world
    scene.world.color = (0.04, 0.04, 0.04)
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    return scene


def import_glb(glb_path: Path):
    assert glb_path.exists(), f"GLB not found: {glb_path}"
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    return [obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "ARMATURE", "EMPTY"}]


def evaluated_bounds_worldspace(objs):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    has_mesh = False

    min_v = Vector((float("inf"), float("inf"), float("inf")))
    max_v = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objs:
        if obj.type != "MESH":
            continue

        obj_eval = obj.evaluated_get(depsgraph)
        try:
            mesh = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        except TypeError:
            mesh = obj_eval.to_mesh(preserve_all_data_layers=True)

        if not mesh or len(mesh.vertices) == 0:
            if mesh:
                obj_eval.to_mesh_clear()
            continue

        has_mesh = True
        mw = obj_eval.matrix_world
        for v in mesh.vertices:
            w = mw @ v.co
            min_v.x = min(min_v.x, w.x)
            min_v.y = min(min_v.y, w.y)
            min_v.z = min(min_v.z, w.z)
            max_v.x = max(max_v.x, w.x)
            max_v.y = max(max_v.y, w.y)
            max_v.z = max(max_v.z, w.z)

        obj_eval.to_mesh_clear()

    if not has_mesh:
        raise RuntimeError("No mesh geometry found after evaluation.")

    center = (min_v + max_v) * 0.5
    extents = max_v - min_v
    return (min_v, max_v), center, extents


def compute_camera_distance(extents, hfov_deg, aspect, margin=1.3):
    # Conservative fit: use the largest extent against both H/V FOV constraints
    r = max(extents.x, extents.y, extents.z) * 0.5
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) / max(aspect, 1e-6))
    dist_h = r / math.tan(hfov / 2.0)
    dist_v = r / math.tan(vfov / 2.0)
    return max(dist_h, dist_v) * margin


def create_camera(center, distance, direction=Vector((0.5, 0.5, 0.8)), hfov_deg=45.0, aspect=1.0):
    direction = direction.normalized()
    cam_loc = center + direction * distance

    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    # Set horizontal FOV
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(hfov_deg)  # treated as horizontal FOV

    # Place camera at computed location BEFORE aiming it
    cam_obj.location = cam_loc

    # Look at center
    look_at(cam_obj, center)

    # Set sensor fit to horizontal to honor angle as horizontal FOV
    cam_data.sensor_fit = 'HORIZONTAL'

    # Adjust render resolution-dependent FOV
    bpy.context.scene.camera = cam_obj
    return cam_obj


def look_at(obj, target: Vector, up_override: Vector = None):
    loc = Vector(obj.location)
    forward = (target - loc).normalized()
    up = Vector((0.0, 0.0, 1.0)) if up_override is None else up_override.normalized()
    if abs(forward.dot(up)) > 0.999:
        up = Vector((0.0, 1.0, 0.0))

    right = forward.cross(up).normalized()
    up_corr = right.cross(forward).normalized()

    rot = Matrix((
        (right.x, up_corr.x, -forward.x, 0.0),
        (right.y, up_corr.y, -forward.y, 0.0),
        (right.z, up_corr.z, -forward.z, 0.0),
        (0.0,     0.0,      0.0,         1.0),
    ))
    obj.matrix_world = Matrix.Translation(loc) @ rot


def add_lights(center, distance, camera_obj):
    # Key light aligned with camera
    def add_sun(name, loc, energy):
        sun = bpy.data.lights.new(name=name, type='SUN')
        sun.energy = energy
        sun_obj = bpy.data.objects.new(name, sun)
        bpy.context.scene.collection.objects.link(sun_obj)
        sun_obj.location = loc
        look_at(sun_obj, center)
        return sun_obj

    cam_dir = (center - camera_obj.location).normalized()
    key_pos = center - cam_dir * (distance * 0.5)
    add_sun("KeyLight", key_pos, energy=3.0)
    # Fill
    fill_pos = center + Vector((-distance * 0.5, -distance * 0.3, distance * 0.3))
    add_sun("FillLight", fill_pos, energy=1.2)
    # Rim
    rim_pos = center + Vector((0.0, 0.0, -distance))
    add_sun("RimLight", rim_pos, energy=0.8)


def render_view(scene, camera_obj, filepath: Path):
    scene.camera = camera_obj
    scene.render.filepath = str(filepath)
    bpy.ops.render.render(write_still=True)


def ensure_compositor(scene):
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()
    return tree, nodes, links


def composite_grid(scene, img_paths, view_width: int, view_height: int, out_path: Path):
    # img_paths: dict with keys 'front','side','top','quarter'
    tree, nodes, links = ensure_compositor(scene)

    # Background color
    rgb_bg = nodes.new('CompositorNodeRGB')
    # Use index-based access to be robust across Blender versions
    rgb_bg.outputs[0].default_value = (0.16, 0.16, 0.16, 1.0)

    # Load images
    images = {}
    img_nodes = {}
    for name, p in img_paths.items():
        images[name] = bpy.data.images.load(str(p))
        node = nodes.new('CompositorNodeImage')
        node.image = images[name]
        img_nodes[name] = node

    # Translate nodes to place into 2x2 grid
    def translate(node, x, y):
        t = nodes.new('CompositorNodeTranslate')
        t.inputs['X'].default_value = float(x)
        t.inputs['Y'].default_value = float(y)
        links.new(node.outputs[0], t.inputs[0])
        return t

    tr_front = translate(img_nodes['front'], 0, view_height)
    tr_side = translate(img_nodes['side'], view_width, view_height)
    tr_top = translate(img_nodes['top'], 0, 0)
    tr_quarter = translate(img_nodes['quarter'], view_width, 0)

    # Chain alpha over nodes: ((((bg over front) over side) over top) over quarter)
    ao1 = nodes.new('CompositorNodeAlphaOver')
    ao1.inputs['Fac'].default_value = 1.0
    links.new(rgb_bg.outputs[0], ao1.inputs[1])  # Background
    links.new(tr_front.outputs[0], ao1.inputs[2])

    ao2 = nodes.new('CompositorNodeAlphaOver')
    ao2.inputs['Fac'].default_value = 1.0
    links.new(ao1.outputs[0], ao2.inputs[1])
    links.new(tr_side.outputs[0], ao2.inputs[2])

    ao3 = nodes.new('CompositorNodeAlphaOver')
    ao3.inputs['Fac'].default_value = 1.0
    links.new(ao2.outputs[0], ao3.inputs[1])
    links.new(tr_top.outputs[0], ao3.inputs[2])

    ao4 = nodes.new('CompositorNodeAlphaOver')
    ao4.inputs['Fac'].default_value = 1.0
    links.new(ao3.outputs[0], ao4.inputs[1])
    links.new(tr_quarter.outputs[0], ao4.inputs[2])

    comp = nodes.new('CompositorNodeComposite')
    links.new(ao4.outputs[0], comp.inputs[0])

    # Set final resolution and output filepath
    scene.render.resolution_x = view_width * 2
    scene.render.resolution_y = view_height * 2
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    glb_path = Path(args.glb_file)
    out_path = Path(args.output) if args.output else glb_path.with_suffix(".png")

    scene = reset_scene()
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(out_path)

    print(f"Loading: {glb_path}")
    objs = import_glb(glb_path)

    # Compute evaluated bounds (respect armature skinning)
    bounds, center, extents = evaluated_bounds_worldspace(objs)
    print(f"Center: {tuple(round(c, 4) for c in center)}  Extents: {tuple(round(e, 4) for e in extents)}")

    aspect = args.width / max(args.height, 1)
    distance = compute_camera_distance(extents, hfov_deg=args.fov, aspect=aspect, margin=args.margin)

    if args.single_view:
        cam = create_camera(center, distance, hfov_deg=args.fov, aspect=aspect)
        add_lights(center, distance, cam)
        print(f"Rendering to: {out_path}")
        bpy.ops.render.render(write_still=True)
        print("Done.")
        return

    # Multi-view: create cameras for four views
    cam_front = create_camera(center, distance, direction=Vector((0.0, -1.0, 0.0)), hfov_deg=args.fov, aspect=aspect)
    cam_side = create_camera(center, distance, direction=Vector((1.0, 0.0, 0.0)), hfov_deg=args.fov, aspect=aspect)
    cam_top = create_camera(center, distance, direction=Vector((0.0, 0.0, 1.0)), hfov_deg=args.fov, aspect=aspect)
    cam_quarter = create_camera(center, distance, direction=Vector((0.5, 0.5, 0.8)), hfov_deg=args.fov, aspect=aspect)

    # Add lights once (aligned with quarter view for pleasing results)
    add_lights(center, distance, cam_quarter)

    # Render each view to temp files
    tmp_dir = out_path.parent
    tmp_paths = {
        'front': tmp_dir / f"{out_path.stem}_front.png",
        'side': tmp_dir / f"{out_path.stem}_side.png",
        'top': tmp_dir / f"{out_path.stem}_top.png",
        'quarter': tmp_dir / f"{out_path.stem}_quarter.png",
    }

    # Ensure per-view resolution
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height

    print("Rendering views...")
    render_view(scene, cam_front, tmp_paths['front'])
    render_view(scene, cam_side, tmp_paths['side'])
    render_view(scene, cam_top, tmp_paths['top'])
    render_view(scene, cam_quarter, tmp_paths['quarter'])

    if args.separate_views:
        print("Saved separate views:")
        for name, p in tmp_paths.items():
            print(f"  {name}: {p}")
        print("Done.")
        return

    print("Compositing grid...")
    composite_grid(scene, tmp_paths, args.width, args.height, out_path)

    # Cleanup temp images
    for p in tmp_paths.values():
        try:
            if Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass
    print("Done.")


if __name__ == "__main__":
    main()