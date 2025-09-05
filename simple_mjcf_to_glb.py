#!/usr/bin/env python3
"""
Bare minimum script to convert MJCF to GLB with embedded functions.
"""

import sys
import trimesh
import numpy as np
import pyrender
from pathlib import Path
from typing import Union
from PIL import Image, ImageDraw, ImageFont
from armature_exporter.mjcf_parser import MJCFParser


def load_stl(stl_path: Union[str, Path]) -> trimesh.Trimesh:
    """
    Load an STL file and return a trimesh.Trimesh object.
    """
    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    print(f"Loading STL file: {stl_path}")
    mesh = trimesh.load(str(stl_path))
    
    # Ensure it's a mesh (not a scene)
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, get the first mesh
        if len(mesh.geometry) == 0:
            raise ValueError(f"No geometry found in STL file: {stl_path}")
        mesh = list(mesh.geometry.values())[0]
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a valid mesh: {type(mesh)}")
    
    return mesh


def export_scene_to_glb(scene: trimesh.Scene, output_path: Union[str, Path]) -> None:
    """
    Export a scene to GLB format.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to GLB
    exported = scene.export(file_type='glb')
    
    with open(output_path, 'wb') as f:
        f.write(exported)
    
    print(f"Exported scene to: {output_path}")


def get_camera_poses(scene_center, scene_scale):
    """Generate camera poses for different views: front, side, top, and 3/4 view."""
    camera_distance = scene_scale * 2.0
    poses = {}
    
    # Front view (looking from front, Y- direction)
    front_pos = scene_center + np.array([0, -camera_distance, 0])
    poses['front'] = create_lookat_matrix(front_pos, scene_center, np.array([0, 0, 1]))
    
    # Side view (looking from right side, X+ direction)  
    side_pos = scene_center + np.array([camera_distance, 0, 0])
    poses['side'] = create_lookat_matrix(side_pos, scene_center, np.array([0, 0, 1]))
    
    # Top view (looking from above, Z+ direction)
    top_pos = scene_center + np.array([0, 0, camera_distance])
    poses['top'] = create_lookat_matrix(top_pos, scene_center, np.array([0, 1, 0]))
    
    # 3/4 view (current angled view)
    quarter_pos = scene_center + np.array([camera_distance * 0.5, camera_distance * 0.5, camera_distance * 0.8])
    poses['quarter'] = create_lookat_matrix(quarter_pos, scene_center, np.array([0, 0, 1]))
    
    return poses


def create_lookat_matrix(camera_pos, target_pos, up_vector):
    """Create a camera transformation matrix that looks from camera_pos to target_pos."""
    # Calculate view direction
    view_direction = target_pos - camera_pos
    view_direction = view_direction / np.linalg.norm(view_direction)
    
    # Create right and up vectors
    right = np.cross(view_direction, up_vector)
    if np.linalg.norm(right) < 1e-6:  # Handle case where view_direction is parallel to up_vector
        # Use a different up vector
        alternate_up = np.array([1, 0, 0]) if abs(up_vector[0]) < 0.9 else np.array([0, 1, 0])
        right = np.cross(view_direction, alternate_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, view_direction)
    
    # Create camera pose matrix
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -view_direction  # Negative because camera looks down -Z
    pose[:3, 3] = camera_pos
    
    return pose


def render_single_view(scene, camera_pose, width, height):
    """Render a single view of the scene with given camera pose."""
    # Create pyrender scene
    pyrender_scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    
    # Add all meshes to the scene with materials
    for name, geom in scene.geometry.items():
        if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                metallicFactor=0.1,
                roughnessFactor=0.8
            )
            mesh = pyrender.Mesh.from_trimesh(geom, material=material)
            pyrender_scene.add(mesh)
    
    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=width/height)
    pyrender_scene.add(camera, pose=camera_pose)
    
    # Add lighting
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    pyrender_scene.add(key_light, pose=camera_pose)
    
    # Fill light from different angle
    fill_light_pose = camera_pose.copy()
    fill_light_pose[:3, 3] += np.array([0.5, 0.5, 0.5])  # Offset the light position
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    pyrender_scene.add(fill_light, pose=fill_light_pose)
    
    # Render
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = renderer.render(pyrender_scene)
        return Image.fromarray(color)
    finally:
        renderer.delete()


def create_multiview_screenshot(glb_path: Union[str, Path], output_path: Union[str, Path], view_width: int = 400, view_height: int = 300):
    """Create a 2x2 grid of different camera views of the GLB file."""
    glb_path = Path(glb_path)
    output_path = Path(output_path)
    
    print(f"Creating multiview screenshot: {glb_path} -> {output_path}")
    
    # Load the GLB file
    try:
        scene = trimesh.load(str(glb_path))
    except Exception as e:
        print(f"Error loading GLB file: {e}")
        return False
    
    # Get scene bounds for camera positioning
    scene_bounds = scene.bounds
    scene_center = scene_bounds.mean(axis=0)
    scene_scale = np.linalg.norm(scene_bounds[1] - scene_bounds[0])
    
    # Get camera poses for different views
    camera_poses = get_camera_poses(scene_center, scene_scale)
    
    # Render each view
    views = {}
    view_names = ['front', 'side', 'top', 'quarter']
    view_labels = ['Front', 'Side', 'Top', '3/4 View']
    
    print("Rendering views...")
    for name, label in zip(view_names, view_labels):
        print(f"  Rendering {label}...")
        views[name] = render_single_view(scene, camera_poses[name], view_width, view_height)
    
    # Create 2x2 grid
    grid_width = view_width * 2
    grid_height = view_height * 2
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(40, 40, 40))
    
    # Paste views into grid
    # Top row: Front, Side
    grid_image.paste(views['front'], (0, 0))
    grid_image.paste(views['side'], (view_width, 0))
    
    # Bottom row: Top, 3/4 View  
    grid_image.paste(views['top'], (0, view_height))
    grid_image.paste(views['quarter'], (view_width, view_height))
    
    # Add labels
    draw = ImageDraw.Draw(grid_image)
    try:
        # Try to use a nice font, fall back to default if not available
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    label_positions = [
        (10, 10, 'Front'),
        (view_width + 10, 10, 'Side'),
        (10, view_height + 10, 'Top'),
        (view_width + 10, view_height + 10, '3/4 View')
    ]
    
    for x, y, label in label_positions:
        # Draw text with outline for visibility
        draw.text((x-1, y-1), label, font=font, fill=(0, 0, 0))
        draw.text((x+1, y-1), label, font=font, fill=(0, 0, 0))
        draw.text((x-1, y+1), label, font=font, fill=(0, 0, 0))
        draw.text((x+1, y+1), label, font=font, fill=(0, 0, 0))
        draw.text((x, y), label, font=font, fill=(255, 255, 255))
    
    # Save the grid
    grid_image.save(output_path)
    
    print(f"✅ Multiview screenshot saved: {output_path}")
    
    return True


def mjcf_to_glb(mjcf_path: Union[str, Path], output_path: Union[str, Path], 
                simplify: bool = True, max_faces_per_mesh: int = 5000) -> trimesh.Scene:
    """
    Export MJCF robot model to GLB format (meshes only, no rigging).
    """
    print(f"Converting MJCF to GLB: {mjcf_path} -> {output_path}")
    
    # Parse MJCF
    parser = MJCFParser(mjcf_path)
    mesh_transforms = parser.get_mesh_transforms()
    
    print(f"Found {len(mesh_transforms)} unique meshes")
    
    # Create scene
    scene = trimesh.Scene()
    processed_meshes = {}
    
    # Process each mesh
    for mesh_name, transforms in mesh_transforms.items():
        print(f"  Processing {mesh_name}...")
        
        # Get mesh file path
        mesh_file_path = parser.get_mesh_file_path(mesh_name)
        if mesh_file_path is None or not mesh_file_path.exists():
            print(f"    Warning: Skipping missing mesh {mesh_name}")
            continue
        
        # Load mesh (cache it if we haven't loaded it before)
        if mesh_name not in processed_meshes:
            try:
                mesh = load_stl(mesh_file_path)
                
                # Simplify mesh if requested
                if simplify and len(mesh.faces) > max_faces_per_mesh:
                    print(f"    Simplifying {mesh_name}: {len(mesh.faces)} -> {max_faces_per_mesh} faces")
                    mesh = mesh.simplify_quadric_decimation(face_count=max_faces_per_mesh)
                
                processed_meshes[mesh_name] = mesh
            except Exception as e:
                print(f"    Error loading {mesh_name}: {e}")
                continue
        
        base_mesh = processed_meshes[mesh_name]
        
        # Add transformed instances to scene
        for i, (body_name, position, rotation_matrix, material) in enumerate(transforms):
            # Apply transform
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = position
            
            transformed_mesh = base_mesh.copy()
            transformed_mesh.apply_transform(transform_matrix)
            
            instance_name = f"{body_name}_{mesh_name}"
            if len(transforms) > 1:
                instance_name += f"_{i}"
            scene.add_geometry(transformed_mesh, node_name=instance_name)
    
    # Export to GLB
    export_scene_to_glb(scene, output_path)
    
    print(f"✅ MJCF exported to GLB: {output_path}")
    return scene


def main():
    # Default to mvp_test.xml in output directory
    mjcf_file = "g1_description/mvp_test.xml"
    output_file = "output/mvp_simple.glb"
    
    # Override with command line args if provided
    if len(sys.argv) > 1:
        mjcf_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        # If no path separator in output_file, put it in output dir
        if "/" not in output_file and "\\" not in output_file:
            output_file = f"output/{output_file}"
    
    # Convert MJCF to GLB
    mjcf_to_glb(mjcf_file, output_file)
    
    # Create multiview screenshot in same directory as GLB
    screenshot_path = Path(output_file).with_suffix('.png')
    print(f"\nGenerating multiview screenshot...")
    try:
        create_multiview_screenshot(output_file, screenshot_path)
    except Exception as e:
        print(f"Warning: Could not create screenshot: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
