#!/usr/bin/env python3
"""
CLI-only script to render GLB files to PNG screenshots.
Uses pyrender for headless rendering without any GUI windows.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont


def render_glb_multiview(glb_file, output_file=None, view_width=400, view_height=300):
    """
    Convenient function to render a GLB file to a multi-view PNG screenshot.
    Designed for programmatic use from other scripts.
    
    Args:
        glb_file: Path to GLB file (str or Path)
        output_file: Output PNG path (str or Path). If None, uses GLB name with .png
        view_width: Width of each individual view (default: 400)
        view_height: Height of each individual view (default: 300)
        
    Returns:
        bool: True if successful, False otherwise
    """
    glb_path = Path(glb_file)
    
    if output_file is None:
        output_path = glb_path.with_suffix('.png')
    else:
        output_path = Path(output_file)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering multi-view: {glb_path} -> {output_path}")
    
    return create_multiview_grid(
        glb_path=glb_path,
        output_path=output_path,
        view_width=view_width,
        view_height=view_height
    )


def render_glb_single(glb_file, output_file=None, width=800, height=600):
    """
    Convenient function to render a GLB file to a single-view PNG screenshot.
    Designed for programmatic use from other scripts.
    
    Args:
        glb_file: Path to GLB file (str or Path)
        output_file: Output PNG path (str or Path). If None, uses GLB name with .png
        width: Image width (default: 800)
        height: Image height (default: 600)
        
    Returns:
        bool: True if successful, False otherwise
    """
    glb_path = Path(glb_file)
    
    if output_file is None:
        output_path = glb_path.with_suffix('.png')
    else:
        output_path = Path(output_file)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering single view: {glb_path} -> {output_path}")
    
    return render_glb_to_png(
        glb_path=glb_path,
        output_path=output_path,
        width=width,
        height=height
    )


def render_glb_to_png(glb_path: Path, output_path: Path, width: int = 800, height: int = 600):
    """
    Render a GLB file to a PNG screenshot using headless rendering.
    
    Args:
        glb_path: Path to the GLB file
        output_path: Path for the output PNG file
        width: Image width in pixels
        height: Image height in pixels
    """
    print(f"Loading GLB file: {glb_path}")
    
    # Load the GLB file using trimesh
    try:
        scene = trimesh.load(str(glb_path))
    except Exception as e:
        print(f"Error loading GLB file: {e}")
        return False
    
    print(f"Loaded scene with {len(scene.geometry)} geometries")
    
    # Get scene bounds to set up camera
    scene_bounds = scene.bounds
    scene_center = scene_bounds.mean(axis=0)
    scene_scale = np.linalg.norm(scene_bounds[1] - scene_bounds[0])
    
    print(f"Scene center: {scene_center}")
    print(f"Scene scale: {scene_scale:.2f}")
    print(f"Scene bounds: {scene_bounds}")
    
    # Create pyrender scene manually to have more control
    pyrender_scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    
    # Add all meshes to the scene with materials
    for name, geom in scene.geometry.items():
        if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
            # Create material for better visibility
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                metallicFactor=0.1,
                roughnessFactor=0.8
            )
            
            # Create pyrender mesh
            mesh = pyrender.Mesh.from_trimesh(geom, material=material)
            pyrender_scene.add(mesh)
            print(f"Added mesh: {name}")
    
    # Set up camera - position it to view the entire model
    camera_distance = scene_scale * 2.0  # Move camera further back
    
    # Position camera at an angle to see the model better
    camera_pos = scene_center + np.array([camera_distance * 0.5, camera_distance * 0.5, camera_distance * 0.8])
    
    # Create camera transformation matrix (look at the center)
    # Camera looks toward scene center from camera_pos
    view_direction = scene_center - camera_pos
    view_direction = view_direction / np.linalg.norm(view_direction)
    
    # Create right and up vectors
    up = np.array([0, 0, 1])  # Z-up
    right = np.cross(view_direction, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, view_direction)
    
    # Create camera pose matrix
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -view_direction  # Negative because camera looks down -Z
    camera_pose[:3, 3] = camera_pos
    
    print(f"Camera position: {camera_pos}")
    print(f"Looking at: {scene_center}")
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=width/height)
    pyrender_scene.add(camera, pose=camera_pose)
    
    # Add multiple lights for better illumination
    # Key light
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    key_light_pose = camera_pose.copy()
    pyrender_scene.add(key_light, pose=key_light_pose)
    
    # Fill light from opposite side
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    fill_light_pose = camera_pose.copy()
    fill_light_pose[:3, 3] = scene_center + np.array([-camera_distance * 0.3, -camera_distance * 0.3, camera_distance * 0.5])
    pyrender_scene.add(fill_light, pose=fill_light_pose)
    
    # Rim light from behind
    rim_light = pyrender.DirectionalLight(color=np.ones(3), intensity=0.5)
    rim_light_pose = camera_pose.copy()
    rim_light_pose[:3, 3] = scene_center + np.array([0, 0, -camera_distance])
    pyrender_scene.add(rim_light, pose=rim_light_pose)
    
    print("Setting up headless renderer...")
    
    # Create headless renderer (no GUI)
    try:
        renderer = pyrender.OffscreenRenderer(width, height)
    except Exception as e:
        print(f"Error creating renderer: {e}")
        print("Make sure you have proper OpenGL support for headless rendering")
        return False
    
    print("Rendering scene...")
    
    try:
        # Render the scene
        color, depth = renderer.render(pyrender_scene)
        
        print(f"Rendered color array shape: {color.shape}")
        print(f"Color array min/max: {color.min()}/{color.max()}")
        
        # Check if we got any non-zero pixels
        non_zero_pixels = np.count_nonzero(color)
        print(f"Non-zero pixels: {non_zero_pixels}/{color.size}")
        
        # Convert to PIL Image and save
        image = Image.fromarray(color)
        image.save(output_path)
        
        print(f"✅ Screenshot saved to: {output_path}")
        print(f"   Image size: {width}x{height}")
        
        return True
        
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        renderer.delete()


def get_camera_poses(scene_center, scene_scale):
    """
    Generate camera poses for different views: front, side, top, and 3/4 view.
    
    Args:
        scene_center: Center point of the scene
        scene_scale: Scale of the scene for camera distance
        
    Returns:
        dict: Dictionary of camera poses with view names as keys
    """
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
    """
    Create a camera transformation matrix that looks from camera_pos to target_pos.
    
    Args:
        camera_pos: Camera position
        target_pos: Point to look at
        up_vector: Up direction
        
    Returns:
        4x4 transformation matrix
    """
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
    """
    Render a single view of the scene with given camera pose.
    
    Args:
        scene: Loaded trimesh scene
        camera_pose: 4x4 camera transformation matrix
        width: Image width
        height: Image height
        
    Returns:
        PIL Image of the rendered view
    """
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


def create_multiview_grid(glb_path: Path, output_path: Path, view_width: int = 400, view_height: int = 300):
    """
    Create a 2x2 grid of different camera views of the GLB file.
    
    Args:
        glb_path: Path to the GLB file
        output_path: Path for the output PNG file
        view_width: Width of each individual view
        view_height: Height of each individual view
    """
    print(f"Creating multi-view grid from: {glb_path}")
    
    # Load the GLB file
    try:
        scene = trimesh.load(str(glb_path))
    except Exception as e:
        print(f"Error loading GLB file: {e}")
        return False
    
    print(f"Loaded scene with {len(scene.geometry)} geometries")
    
    # Get scene bounds for camera positioning
    scene_bounds = scene.bounds
    scene_center = scene_bounds.mean(axis=0)
    scene_scale = np.linalg.norm(scene_bounds[1] - scene_bounds[0])
    
    print(f"Scene center: {scene_center}")
    print(f"Scene scale: {scene_scale:.2f}")
    
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
    
    print(f"✅ Multi-view grid saved to: {output_path}")
    print(f"   Grid size: {grid_width}x{grid_height} ({view_width}x{view_height} per view)")
    
    return True


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Render GLB files to PNG screenshots using headless rendering"
    )
    parser.add_argument(
        "glb_file", 
        type=str,
        help="Path to the GLB file to render"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output PNG file path (default: same name as GLB with .png extension)"
    )
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=800,
        help="Image width in pixels (default: 800, or 400 per view for multi-view)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Image height in pixels (default: 600, or 300 per view for multi-view)"
    )
    parser.add_argument(
        "--single-view",
        action="store_true",
        help="Render only a single 3/4 view instead of the default 2x2 multi-view grid"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    glb_path = Path(args.glb_file)
    if not glb_path.exists():
        print(f"Error: GLB file not found: {glb_path}")
        return 1
    
    if not glb_path.suffix.lower() == '.glb':
        print(f"Error: File must have .glb extension: {glb_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = glb_path.with_suffix('.png')
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("GLB to PNG Screenshot Renderer")
    print("=" * 40)
    print(f"Input:  {glb_path}")
    print(f"Output: {output_path}")
    if args.single_view:
        print(f"Mode:   Single view")
        print(f"Size:   {args.width}x{args.height}")
    else:
        print(f"Mode:   Multi-view (2x2 grid)")
        print(f"Size:   {args.width * 2}x{args.height * 2} ({args.width}x{args.height} per view)")
    print("=" * 40)
    
    # Render the screenshot(s)
    if args.single_view:
        success = render_glb_to_png(
            glb_path=glb_path,
            output_path=output_path,
            width=args.width,
            height=args.height
        )
    else:
        success = create_multiview_grid(
            glb_path=glb_path,
            output_path=output_path,
            view_width=args.width,
            view_height=args.height
        )
    
    if success:
        print("\n🎉 Screenshot rendered successfully!")
        return 0
    else:
        print("\n❌ Failed to render screenshot")
        return 1


if __name__ == "__main__":
    sys.exit(main())
