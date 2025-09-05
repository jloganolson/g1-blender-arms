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


def get_scene_bounds_and_center(scene):
    """
    Get the actual bounding box and center of all visible geometries in the scene.
    This ensures we capture the true extent of the model, not just the scene bounds.
    
    Args:
        scene: Loaded trimesh scene
        
    Returns:
        tuple: (bounds, center, extents) where:
            - bounds: [min_point, max_point] array
            - center: center point of the actual geometry
            - extents: size in each dimension [x, y, z]
    """
    all_vertices = []
    
    # Collect all vertices from all geometries
    for name, geom in scene.geometry.items():
        if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
            # Apply any transforms from the scene graph
            vertices = geom.vertices
            if name in scene.graph.nodes:
                # Get the transform for this geometry
                transform, _ = scene.graph.get(name)
                if transform is not None:
                    # Apply transform to vertices
                    homogeneous_verts = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
                    transformed_verts = (transform @ homogeneous_verts.T).T[:, :3]
                    vertices = transformed_verts
            all_vertices.append(vertices)
    
    if not all_vertices:
        # Fallback to scene bounds if no vertices found
        print("Warning: No vertices found in geometries, using scene bounds")
        scene_bounds = scene.bounds
        scene_center = scene_bounds.mean(axis=0)
        scene_extents = scene_bounds[1] - scene_bounds[0]
        return scene_bounds, scene_center, scene_extents
    
    # Combine all vertices
    combined_vertices = np.vstack(all_vertices)
    
    # Calculate true bounds
    min_point = np.min(combined_vertices, axis=0)
    max_point = np.max(combined_vertices, axis=0)
    bounds = np.array([min_point, max_point])
    
    # Calculate center and extents
    center = (min_point + max_point) / 2.0
    extents = max_point - min_point
    
    print(f"Analyzed {len(all_vertices)} geometry groups with {combined_vertices.shape[0]} total vertices")
    
    return bounds, center, extents


def calculate_optimal_camera_distance(extents, yfov, aspect, margin_factor=1.3):
    """
    Calculate the optimal camera distance to fit the model in frame.
    Uses both horizontal and vertical FOV to ensure the model fits regardless of aspect ratio.
    
    Args:
        extents: Model extents in [x, y, z]
        yfov: Vertical field of view in radians
        aspect: Width/height ratio
        margin_factor: Extra margin around the model (default: 1.3 = 30% margin)
        
    Returns:
        float: Optimal camera distance
    """
    # For a perspective camera, we need to fit the model in both dimensions
    # The model's "radius" in the camera's view depends on the viewing direction
    # We'll use the maximum extent in any dimension as a conservative estimate
    
    # Calculate distances needed for vertical and horizontal FOV
    max_extent_y = max(extents[1], extents[2])  # Y or Z extent for vertical FOV
    max_extent_x = max(extents[0], extents[2])  # X or Z extent for horizontal FOV
    
    # Distance needed to fit vertically
    dist_vertical = (max_extent_y / 2.0) / np.tan(yfov / 2.0)
    
    # Distance needed to fit horizontally (calculate horizontal FOV)
    hfov = 2.0 * np.arctan(np.tan(yfov / 2.0) * aspect)
    dist_horizontal = (max_extent_x / 2.0) / np.tan(hfov / 2.0)
    
    # Use the larger distance to ensure model fits in both dimensions
    optimal_distance = max(dist_vertical, dist_horizontal) * margin_factor
    
    print(f"Camera distance calculation:")
    print(f"  Extents: {extents}")
    print(f"  Vertical FOV: {np.degrees(yfov):.1f}°, Horizontal FOV: {np.degrees(hfov):.1f}°")
    print(f"  Required distances: vertical={dist_vertical:.2f}, horizontal={dist_horizontal:.2f}")
    print(f"  Optimal distance with {margin_factor}x margin: {optimal_distance:.2f}")
    
    return optimal_distance


def render_glb_multiview(glb_file, output_file=None, view_width=512, view_height=512):
    """
    Convenient function to render a GLB file to a multi-view PNG screenshot.
    Designed for programmatic use from other scripts.
    
    Args:
        glb_file: Path to GLB file (str or Path)
        output_file: Output PNG path (str or Path). If None, uses GLB name with .png
        view_width: Width of each individual view (default: 512)
        view_height: Height of each individual view (default: 512)
        
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


def render_glb_single(glb_file, output_file=None, width=512, height=512):
    """
    Convenient function to render a GLB file to a single-view PNG screenshot.
    Designed for programmatic use from other scripts.
    
    Args:
        glb_file: Path to GLB file (str or Path)
        output_file: Output PNG path (str or Path). If None, uses GLB name with .png
        width: Image width (default: 512)
        height: Image height (default: 512)
        
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


def render_glb_to_png(glb_path: Path, output_path: Path, width: int = 512, height: int = 512):
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
    
    # Get scene bounds and calculate proper centering
    scene_bounds, scene_center, scene_extents = get_scene_bounds_and_center(scene)
    
    print(f"Scene center: {scene_center}")
    print(f"Scene extents: {scene_extents}")
    print(f"Scene bounds: {scene_bounds}")
    
    # Build pyrender scene from the trimesh scene to preserve node transforms
    pyrender_scene = pyrender.Scene.from_trimesh_scene(scene, ambient_light=[0.1, 0.1, 0.1])
    
    # Compute optimal camera distance that fits the model in frame
    yfov = np.pi / 4.0
    aspect = width / height
    camera_distance = calculate_optimal_camera_distance(scene_extents, yfov, aspect)
    
    # Position camera at an angle to see the model better
    view_dir = np.array([0.5, 0.5, 0.8])
    view_dir = view_dir / np.linalg.norm(view_dir)
    camera_pos = scene_center + view_dir * camera_distance
    
    # Create camera pose matrix using a look-at helper
    camera_pose = create_lookat_matrix(camera_pos, scene_center, np.array([0, 0, 1]))
    
    print(f"Camera position: {camera_pos}")
    print(f"Looking at: {scene_center}")
    
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect)
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


def get_camera_poses(scene_center, scene_extents, yfov=np.pi/4.0, aspect=1.0):
    """
    Generate camera poses for different views: front, side, top, and 3/4 view.
    
    Args:
        scene_center: Center point of the scene
        scene_extents: Scene extents in [x, y, z]
        yfov: Vertical field of view for camera distance calculation
        aspect: Aspect ratio for camera distance calculation
        
    Returns:
        dict: Dictionary of camera poses with view names as keys
    """
    camera_distance = calculate_optimal_camera_distance(scene_extents, yfov, aspect)
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
    
    # 3/4 view (angled)
    quarter_dir = np.array([0.5, 0.5, 0.8])
    quarter_dir = quarter_dir / np.linalg.norm(quarter_dir)
    quarter_pos = scene_center + quarter_dir * camera_distance
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


def create_multiview_grid(glb_path: Path, output_path: Path, view_width: int = 512, view_height: int = 512):
    """
    Create a 2x2 grid of different camera views of the GLB file.
    
    Args:
        glb_path: Path to the GLB file
        output_path: Path for the output PNG file
        view_width: Width of each individual view (default: 512)
        view_height: Height of each individual view (default: 512)
    """
    print(f"Creating multi-view grid from: {glb_path}")
    
    # Load the GLB file
    try:
        scene = trimesh.load(str(glb_path))
    except Exception as e:
        print(f"Error loading GLB file: {e}")
        return False
    
    print(f"Loaded scene with {len(scene.geometry)} geometries")
    
    # Get scene bounds and calculate proper centering
    scene_bounds, scene_center, scene_extents = get_scene_bounds_and_center(scene)
    
    print(f"Scene center: {scene_center}")
    print(f"Scene extents: {scene_extents}")
    print(f"Scene bounds: {scene_bounds}")
    
    # Get camera poses for different views
    aspect = view_width / view_height
    camera_poses = get_camera_poses(scene_center, scene_extents, yfov=np.pi/4.0, aspect=aspect)
    
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
        default=512,
        help="Image width in pixels (default: 512, or 512 per view for multi-view)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels (default: 512, or 512 per view for multi-view)"
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
