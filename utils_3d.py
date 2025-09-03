#!/usr/bin/env python3
"""
3D Mesh Utilities
This module provides utilities for working with 3D meshes, including simplification,
combining multiple meshes, and exporting to various formats.
"""

import trimesh
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: Optional[int] = None, reduction_ratio: float = 0.5) -> trimesh.Trimesh:
    """
    Simplify a mesh by reducing the number of faces.
    
    Args:
        mesh: trimesh.Trimesh object
        target_faces: Target number of faces (optional)
        reduction_ratio: Ratio to reduce faces by (default 0.5 = 50% reduction)
    
    Returns:
        Simplified trimesh.Trimesh object
    """
    print(f"Original mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    
    if target_faces is not None:
        # Calculate reduction ratio from target faces
        reduction_ratio = target_faces / len(mesh.faces)
        if reduction_ratio > 1.0:
            reduction_ratio = 1.0
            print(f"Warning: Target faces ({target_faces}) is greater than original faces ({len(mesh.faces)}). Using original mesh.")
        elif reduction_ratio < 0.01:
            reduction_ratio = 0.01
            print(f"Warning: Target faces too low. Using minimum reduction ratio of 0.01")
    
    print(f"Reduction ratio: {reduction_ratio:.3f}")
    target_faces_calc = int(len(mesh.faces) * reduction_ratio)
    print(f"Target faces: {target_faces_calc}")
    
    # Use trimesh's built-in simplification with face count
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces_calc)
    
    print(f"Simplified mesh: {len(simplified.faces)} faces, {len(simplified.vertices)} vertices")
    
    return simplified


def load_stl(stl_path: Union[str, Path]) -> trimesh.Trimesh:
    """
    Load an STL file and return a trimesh.Trimesh object.
    
    Args:
        stl_path: Path to the STL file
        
    Returns:
        trimesh.Trimesh object
        
    Raises:
        FileNotFoundError: If the STL file doesn't exist
        ValueError: If the file cannot be loaded as a mesh
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


def combine_meshes(meshes: List[trimesh.Trimesh], mesh_names: Optional[List[str]] = None) -> trimesh.Scene:
    """
    Combine multiple meshes into a single scene.
    
    Args:
        meshes: List of trimesh.Trimesh objects
        mesh_names: Optional list of names for the meshes
        
    Returns:
        trimesh.Scene containing all meshes
    """
    if not meshes:
        raise ValueError("No meshes provided")
    
    if mesh_names is None:
        mesh_names = [f"mesh_{i}" for i in range(len(meshes))]
    elif len(mesh_names) != len(meshes):
        raise ValueError("Number of mesh names must match number of meshes")
    
    print(f"Combining {len(meshes)} meshes into a single scene")
    
    # Create a scene and add all meshes
    scene = trimesh.Scene()
    
    for i, (mesh, name) in enumerate(zip(meshes, mesh_names)):
        print(f"  Adding mesh {i+1}: {name} ({len(mesh.faces)} faces, {len(mesh.vertices)} vertices)")
        scene.add_geometry(mesh, geom_name=name)
    
    return scene


def export_scene_to_glb(scene: trimesh.Scene, output_path: Union[str, Path]) -> None:
    """
    Export a scene to GLB format.
    
    Args:
        scene: trimesh.Scene object
        output_path: Path to save GLB file
    """
    output_path = Path(output_path)
    
    # Export to GLB
    exported = scene.export(file_type='glb')
    
    with open(output_path, 'wb') as f:
        f.write(exported)
    
    print(f"Exported scene to: {output_path}")


def export_mesh_to_glb(mesh: trimesh.Trimesh, output_path: Union[str, Path]) -> None:
    """
    Export a single mesh to GLB format.
    
    Args:
        mesh: trimesh.Trimesh object
        output_path: Path to save GLB file
    """
    # Create a scene with the mesh
    scene = trimesh.Scene([mesh])
    export_scene_to_glb(scene, output_path)


def render_mesh(mesh: trimesh.Trimesh, title: str = "Mesh Preview", save_path: Optional[Union[str, Path]] = None, resolution: Tuple[int, int] = (1200, 600)) -> None:
    """
    Render a mesh using trimesh's built-in visualization.
    
    Args:
        mesh: trimesh.Trimesh object
        title: Title for the plot (used in filename if save_path not provided)
        save_path: Path to save the plot (optional)
        resolution: Image resolution as (width, height) tuple
    """
    try:
        # Create a scene with the mesh
        scene = trimesh.Scene([mesh])
        
        # Generate the image using trimesh's built-in renderer
        image = scene.save_image(resolution=resolution)
        
        # Determine save path
        if save_path:
            save_path = Path(save_path)
        else:
            # Generate default filename
            safe_title = title.replace(" ", "_").lower()
            save_path = Path(f"{safe_title}.png")
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(image)
        
        print(f"Mesh render saved to: {save_path}")
        
    except Exception as e:
        print(f"Warning: Could not render mesh with trimesh renderer: {e}")
        print("Falling back to basic mesh export...")
        
        # Fallback: export as PLY for viewing in external tools
        if save_path:
            fallback_path = Path(save_path).with_suffix('.ply')
        else:
            safe_title = title.replace(" ", "_").lower()
            fallback_path = Path(f"{safe_title}.ply")
        
        mesh.export(str(fallback_path))
        print(f"Mesh exported as PLY to: {fallback_path}")


def render_scene(scene: trimesh.Scene, title: str = "Scene Preview", save_path: Optional[Union[str, Path]] = None, resolution: Tuple[int, int] = (1200, 600)) -> None:
    """
    Render a scene with multiple meshes using trimesh's built-in visualization.
    
    Args:
        scene: trimesh.Scene object
        title: Title for the plot (used in filename if save_path not provided)
        save_path: Path to save the plot (optional)
        resolution: Image resolution as (width, height) tuple
    """
    try:
        # Apply different colors to each geometry for better visualization
        for i, (name, geometry) in enumerate(scene.geometry.items()):
            if isinstance(geometry, trimesh.Trimesh):
                # Create a single color for each mesh
                color = trimesh.visual.random_color()
                # Set the same color for all faces of this mesh
                geometry.visual.face_colors = color
        
        # Generate the image using trimesh's built-in renderer
        image = scene.save_image(resolution=resolution)
        
        # Determine save path
        if save_path:
            save_path = Path(save_path)
        else:
            # Generate default filename
            safe_title = title.replace(" ", "_").lower()
            save_path = Path(f"{safe_title}.png")
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(image)
        
        print(f"Scene render saved to: {save_path}")
        
    except Exception as e:
        print(f"Warning: Could not render scene with trimesh renderer: {e}")
        print("Falling back to basic scene export...")
        
        # Fallback: export as GLB for viewing in external tools
        if save_path:
            fallback_path = Path(save_path).with_suffix('.glb')
        else:
            safe_title = title.replace(" ", "_").lower()
            fallback_path = Path(f"{safe_title}.glb")
        
        export_scene_to_glb(scene, fallback_path)
        print(f"Scene exported as GLB to: {fallback_path}")


def get_mesh_info(mesh: trimesh.Trimesh) -> Dict[str, Union[int, float, Tuple[float, float, float]]]:
    """
    Get information about a mesh.
    
    Args:
        mesh: trimesh.Trimesh object
        
    Returns:
        Dictionary with mesh information
    """
    bounds = mesh.bounds
    return {
        'faces': len(mesh.faces),
        'vertices': len(mesh.vertices),
        'volume': mesh.volume,
        'surface_area': mesh.area,
        'bounding_box_min': tuple(bounds[0]),
        'bounding_box_max': tuple(bounds[1]),
        'center_mass': tuple(mesh.center_mass),
        'is_watertight': mesh.is_watertight,
        'is_winding_consistent': mesh.is_winding_consistent
    }


def print_mesh_info(mesh: trimesh.Trimesh, name: str = "Mesh") -> None:
    """
    Print detailed information about a mesh.
    
    Args:
        mesh: trimesh.Trimesh object
        name: Name/identifier for the mesh
    """
    info = get_mesh_info(mesh)
    print(f"\n{name} Information:")
    print(f"  Faces: {info['faces']:,}")
    print(f"  Vertices: {info['vertices']:,}")
    print(f"  Volume: {info['volume']:.6f}")
    print(f"  Surface Area: {info['surface_area']:.6f}")
    print(f"  Bounding Box: {info['bounding_box_min']} to {info['bounding_box_max']}")
    print(f"  Center of Mass: {info['center_mass']}")
    print(f"  Watertight: {info['is_watertight']}")
    print(f"  Winding Consistent: {info['is_winding_consistent']}")


def multi_stl_to_glb(
    stl_paths: List[Union[str, Path]], 
    output_path: Union[str, Path],
    simplify: bool = True,
    reduction_ratio: float = 0.5,
    target_faces: Optional[int] = None,
    render_preview: bool = True
) -> trimesh.Scene:
    """
    Convert multiple STL files to a single GLB file.
    
    Args:
        stl_paths: List of paths to STL files
        output_path: Path for the output GLB file
        simplify: Whether to simplify meshes (default True)
        reduction_ratio: Reduction ratio for simplification (default 0.5)
        target_faces: Target number of faces for each mesh (overrides reduction_ratio)
        render_preview: Whether to render preview images (default True)
        
    Returns:
        The combined trimesh.Scene object
    """
    if not stl_paths:
        raise ValueError("No STL paths provided")
    
    meshes = []
    mesh_names = []
    
    print(f"Processing {len(stl_paths)} STL files...")
    
    for i, stl_path in enumerate(stl_paths):
        stl_path = Path(stl_path)
        mesh_name = stl_path.stem
        
        try:
            # Load the mesh
            mesh = load_stl(stl_path)
            print_mesh_info(mesh, f"Original {mesh_name}")
            
            # Simplify if requested
            if simplify:
                print(f"Simplifying {mesh_name}...")
                mesh = simplify_mesh(mesh, target_faces, reduction_ratio)
                print_mesh_info(mesh, f"Simplified {mesh_name}")
            
            meshes.append(mesh)
            mesh_names.append(mesh_name)
            
        except Exception as e:
            print(f"Error processing {stl_path}: {e}")
            continue
    
    if not meshes:
        raise ValueError("No meshes could be loaded successfully")
    
    # Combine all meshes into a scene
    scene = combine_meshes(meshes, mesh_names)
    
    # Export to GLB
    export_scene_to_glb(scene, output_path)
    
    # Render preview if requested
    if render_preview:
        print("Rendering combined scene preview...")
        preview_path = Path(output_path).with_suffix('.png')
        render_scene(scene, "Combined Meshes", preview_path)
    
    # Print summary
    total_faces = sum(len(mesh.faces) for mesh in meshes)
    total_vertices = sum(len(mesh.vertices) for mesh in meshes)
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully combined {len(meshes)} meshes")
    print(f"Total faces: {total_faces:,}")
    print(f"Total vertices: {total_vertices:,}")
    print(f"Output saved to: {output_path}")
    
    return scene


def render_scene_multiview(scene: trimesh.Scene, title: str = "Multi-View Scene", 
                          save_path: Optional[Union[str, Path]] = None,
                          view_size: int = 512, colorful: bool = True, 
                          camera_distance_factor: float = 1.8) -> None:
    """
    Render a scene from 4 different camera angles and stitch them into a 2x2 grid.
    
    Args:
        scene: trimesh.Scene object
        title: Title for the render (used in filename if save_path not provided)
        save_path: Path to save the combined image (optional)
        view_size: Size of each individual view (default 512x512)
        colorful: Whether to apply vibrant random colors to each part (default True)
        camera_distance_factor: Multiplier for camera distance (lower = closer = larger robot, default 1.8)
    """
    try:
        from PIL import Image
        import io
        
        # Get scene bounds
        bounds = scene.bounds
        center = scene.centroid
        size = np.max(bounds[1] - bounds[0]) if bounds is not None else 1.0
        
        # Apply colorful random colors to each geometry if requested
        if colorful:
            print(f"  Applying vibrant colors to {len(scene.geometry)} parts...")
            for i, (name, geometry) in enumerate(scene.geometry.items()):
                if isinstance(geometry, trimesh.Trimesh):
                    # Create vibrant colors using HSV color space
                    import colorsys
                    hue = (i * 137.508) % 360  # Golden angle for good distribution
                    saturation = 0.8 + (i % 3) * 0.1  # 0.8-1.0 for vibrant colors
                    value = 0.9 + (i % 2) * 0.1  # 0.9-1.0 for bright colors
                    
                    rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
                    color = [int(c * 255) for c in rgb] + [255]  # Add alpha
                    geometry.visual.face_colors = color
        
        # Distance for camera placement (closer = larger robot)
        cam_distance = size * camera_distance_factor
        
        # Create simplified renders for each view
        rendered_views = {}
        view_names = ['front', 'side', 'top', 'perspective']
        
        print(f"Rendering scene from {len(view_names)} views at {view_size}x{view_size} each...")
        
        for i, view_name in enumerate(view_names):
            print(f"  Rendering {view_name} view...")
            
            try:
                # Create a copy of the scene for this view
                scene_copy = scene.copy()
                
                # Render the view (trimesh will handle camera automatically)
                image_data = scene_copy.save_image(resolution=(view_size, view_size))
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(image_data))
                rendered_views[view_name] = img
                
            except Exception as e:
                print(f"    Warning: Could not render {view_name} view: {e}")
                # Create a placeholder
                placeholder = Image.new('RGB', (view_size, view_size), color='lightgray')
                rendered_views[view_name] = placeholder
        
        # Stitch views together in 2x2 grid
        grid_size = view_size * 2
        combined_image = Image.new('RGB', (grid_size, grid_size), color='white')
        
        positions = {
            'front': (0, 0),
            'side': (view_size, 0),
            'top': (0, view_size),
            'perspective': (view_size, view_size)
        }
        
        for view_name, position in positions.items():
            if view_name in rendered_views:
                combined_image.paste(rendered_views[view_name], position)
        
        # Add labels
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined_image)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            label_positions = {
                'FRONT': (10, 10),
                'SIDE': (view_size + 10, 10),
                'TOP': (10, view_size + 10),
                '3/4 VIEW': (view_size + 10, view_size + 10)
            }
            
            for label, pos in label_positions.items():
                x, y = pos
                # Outline
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), label, font=font, fill='black')
                # Main text
                draw.text((x, y), label, font=font, fill='white')
                
        except ImportError:
            pass
        
        # Save the combined image
        if save_path:
            save_path = Path(save_path)
        else:
            safe_title = title.replace(" ", "_").lower()
            save_path = Path(f"{safe_title}_multiview.png")
        
        combined_image.save(save_path)
        print(f"Multi-view scene render saved to: {save_path}")
        print(f"  Combined size: {grid_size}x{grid_size} pixels")
        
    except ImportError as e:
        print(f"Error: Required PIL (Pillow) library not available: {e}")
        print("Install with: pip install pillow")
    except Exception as e:
        print(f"Error creating multi-view scene render: {e}")
        # Fallback to regular scene render
        print("Falling back to single view render...")
        render_scene(scene, title, save_path)
