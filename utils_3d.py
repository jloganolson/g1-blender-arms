#!/usr/bin/env python3
"""
3D Mesh Utilities
This module provides utilities for working with 3D meshes, including simplification,
combining multiple meshes, and exporting to various formats.
"""

import trimesh
import numpy as np
import matplotlib.pyplot as plt
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
        scene.add_geometry(mesh, node_name=name)
    
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


def render_mesh(mesh: trimesh.Trimesh, title: str = "Mesh Preview", save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Render a mesh using matplotlib.
    
    Args:
        mesh: trimesh.Trimesh object
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    # Set matplotlib to use a non-interactive backend
    import matplotlib
    matplotlib.use('Agg')  # Use Anti-Grain Geometry backend for non-interactive plotting
    
    # Create a figure with 3D projection
    fig = plt.figure(figsize=(12, 5))
    
    # Original mesh wireframe
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_trisurf(mesh.vertices[:, 0], 
                     mesh.vertices[:, 1], 
                     mesh.vertices[:, 2], 
                     triangles=mesh.faces,
                     alpha=0.8,
                     cmap='viridis')
    ax1.set_title(f'{title} - Wireframe')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Solid rendering
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_trisurf(mesh.vertices[:, 0], 
                     mesh.vertices[:, 1], 
                     mesh.vertices[:, 2], 
                     triangles=mesh.faces,
                     alpha=0.9,
                     cmap='plasma',
                     shade=True)
    ax2.set_title(f'{title} - Solid')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Generate default filename
        safe_title = title.replace(" ", "_").lower()
        save_path = f"{safe_title}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)  # Close figure to free memory


def render_scene(scene: trimesh.Scene, title: str = "Scene Preview", save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Render a scene with multiple meshes using matplotlib.
    
    Args:
        scene: trimesh.Scene object
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    # Set matplotlib to use a non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    # Create a figure with 3D projection
    fig = plt.figure(figsize=(12, 5))
    
    # Combined mesh wireframe
    ax1 = fig.add_subplot(121, projection='3d')
    colors = plt.cm.Set3(np.linspace(0, 1, len(scene.geometry)))
    
    for i, (name, geometry) in enumerate(scene.geometry.items()):
        if isinstance(geometry, trimesh.Trimesh):
            ax1.plot_trisurf(geometry.vertices[:, 0], 
                           geometry.vertices[:, 1], 
                           geometry.vertices[:, 2], 
                           triangles=geometry.faces,
                           alpha=0.7,
                           color=colors[i])
    
    ax1.set_title(f'{title} - Combined Wireframe')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Combined mesh solid
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i, (name, geometry) in enumerate(scene.geometry.items()):
        if isinstance(geometry, trimesh.Trimesh):
            ax2.plot_trisurf(geometry.vertices[:, 0], 
                           geometry.vertices[:, 1], 
                           geometry.vertices[:, 2], 
                           triangles=geometry.faces,
                           alpha=0.8,
                           color=colors[i],
                           shade=True)
    
    ax2.set_title(f'{title} - Combined Solid')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        # Generate default filename
        safe_title = title.replace(" ", "_").lower()
        save_path = f"{safe_title}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)  # Close figure to free memory


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
