#!/usr/bin/env python3
"""
Simple MJCF to GLB Converter
A streamlined script that converts MJCF files to GLB format and generates a preview PNG.
No command line arguments - just simple, clean conversion.
"""

import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation

from mjcf_parser import MJCFParser
from utils_3d import load_stl, export_scene_to_glb, render_scene_multiview


def apply_transform_to_mesh(mesh: trimesh.Trimesh, position: np.ndarray, rotation_matrix: np.ndarray) -> trimesh.Trimesh:
    """Apply position and rotation transform to a mesh."""
    # Create transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    
    # Apply transformation
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(transform_matrix)
    
    return transformed_mesh


def simple_mjcf_to_glb(mjcf_path: str = "./g1_description/g1_mjx_alt.xml", 
                       output_name: str = "robot") -> None:
    """
    Convert MJCF to GLB with basic settings.
    
    Args:
        mjcf_path: Path to MJCF file
        output_name: Output filename (without extension)
    """
    mjcf_path = Path(mjcf_path)
    
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
    
    print(f"Converting {mjcf_path} to GLB...")
    
    # Parse MJCF
    parser = MJCFParser(mjcf_path)
    mesh_transforms = parser.get_mesh_transforms()
    
    print(f"Found {len(mesh_transforms)} unique meshes")
    
    # Create scene
    scene = trimesh.Scene()
    processed_meshes = {}
    
    # Process each mesh
    for mesh_name, transforms in mesh_transforms.items():
        print(f"Processing {mesh_name}...")
        
        # Get mesh file path
        mesh_file_path = parser.get_mesh_file_path(mesh_name)
        if mesh_file_path is None or not mesh_file_path.exists():
            print(f"  Warning: Skipping missing mesh {mesh_name}")
            continue
        
        # Load mesh (cache it if we haven't loaded it before)
        if mesh_name not in processed_meshes:
            try:
                mesh = load_stl(mesh_file_path)
                # Simple reduction to keep file size reasonable
                if len(mesh.faces) > 5000:
                    mesh = mesh.simplify_quadric_decimation(face_count=5000)
                processed_meshes[mesh_name] = mesh
            except Exception as e:
                print(f"  Error loading {mesh_name}: {e}")
                continue
        
        base_mesh = processed_meshes[mesh_name]
        
        # Add transformed instances to scene
        for i, (body_name, position, rotation_matrix, material) in enumerate(transforms):
            transformed_mesh = apply_transform_to_mesh(base_mesh, position, rotation_matrix)
            instance_name = f"{body_name}_{mesh_name}"
            if len(transforms) > 1:
                instance_name += f"_{i}"
            scene.add_geometry(transformed_mesh, node_name=instance_name)
    
    # Export GLB
    glb_path = f"output/{output_name}.glb"
    export_scene_to_glb(scene, glb_path)
    
    # Render multi-view technical drawing  
    png_path = f"output/{output_name}_multiview.png"
    render_scene_multiview(
        scene, 
        f"{output_name.title()} Robot Technical Drawing", 
        png_path,
        view_size=512,
        colorful=True,
        camera_distance_factor=1.8
    )
    
    print(f"✅ Conversion complete!")
    print(f"   GLB: {glb_path}")
    print(f"   PNG: {png_path}")


if __name__ == "__main__":
    # Simple usage - convert G1 robot
    simple_mjcf_to_glb()
