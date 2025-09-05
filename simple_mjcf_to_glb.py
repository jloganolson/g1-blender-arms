#!/usr/bin/env python3
"""
Simple script to convert MJCF to unrigged GLB format.
"""

import sys
import trimesh
import numpy as np
from pathlib import Path
from typing import Union
from armature_exporter.mjcf_parser import MJCFParser
from render_glb_screenshot import render_glb_multiview


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
    # Defaults
    default_mjcf = "g1_description/mvp_test.xml"
    default_output = "output/mvp_simple.glb"

    # Simple argument parsing
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]

    mjcf_file = default_mjcf
    output_file = default_output

    if len(args) > 0:
        mjcf_file = args[0]
    if len(args) > 1:
        output_file = args[1]
        if "/" not in output_file and "\\" not in output_file:
            output_file = f"output/{output_file}"

    # Ensure output has .glb extension
    output_path = Path(output_file)
    if output_path.suffix.lower() != ".glb":
        output_path = output_path.with_suffix(".glb")

    # Export unrigged GLB
    print(f"Converting MJCF to unrigged GLB: {mjcf_file} -> {output_path}")
    mjcf_to_glb(mjcf_file, output_path)
    
    # Generate screenshot
    screenshot_path = output_path.with_suffix('.png')
    print(f"\nGenerating multiview screenshot...")
    try:
        render_glb_multiview(str(output_path), screenshot_path, view_width=400, view_height=300)
    except Exception as e:
        print(f"Warning: Could not create screenshot: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
