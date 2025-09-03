#!/usr/bin/env python3
"""
MJCF to GLB Converter
This script parses an MJCF file and converts it to a GLB file with correctly positioned meshes.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from mjcf_parser import MJCFParser
from utils_3d import load_stl, simplify_mesh, export_scene_to_glb, render_scene, print_mesh_info
from armature_utils import ArmatureBuilder
from skinning_utils import SkinnedMeshBuilder, create_test_poses
from pose_deformation import PoseDeformer


def apply_transform_to_mesh(mesh: trimesh.Trimesh, position: np.ndarray, rotation_matrix: np.ndarray) -> trimesh.Trimesh:
    """
    Apply a transformation (position and rotation) to a mesh.
    
    Args:
        mesh: The mesh to transform
        position: 3D translation vector
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        Transformed mesh
    """
    # Create transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    
    # Apply transformation
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(transform_matrix)
    
    return transformed_mesh


def mjcf_to_glb(
    mjcf_path: str,
    output_path: str,
    meshdir: Optional[str] = None,
    simplify: bool = True,
    reduction_ratio: float = 0.3,
    target_faces: Optional[int] = None,
    render_preview: bool = True,
    skip_missing: bool = True,
    include_armature: bool = False,
    include_skinning: bool = False,
    create_test_poses_flag: bool = False
) -> trimesh.Scene:
    """
    Convert an MJCF file to GLB with correctly positioned meshes.
    
    Args:
        mjcf_path: Path to the MJCF file
        output_path: Path for the output GLB file
        meshdir: Directory containing mesh files (overrides MJCF meshdir)
        simplify: Whether to simplify meshes
        reduction_ratio: Reduction ratio for simplification
        target_faces: Target number of faces per mesh
        render_preview: Whether to render preview images
        skip_missing: Whether to skip missing mesh files
        include_armature: Whether to include skeleton/armature in the output
        include_skinning: Whether to include vertex weights for rigging
        create_test_poses_flag: Whether to create test pose images
        
    Returns:
        The combined trimesh.Scene object
    """
    mjcf_path = Path(mjcf_path)
    output_path = Path(output_path)
    
    print(f"=== MJCF to GLB Conversion ===")
    print(f"Input MJCF: {mjcf_path}")
    print(f"Output GLB: {output_path}")
    
    # Parse the MJCF file
    print(f"\nParsing MJCF file...")
    parser = MJCFParser(mjcf_path)
    
    # Print summary
    summary = parser.get_summary()
    print(f"Found {summary['total_meshes_defined']} mesh definitions")
    print(f"Found {summary['total_bodies']} bodies")
    print(f"Found {summary['meshes_used']} unique meshes used")
    
    # Get mesh transforms
    mesh_transforms = parser.get_mesh_transforms()
    
    # Create the scene
    scene = trimesh.Scene()
    processed_meshes = {}
    simplified_meshes = {}  # Store simplified meshes for skinning
    total_faces = 0
    total_vertices = 0
    
    print(f"\nProcessing meshes...")
    
    for mesh_name, transforms in mesh_transforms.items():
        print(f"\n--- Processing mesh: {mesh_name} ---")
        print(f"Used in {len(transforms)} locations")
        
        # Get the mesh file path
        mesh_file_path = parser.get_mesh_file_path(mesh_name, meshdir)
        if mesh_file_path is None:
            print(f"  ERROR: Mesh file path not found for {mesh_name}")
            if skip_missing:
                continue
            else:
                raise FileNotFoundError(f"Mesh file not found for {mesh_name}")
        
        # Check if file exists
        if not mesh_file_path.exists():
            print(f"  ERROR: Mesh file does not exist: {mesh_file_path}")
            if skip_missing:
                continue
            else:
                raise FileNotFoundError(f"Mesh file does not exist: {mesh_file_path}")
        
        try:
            # Load the base mesh (only once per unique mesh)
            if mesh_name not in processed_meshes:
                print(f"  Loading: {mesh_file_path}")
                base_mesh = load_stl(mesh_file_path)
                print_mesh_info(base_mesh, f"Original {mesh_name}")
                
                # Simplify if requested
                if simplify:
                    print(f"  Simplifying...")
                    simplified_mesh = simplify_mesh(base_mesh, target_faces, reduction_ratio)
                    print_mesh_info(simplified_mesh, f"Simplified {mesh_name}")
                    processed_meshes[mesh_name] = simplified_mesh
                    simplified_meshes[mesh_name] = simplified_mesh  # Store for skinning
                else:
                    processed_meshes[mesh_name] = base_mesh
                    simplified_meshes[mesh_name] = base_mesh  # Store for skinning
            
            base_mesh = processed_meshes[mesh_name]
            
            # Skip regular mesh processing if we're doing skinning (will be handled later)
            if not include_skinning:
                # Create a transformed copy for each location
                for i, (body_name, position, rotation_matrix, material) in enumerate(transforms):
                    print(f"  Adding to body: {body_name}")
                    print(f"    Position: {position}")
                    print(f"    Material: {material}")
                    
                    # Transform the mesh
                    transformed_mesh = apply_transform_to_mesh(base_mesh, position, rotation_matrix)
                    
                    # Create a unique name for this instance
                    instance_name = f"{body_name}_{mesh_name}"
                    if len(transforms) > 1:
                        instance_name += f"_instance_{i}"
                    
                    # Add to scene
                    scene.add_geometry(transformed_mesh, node_name=instance_name)
                    
                    # Update totals
                    total_faces += len(transformed_mesh.faces)
                    total_vertices += len(transformed_mesh.vertices)
        
        except Exception as e:
            print(f"  ERROR processing {mesh_name}: {e}")
            if not skip_missing:
                raise
    
    # Handle skinning if requested
    if include_skinning:
        print(f"\n=== Creating Skinned Scene ===")
        armature_builder = ArmatureBuilder(parser)
        skinned_builder = SkinnedMeshBuilder(parser, armature_builder)
        
        # Create skinned scene (replaces regular scene)
        scene = skinned_builder.create_skinned_scene(mesh_transforms, simplified_meshes)
        
        # Update totals from skinned scene
        total_faces = sum(len(geom.faces) for geom in scene.geometry.values() if hasattr(geom, 'faces'))
        total_vertices = sum(len(geom.vertices) for geom in scene.geometry.values() if hasattr(geom, 'vertices'))
        
        print(f"\n=== Skinned Scene Complete ===")
        print(f"Total geometries: {len(scene.geometry)}")
        print(f"Total faces: {total_faces:,}")
        print(f"Total vertices: {total_vertices:,}")
        
        # Add armature visualization if also requested
        if include_armature:
            print(f"\nAdding armature visualization...")
            armature_scene = armature_builder.create_trimesh_armature()
            if armature_scene:
                print(f"Adding {len(armature_scene.geometry)} bones to scene")
                for name, geom in armature_scene.geometry.items():
                    scene.add_geometry(geom, node_name=f"armature_{name}")
        
        # Create test poses if requested
        if create_test_poses_flag:
            print(f"\n=== Creating Test Poses ===")
            bones = armature_builder.bones
            test_poses = create_test_poses(bones)
            
            # Create pose deformer
            pose_deformer = PoseDeformer(bones)
            
            for pose_name, pose_angles in test_poses.items():
                print(f"\n{'='*50}")
                print(f"CREATING POSE: {pose_name.upper()}")
                print(f"{'='*50}")
                
                if pose_angles:
                    # Apply pose deformation
                    posed_scene = pose_deformer.create_posed_scene(scene, pose_angles)
                    pose_output_path = Path("output") / f"{output_path.stem}_{pose_name}.png"
                    render_scene(posed_scene, f"G1 Robot - {pose_name}", pose_output_path)
                    print(f"✅ Saved deformed pose: {pose_output_path}")
                else:
                    # No pose changes, just render base scene
                    pose_output_path = Path("output") / f"{output_path.stem}_{pose_name}.png"
                    render_scene(scene, f"G1 Robot - {pose_name}", pose_output_path)
                    print(f"📷 Saved base pose: {pose_output_path}")
    
    else:
        print(f"\n=== Scene Assembly Complete ===")
        print(f"Total geometries: {len(scene.geometry)}")
        print(f"Total faces: {total_faces:,}")
        print(f"Total vertices: {total_vertices:,}")
        
        # Add armature if requested
        if include_armature:
            print(f"\nGenerating armature...")
            armature_builder = ArmatureBuilder(parser)
            armature_scene = armature_builder.create_trimesh_armature()
            
            if armature_scene:
                print(f"Adding {len(armature_scene.geometry)} bones to scene")
                # Merge armature into main scene
                for name, geom in armature_scene.geometry.items():
                    scene.add_geometry(geom, node_name=f"armature_{name}")
                
                # Print armature info
                armature_builder.print_armature_info()
    
    # Export to GLB
    print(f"\nExporting to GLB...")
    export_scene_to_glb(scene, output_path)
    
    # Render preview if requested
    if render_preview:
        print(f"Rendering preview...")
        preview_path = Path("output") / f"{output_path.stem}.png"
        render_scene(scene, "MJCF Robot Assembly", preview_path)
    
    print(f"\n✅ Conversion complete!")
    print(f"Output: {output_path}")
    
    return scene


def main():
    parser = argparse.ArgumentParser(description='Convert MJCF file to GLB with positioned meshes')
    
    # Input/Output
    parser.add_argument('mjcf_file', nargs='?', default='./g1_description/g1_mjx_alt.xml', 
                       help='Path to MJCF file (default: ./g1_description/g1_mjx_alt.xml)')
    parser.add_argument('-o', '--output', default='g1_robot_positioned.glb', 
                       help='Output GLB file path (default: g1_robot_positioned.glb)')
    parser.add_argument('--meshdir', help='Override mesh directory path')
    
    # Simplification options
    parser.add_argument('--no-simplify', action='store_true', help='Skip mesh simplification')
    parser.add_argument('-r', '--ratio', type=float, default=0.2,
                       help='Reduction ratio for simplification (default: 0.2)')
    parser.add_argument('-t', '--target-faces', type=int,
                       help='Target number of faces per mesh (overrides ratio)')
    
    # Processing options
    parser.add_argument('--no-render', action='store_true', help='Skip rendering preview')
    parser.add_argument('--strict', action='store_true', 
                       help='Fail on missing meshes (default: skip missing)')
    parser.add_argument('--armature', action='store_true', 
                       help='Include skeleton/armature in the output GLB')
    parser.add_argument('--skinning', action='store_true',
                       help='Include vertex weights for rigging (enables skinning)')
    parser.add_argument('--test-poses', action='store_true',
                       help='Create test pose images to verify rigging')
    
    # Debug options
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    # Validate inputs
    mjcf_file = Path(args.mjcf_file)
    if not mjcf_file.exists():
        print(f"Error: MJCF file does not exist: {mjcf_file}")
        return 1
    
    output_path = Path(args.output)
    if not output_path.suffix.lower() == '.glb':
        print("Warning: Output file extension is not .glb, adding .glb suffix")
        output_path = output_path.with_suffix('.glb')
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Debug mode
    if args.debug:
        print("=== Debug Mode: Parsing MJCF ===")
        debug_parser = MJCFParser(mjcf_file)
        debug_parser.print_hierarchy()
        
        print("\n=== Mesh Transforms ===")
        mesh_transforms = debug_parser.get_mesh_transforms()
        for mesh_name, transforms in mesh_transforms.items():
            print(f"\nMesh: {mesh_name}")
            for body_name, pos, rot, material in transforms:
                print(f"  Body: {body_name}, Material: {material}")
                print(f"    Position: {pos}")
        print()
    
    try:
        # Convert MJCF to GLB
        scene = mjcf_to_glb(
            mjcf_path=mjcf_file,
            output_path=output_path,
            meshdir=args.meshdir,
            simplify=not args.no_simplify,
            reduction_ratio=args.ratio,
            target_faces=args.target_faces,
            render_preview=not args.no_render,
            skip_missing=not args.strict,
            include_armature=args.armature,
            include_skinning=args.skinning,
            create_test_poses_flag=args.test_poses
        )
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # If run without arguments, show usage and run with defaults if G1 MJCF exists
    if len(sys.argv) == 1:
        print("MJCF to GLB Converter")
        print("\nDefault usage (no arguments needed for G1 robot):")
        print("  python mjcf_to_glb.py")
        print("  # Uses: ./g1_description/g1_mjx_alt.xml → g1_robot_positioned.glb (20% reduction)")
        print("\nCustom usage:")
        print("  python mjcf_to_glb.py custom_robot.xml -o custom_output.glb")
        print("  python mjcf_to_glb.py --ratio 0.1  # Higher quality (10% reduction)")
        print("  python mjcf_to_glb.py --debug      # Show hierarchy info")
        print("")
        
        # Check if G1 robot is available for default processing
        g1_mjcf = Path("./g1_description/g1_mjx_alt.xml")
        if g1_mjcf.exists():
            print("✅ Found G1 robot MJCF! Running with default settings...")
            print("   Input: ./g1_description/g1_mjx_alt.xml")
            print("   Output: g1_robot_positioned.glb")
            print("   Reduction: 20%")
            print("")
            # Don't modify sys.argv, let it proceed with defaults
        else:
            print("❌ No G1 MJCF found at ./g1_description/g1_mjx_alt.xml")
            print("Please provide an MJCF file path as argument.")
            sys.exit(1)
    
    sys.exit(main())
