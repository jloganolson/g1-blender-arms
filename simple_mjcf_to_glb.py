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
from render_glb_screenshot import render_glb_multiview
try:
    # Optional rigging support
    from armature_exporter.gltf_armature_builder import (
        GLTFArmatureBuilder,
        build_bone_hierarchy,
    )
    _RIGGING_AVAILABLE = True
    _RIGGING_IMPORT_ERROR = None
except Exception as _e:
    _RIGGING_AVAILABLE = False
    _RIGGING_IMPORT_ERROR = _e


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



def mjcf_to_rigged_glb(mjcf_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Export a minimally rigged GLB using the armature builder (if available).
    Falls back gracefully if rigging dependencies are missing.
    """
    output_path = Path(output_path)
    print(f"Converting MJCF to RIGGED GLB: {mjcf_path} -> {output_path}")
    if not _RIGGING_AVAILABLE:
        raise RuntimeError(f"Rigging dependency unavailable: {_RIGGING_IMPORT_ERROR}")

    parser = MJCFParser(mjcf_path)

    # Minimal joint set for mvp_test.xml (two joints)
    target_joints = [
        "waist_yaw_joint",
        "right_shoulder_pitch_joint",
    ]

    bones, bone_hierarchy, mesh_instance_weights = build_bone_hierarchy(parser, target_joints)
    builder = GLTFArmatureBuilder(parser, bones, bone_hierarchy, mesh_instance_weights)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder.build_rigged_gltf(str(output_path))
    print(f"✅ MJCF exported to rigged GLB: {output_path}")


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


def mjcf_to_glb_riglike(mjcf_path: Union[str, Path], output_path: Union[str, Path],
                        simplify: bool = True, max_faces_per_mesh: int = 5000,
                        target_joints: list[str] | None = None) -> trimesh.Scene:
    """
    Export MJCF to GLB (unrigged) but compute and apply transforms using the
    same bone/control logic as the rigged exporter. This helps diagnose any
    placement disparities introduced by rig building.

    Strategy:
    - Build the same bone hierarchy and mesh-instance-to-bone mapping used by the rigged path.
    - Use the exact transform logic from GLTFArmatureBuilder to position meshes.
    - Apply bone-relative transforms as done in the rigged exporter.
    - Debug output shows bone hierarchy, weights, and transform calculations.
    """
    print(f"Converting MJCF to GLB (rig-like placement): {mjcf_path} -> {output_path}")

    if not _RIGGING_AVAILABLE:
        raise RuntimeError(f"Rig-like placement requires rigging utilities: {_RIGGING_IMPORT_ERROR}")

    # Parse MJCF and build rig data
    parser = MJCFParser(mjcf_path)

    # Default to the same minimal set used by mjcf_to_rigged_glb
    if target_joints is None:
        target_joints = [
            "waist_yaw_joint",
            "right_shoulder_pitch_joint",
        ]

    bones, bone_hierarchy, mesh_instance_weights = build_bone_hierarchy(parser, target_joints)

    # Debug: Print bone hierarchy and transforms
    print(f"\n=== Bone Hierarchy Debug ===")
    print(f"Target joints: {target_joints}")
    print(f"Bone hierarchy order: {bone_hierarchy}")
    
    for bone_name in bone_hierarchy:
        bone = bones[bone_name]
        print(f"\nBone: {bone_name}")
        print(f"  Body: {bone.body_name}")
        print(f"  Parent: {bone.parent_bone}")
        print(f"  Position: {bone.transform_matrix[:3, 3]}")
        print(f"  Children: {bone.children}")

    # Debug: Print mesh instance weights
    print(f"\n=== Mesh Instance Weights ===")
    for instance_name, weights in mesh_instance_weights.items():
        if weights:
            print(f"{instance_name}: {weights}")

    # Prepare scene and mesh cache
    scene = trimesh.Scene()
    processed_meshes: dict[str, trimesh.Trimesh] = {}

    mesh_transforms = parser.get_mesh_transforms()
    print(f"\n=== Processing {len(mesh_transforms)} unique meshes (rig-like) ===")

    for mesh_name, transforms in mesh_transforms.items():
        mesh_file_path = parser.get_mesh_file_path(mesh_name)
        if mesh_file_path is None or not mesh_file_path.exists():
            print(f"  Warning: Skipping missing mesh {mesh_name}")
            continue

        print(f"\nProcessing mesh: {mesh_name}")

        # Load and optionally simplify base mesh
        if mesh_name not in processed_meshes:
            try:
                mesh = load_stl(mesh_file_path)
                if simplify and len(mesh.faces) > max_faces_per_mesh:
                    print(f"  Simplifying {mesh_name}: {len(mesh.faces)} -> {max_faces_per_mesh} faces")
                    mesh = mesh.simplify_quadric_decimation(face_count=max_faces_per_mesh)
                processed_meshes[mesh_name] = mesh
            except Exception as e:
                print(f"  Error loading {mesh_name}: {e}")
                continue

        base_mesh = processed_meshes[mesh_name]

        for i, (body_name, position, rotation_matrix, material) in enumerate(transforms):
            # Create instance name for weight lookup (same as GLTFArmatureBuilder)
            instance_name = f"{body_name}_{mesh_name}"
            if len(transforms) > 1:
                instance_name += f"_{i}"

            print(f"  Instance: {instance_name}")
            print(f"    Body: {body_name}")
            print(f"    Original position: {position}")

            # Create transform matrix from parser data
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = position

            # Apply the same logic as GLTFArmatureBuilder.build_rigged_gltf
            final_transform = transform_matrix.copy()
            
            if instance_name in mesh_instance_weights and mesh_instance_weights[instance_name]:
                # This mesh instance is controlled by a bone - make transform relative to bone
                controlling_bones = list(mesh_instance_weights[instance_name].keys())
                if controlling_bones:
                    controlling_bone_name = controlling_bones[0]  # Use first controlling bone
                    print(f"    Controlled by bone: {controlling_bone_name}")
                    
                    if controlling_bone_name in bones:
                        bone = bones[controlling_bone_name]
                        
                        # Get bone transform (using same method as GLTFArmatureBuilder)
                        bone_transform = bone.transform_matrix
                        print(f"    Bone position: {bone_transform[:3, 3]}")
                        
                        # Make mesh transform relative to bone by removing bone's transform
                        # This matches the logic in GLTFArmatureBuilder lines 400-405
                        try:
                            bone_inverse = np.linalg.inv(bone_transform)
                            relative_transform = bone_inverse @ transform_matrix
                            
                            # For rig-like placement, we want to see the effect of this logic
                            # So we'll apply both the relative computation and then recompose
                            final_transform = bone_transform @ relative_transform
                            
                            print(f"    Relative transform applied")
                            print(f"    Final position: {final_transform[:3, 3]}")
                            
                        except np.linalg.LinAlgError:
                            print(f"    Warning: Could not invert bone transform, using original")
                            # If bone transform is not invertible, use original transform
                            pass
                    else:
                        print(f"    Warning: Controlling bone {controlling_bone_name} not found in bones")
            else:
                print(f"    No bone control (unweighted)")

            # Apply final transform to mesh
            transformed_mesh = base_mesh.copy()
            transformed_mesh.apply_transform(final_transform)

            node_name = instance_name + "_riglike"
            scene.add_geometry(transformed_mesh, node_name=node_name)

    # Export and return
    export_scene_to_glb(scene, output_path)
    print(f"\n✅ MJCF exported to GLB (rig-like placement): {output_path}")
    return scene


def main():
    # Defaults
    default_mjcf = "g1_description/mvp_test.xml"
    default_unrigged = "output/mvp_simple.glb"
    default_rigged = "output/mvp_simple_rigged.glb"

    # Simple flag parsing
    argv = sys.argv[1:]
    # Legacy flags (kept for compatibility but not used in default flow)
    rigged_only = "--rigged-only" in argv
    riglike_unrigged = "--riglike-unrigged" in argv
    # New single-mode flags
    only_unrigged = "--only-unrigged" in argv
    only_riglike = "--only-riglike" in argv
    args = [a for a in argv if not a.startswith("--")]

    mjcf_file = default_mjcf
    output_file = default_rigged if rigged_only else default_unrigged

    if len(args) > 0:
        mjcf_file = args[0]
    if len(args) > 1:
        output_file = args[1]
        if "/" not in output_file and "\\" not in output_file:
            output_file = f"output/{output_file}"

    # Rigged path intentionally left out by default per request
    if rigged_only:
        print("--rigged-only is currently disabled in this workflow.")
        return 0

    # Derive base output and both filenames
    base_output = Path(output_file)
    if base_output.suffix.lower() != ".glb":
        base_output = base_output.with_suffix(".glb")
    unrigged_out = str(base_output)
    riglike_out = str(base_output.with_name(f"{base_output.stem}_riglike.glb"))

    # Handle legacy riglike flag as a single-mode alias
    if riglike_unrigged and not (only_unrigged or only_riglike):
        only_riglike = True

    # If both single-mode flags given, fall back to both
    if only_unrigged and only_riglike:
        print("Both --only-unrigged and --only-riglike provided; exporting both modes.")
        only_unrigged = False
        only_riglike = False

    # Single-mode: unrigged only
    if only_unrigged:
        print("Exporting only: unrigged")
        mjcf_to_glb(mjcf_file, unrigged_out)
        screenshot_path = Path(unrigged_out).with_suffix('.png')
        print(f"\nGenerating multiview screenshot...")
        try:
            render_glb_multiview(unrigged_out, screenshot_path, view_width=400, view_height=300)
        except Exception as e:
            print(f"Warning: Could not create screenshot: {e}")
        return 0

    # Single-mode: riglike only
    if only_riglike:
        print("Exporting only: riglike-unrigged")
        mjcf_to_glb_riglike(mjcf_file, riglike_out)
        screenshot_path = Path(riglike_out).with_suffix('.png')
        print(f"\nGenerating multiview screenshot...")
        try:
            render_glb_multiview(riglike_out, screenshot_path, view_width=400, view_height=300)
        except Exception as e:
            print(f"Warning: Could not create screenshot: {e}")
        return 0

    # Default: export both
    print("Exporting both: unrigged and riglike-unrigged")
    mjcf_to_glb(mjcf_file, unrigged_out)
    try:
        render_glb_multiview(unrigged_out, Path(unrigged_out).with_suffix('.png'), view_width=400, view_height=300)
    except Exception as e:
        print(f"Warning: Could not create screenshot for unrigged: {e}")

    mjcf_to_glb_riglike(mjcf_file, riglike_out)
    try:
        render_glb_multiview(riglike_out, Path(riglike_out).with_suffix('.png'), view_width=400, view_height=300)
    except Exception as e:
        print(f"Warning: Could not create screenshot for riglike: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
