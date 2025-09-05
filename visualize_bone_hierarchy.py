#!/usr/bin/env python3
"""
Visualize bone hierarchy using primitive shapes.
This script builds the same bone hierarchy as GLTFArmatureBuilder.create_armature_nodes
and creates a GLB with primitive shapes to visualize the bone structure.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from armature_exporter.mjcf_parser import MJCFParser
from armature_exporter.gltf_armature_builder import build_bone_hierarchy, BoneInfo
from render_glb_screenshot import render_glb_multiview


def create_bone_primitive(bone_name: str, bone: BoneInfo, parent_bone: Optional[BoneInfo] = None) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Create a primitive shape to represent a bone.
    Returns (mesh, transform_matrix)
    """
    # Create a small sphere for the joint
    joint_radius = 0.02
    joint_sphere = trimesh.creation.icosphere(radius=joint_radius, subdivisions=2)
    
    # Color the sphere based on bone type
    if parent_bone is None:
        # Root bone - red
        joint_sphere.visual.vertex_colors = [255, 0, 0, 255]
    else:
        # Child bone - blue
        joint_sphere.visual.vertex_colors = [0, 0, 255, 255]
    
    # Position the sphere at the bone's position
    bone_position = bone.transform_matrix[:3, 3]
    transform = np.eye(4)
    transform[:3, 3] = bone_position
    
    return joint_sphere, transform


def create_bone_connection(parent_bone: BoneInfo, child_bone: BoneInfo) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Create a cylinder to connect parent and child bones.
    Returns (mesh, transform_matrix)
    """
    parent_pos = parent_bone.transform_matrix[:3, 3]
    child_pos = child_bone.transform_matrix[:3, 3]
    
    # Calculate direction and distance
    direction = child_pos - parent_pos
    distance = np.linalg.norm(direction)
    
    if distance < 1e-6:  # Too small to visualize
        return None, None
    
    # Create cylinder
    cylinder_radius = 0.005
    cylinder = trimesh.creation.cylinder(radius=cylinder_radius, height=distance)
    cylinder.visual.vertex_colors = [0, 255, 0, 255]  # Green connections
    
    # Calculate transform to align cylinder with bone direction
    # Default cylinder is along Z-axis, we need to align with direction
    normalized_direction = direction / distance
    
    # Create rotation matrix to align Z-axis with direction
    z_axis = np.array([0, 0, 1])
    if np.allclose(normalized_direction, z_axis):
        rotation_matrix = np.eye(3)
    elif np.allclose(normalized_direction, -z_axis):
        rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        # Use Rodrigues' rotation formula
        v = np.cross(z_axis, normalized_direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, normalized_direction)
        
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    
    # Create transform matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = parent_pos + direction / 2  # Position at midpoint
    
    return cylinder, transform


def load_and_process_stl(mesh_file_path: Path, controlled_by_bone: bool = False, 
                        simplify: bool = True, max_faces: int = 1000) -> trimesh.Trimesh:
    """
    Load and process an STL file for visualization.
    Returns processed mesh with appropriate colors.
    """
    try:
        # Load the STL file
        mesh = trimesh.load(str(mesh_file_path))
        
        # Handle scene vs direct mesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError(f"No geometry found in {mesh_file_path}")
            mesh = list(mesh.geometry.values())[0]
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid mesh: {type(mesh)}")
        
        # Simplify if needed for better performance
        if simplify and len(mesh.faces) > max_faces:
            print(f"    Simplifying {mesh_file_path.name}: {len(mesh.faces)} -> {max_faces} faces")
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        
        # Color based on whether it's controlled by a bone
        if controlled_by_bone:
            # Magenta for bone-controlled meshes
            mesh.visual.vertex_colors = [255, 0, 255, 255]
        else:
            # Yellow for uncontrolled meshes
            mesh.visual.vertex_colors = [255, 255, 0, 255]
        
        return mesh
        
    except Exception as e:
        print(f"    Warning: Could not load {mesh_file_path}: {e}")
        # Fallback to a small cube
        mesh_size = 0.015
        fallback_mesh = trimesh.creation.box(extents=[mesh_size, mesh_size, mesh_size])
        
        # Color the fallback mesh differently (red for errors)
        fallback_mesh.visual.vertex_colors = [255, 0, 0, 255]
        return fallback_mesh


def create_mesh_primitive(instance_name: str, mesh_name: str, original_position: np.ndarray, 
                         final_position: np.ndarray, controlled_by_bone: bool = False,
                         parser: 'MJCFParser' = None, mesh_cache: dict = None) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Create a mesh instance using the actual STL geometry.
    Returns (mesh, transform_matrix)
    """
    if mesh_cache is None:
        mesh_cache = {}
    
    # Try to load the actual STL mesh
    if parser:
        mesh_file_path = parser.get_mesh_file_path(mesh_name)
        if mesh_file_path and mesh_file_path.exists():
            # Check cache first
            cache_key = (str(mesh_file_path), controlled_by_bone)
            if cache_key in mesh_cache:
                mesh = mesh_cache[cache_key].copy()
            else:
                mesh = load_and_process_stl(mesh_file_path, controlled_by_bone)
                mesh_cache[cache_key] = mesh
                mesh = mesh.copy()  # Return a copy to avoid modifying the cached version
        else:
            print(f"    Warning: Could not find mesh file for {mesh_name}")
            # Fallback to cube
            mesh_size = 0.015
            mesh = trimesh.creation.box(extents=[mesh_size, mesh_size, mesh_size])
            mesh.visual.vertex_colors = [255, 0, 0, 255]  # Red for missing files
    else:
        # Fallback to cube if no parser provided
        mesh_size = 0.015
        mesh = trimesh.creation.box(extents=[mesh_size, mesh_size, mesh_size])
        if controlled_by_bone:
            mesh.visual.vertex_colors = [255, 0, 255, 255]
        else:
            mesh.visual.vertex_colors = [255, 255, 0, 255]
    
    # Position the mesh at the final position
    transform = np.eye(4)
    transform[:3, 3] = final_position
    
    return mesh, transform


def create_mesh_displacement_line(original_position: np.ndarray, final_position: np.ndarray) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Create a line showing displacement from original to final position.
    Returns (mesh, transform_matrix)
    """
    displacement = final_position - original_position
    distance = np.linalg.norm(displacement)
    
    if distance < 1e-6:  # No displacement
        return None, None
    
    # Create thin cylinder for the displacement line
    line_radius = 0.001
    line = trimesh.creation.cylinder(radius=line_radius, height=distance)
    line.visual.vertex_colors = [255, 165, 0, 255]  # Orange displacement lines
    
    # Calculate transform to position and orient the line
    normalized_displacement = displacement / distance
    
    # Default cylinder is along Z-axis, align with displacement
    z_axis = np.array([0, 0, 1])
    if np.allclose(normalized_displacement, z_axis):
        rotation_matrix = np.eye(3)
    elif np.allclose(normalized_displacement, -z_axis):
        rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        # Use Rodrigues' rotation formula
        v = np.cross(z_axis, normalized_displacement)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, normalized_displacement)
        
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    
    # Create transform matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = original_position + displacement / 2  # Position at midpoint
    
    return line, transform


def visualize_bone_hierarchy(mjcf_path: str, output_path: str, target_joints: List[str] = None, include_meshes: bool = True):
    """
    Create a GLB visualization of the bone hierarchy and optionally mesh positioning.
    """
    print(f"Visualizing bone hierarchy from: {mjcf_path}")
    
    # Parse MJCF and build bone hierarchy
    parser = MJCFParser(mjcf_path)
    
    # Cache for loaded meshes to avoid reloading the same STL multiple times
    mesh_cache = {}
    
    if target_joints is None:
        target_joints = [
            "waist_yaw_joint",
            "right_shoulder_pitch_joint",
        ]
    
    print(f"Target joints: {target_joints}")
    
    # Build bone hierarchy using the same logic as GLTFArmatureBuilder
    bones, bone_hierarchy, mesh_instance_weights = build_bone_hierarchy(parser, target_joints)
    
    print(f"\n=== Bone Hierarchy Analysis ===")
    print(f"Bone hierarchy order: {bone_hierarchy}")
    
    # Analyze bone relationships and transforms like create_armature_nodes
    print(f"\n=== Bone Transform Analysis (create_armature_nodes logic) ===")
    
    scene = trimesh.Scene()
    
    for bone_name in bone_hierarchy:
        bone = bones[bone_name]
        print(f"\nProcessing bone: {bone_name}")
        print(f"  Body: {bone.body_name}")
        print(f"  Parent: {bone.parent_bone}")
        print(f"  World position: {bone.transform_matrix[:3, 3]}")
        
        # Apply the same logic as create_armature_nodes for position calculation
        if bone.parent_bone and bone.parent_bone in bones:
            # Child bone: position relative to parent
            parent_bone = bones[bone.parent_bone]
            
            # Get global transforms (these already include all intermediate bodies)
            bone_transform = bone.transform_matrix
            parent_transform = parent_bone.transform_matrix
            
            # Calculate relative position: difference in world positions
            bone_world_pos = bone_transform[:3, 3]
            parent_world_pos = parent_transform[:3, 3]
            
            # For GLTF bones, we need the local offset from parent to child
            translation = (bone_world_pos - parent_world_pos).tolist()
            
            print(f"  Parent world position: {parent_world_pos}")
            print(f"  Relative translation to parent: {translation}")
            print(f"  Distance from parent: {np.linalg.norm(translation)}")
        else:
            # Root bone: use world position
            translation = bone.transform_matrix[:3, 3].tolist()
            print(f"  Root bone translation: {translation}")
        
        # Create primitive for this bone
        parent_bone_obj = bones[bone.parent_bone] if bone.parent_bone and bone.parent_bone in bones else None
        joint_mesh, joint_transform = create_bone_primitive(bone_name, bone, parent_bone_obj)
        
        # Add joint to scene
        scene.add_geometry(joint_mesh, node_name=f"joint_{bone_name}", transform=joint_transform)
        
        # Create connection to parent if exists
        if parent_bone_obj:
            connection_mesh, connection_transform = create_bone_connection(parent_bone_obj, bone)
            if connection_mesh is not None:
                scene.add_geometry(connection_mesh, node_name=f"connection_{bone.parent_bone}_to_{bone_name}", 
                                 transform=connection_transform)
    
    # Add mesh positioning analysis if requested
    if include_meshes:
        print(f"\n=== Mesh Positioning Analysis (GLTFArmatureBuilder logic) ===")
        
        mesh_transforms = parser.get_mesh_transforms()
        mesh_count = 0
        
        for mesh_name, transforms in mesh_transforms.items():
            print(f"\nProcessing mesh: {mesh_name}")
            
            for i, (body_name, position, rotation_matrix, material) in enumerate(transforms):
                # Create instance name (same logic as GLTFArmatureBuilder)
                instance_name = f"{body_name}_{mesh_name}"
                if len(transforms) > 1:
                    instance_name += f"_{i}"
                
                print(f"  Instance: {instance_name}")
                print(f"    Original position: {position}")
                
                # Create transform matrix from parser data
                original_transform_matrix = np.eye(4)
                original_transform_matrix[:3, :3] = rotation_matrix
                original_transform_matrix[:3, 3] = position
                
                # Apply the same logic as GLTFArmatureBuilder.build_rigged_gltf lines 387-405
                final_transform_matrix = original_transform_matrix.copy()
                controlled_by_bone = False
                controlling_bone_name = None
                
                if instance_name in mesh_instance_weights and mesh_instance_weights[instance_name]:
                    # This mesh instance is controlled by a bone - make transform relative to bone
                    controlling_bones = list(mesh_instance_weights[instance_name].keys())
                    if controlling_bones:
                        controlling_bone_name = controlling_bones[0]  # Use first controlling bone
                        controlled_by_bone = True
                        print(f"    Controlled by bone: {controlling_bone_name}")
                        
                        if controlling_bone_name in bones:
                            bone = bones[controlling_bone_name]
                            
                            # Get bone transform (using same method as GLTFArmatureBuilder)
                            bone_transform = bone.transform_matrix
                            print(f"    Bone transform position: {bone_transform[:3, 3]}")
                            
                            # Make mesh transform relative to bone by removing bone's transform
                            # This matches the logic in GLTFArmatureBuilder lines 400-405
                            try:
                                bone_inverse = np.linalg.inv(bone_transform)
                                relative_transform = bone_inverse @ original_transform_matrix
                                
                                # Note: In the actual rigged GLB, this relative transform would be used
                                # and the bone would provide the positioning. But for visualization,
                                # we'll show what the final world position would be if we recomposed it.
                                final_transform_matrix = bone_transform @ relative_transform
                                
                                print(f"    Relative transform calculated")
                                print(f"    Final position (recomposed): {final_transform_matrix[:3, 3]}")
                                
                            except np.linalg.LinAlgError:
                                print(f"    Warning: Could not invert bone transform, using original")
                else:
                    print(f"    No bone control (unweighted)")
                
                # Create mesh primitive using actual STL
                original_position = original_transform_matrix[:3, 3]
                final_position = final_transform_matrix[:3, 3]
                
                mesh_primitive, mesh_transform = create_mesh_primitive(
                    instance_name, mesh_name, original_position, final_position, 
                    controlled_by_bone, parser, mesh_cache
                )
                
                scene.add_geometry(mesh_primitive, node_name=f"mesh_{instance_name}", transform=mesh_transform)
                
                # Create displacement line if there's a difference
                displacement_line, displacement_transform = create_mesh_displacement_line(
                    original_position, final_position
                )
                if displacement_line is not None:
                    scene.add_geometry(displacement_line, node_name=f"displacement_{instance_name}", 
                                     transform=displacement_transform)
                
                mesh_count += 1
        
        print(f"\nProcessed {mesh_count} mesh instances")
    
    # Add coordinate axes for reference
    axis_length = 0.1
    axis_radius = 0.002
    
    # X-axis (red)
    x_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length)
    x_axis.visual.vertex_colors = [255, 0, 0, 255]
    x_transform = np.eye(4)
    x_transform[:3, :3] = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])[:3, :3]
    x_transform[:3, 3] = [axis_length/2, 0, 0]
    scene.add_geometry(x_axis, node_name="x_axis", transform=x_transform)
    
    # Y-axis (green)
    y_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length)
    y_axis.visual.vertex_colors = [0, 255, 0, 255]
    y_transform = np.eye(4)
    y_transform[:3, :3] = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])[:3, :3]
    y_transform[:3, 3] = [0, axis_length/2, 0]
    scene.add_geometry(y_axis, node_name="y_axis", transform=y_transform)
    
    # Z-axis (blue)
    z_axis = trimesh.creation.cylinder(radius=axis_radius, height=axis_length)
    z_axis.visual.vertex_colors = [0, 0, 255, 255]
    z_transform = np.eye(4)
    z_transform[:3, 3] = [0, 0, axis_length/2]
    scene.add_geometry(z_axis, node_name="z_axis", transform=z_transform)
    
    # Export scene
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    exported = scene.export(file_type='glb')
    with open(output_path, 'wb') as f:
        f.write(exported)
    
    print(f"\n✅ Bone hierarchy visualization saved: {output_path}")
    
    # Create screenshot
    screenshot_path = output_path.with_suffix('.png')
    print(f"Creating screenshot: {screenshot_path}")
    try:
        render_glb_multiview(str(output_path), screenshot_path, view_width=400, view_height=300)
        print(f"✅ Screenshot saved: {screenshot_path}")
    except Exception as e:
        print(f"Warning: Could not create screenshot: {e}")
    
    # Print summary
    print(f"\n=== Visualization Summary ===")
    print(f"Red spheres: Root bones")
    print(f"Blue spheres: Child bones")
    print(f"Green cylinders: Bone connections")
    if include_meshes:
        print(f"Magenta STL meshes: Bone-controlled meshes")
        print(f"Yellow STL meshes: Uncontrolled meshes")
        print(f"Red cubes: Missing/error meshes (fallback)")
        print(f"Orange lines: Mesh displacement due to bone transforms")
        print(f"Loaded {len(mesh_cache)} unique STL files")
    print(f"RGB axes: Coordinate reference (X=red, Y=green, Z=blue)")
    print(f"Total bones: {len(bones)}")
    print(f"Total connections: {len([b for b in bones.values() if b.parent_bone])}")
    if include_meshes:
        print(f"Total mesh instances: {mesh_count}")


def main():
    """Test with different models and joint sets."""
    
    # Test 1: Simple MVP model with bones and meshes
    print("=== Test 1: MVP model with bones and meshes ===")
    visualize_bone_hierarchy(
        mjcf_path="g1_description/mvp_test.xml",
        output_path="output/bone_mesh_hierarchy_mvp.glb",
        target_joints=["waist_yaw_joint", "right_shoulder_pitch_joint"],
        include_meshes=True
    )
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Bones only for comparison
    print("=== Test 2: MVP model bones only ===")
    visualize_bone_hierarchy(
        mjcf_path="g1_description/mvp_test.xml",
        output_path="output/bone_hierarchy_mvp_bones_only.glb",
        target_joints=["waist_yaw_joint", "right_shoulder_pitch_joint"],
        include_meshes=False
    )
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Check what joints are available in full model
    print("=== Test 3: Full G1 model joint discovery ===")
    parser = MJCFParser("g1_description/g1_mjx_alt.xml")
    print("Available joints in g1_mjx_alt.xml:")
    for joint_name in sorted(parser.joints.keys()):
        print(f"  {joint_name}")
    
    # Test with a reasonable subset
    arm_joints = [
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint", 
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", 
        "right_elbow_joint",
    ]
    
    # Filter to only joints that actually exist
    existing_joints = [j for j in arm_joints if j in parser.joints]
    print(f"\nFiltered joints that exist: {existing_joints}")
    
    if existing_joints:
        print("\n=== Test 4: Full G1 model with arm joints and meshes ===")
        visualize_bone_hierarchy(
            mjcf_path="g1_description/g1_mjx_alt.xml",
            output_path="output/bone_mesh_hierarchy_g1_arms.glb",
            target_joints=existing_joints,
            include_meshes=True
        )


if __name__ == "__main__":
    main()
