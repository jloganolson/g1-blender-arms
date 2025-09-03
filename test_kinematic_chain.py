#!/usr/bin/env python3
"""
Kinematic Chain Validation Test

Tests that joint movements affect only the expected mesh parts in the kinematic chain.
Validates that vertex displacements are reasonable and follow expected patterns.
"""

import numpy as np
import trimesh
from mjcf_parser import MJCFParser
from armature_utils import ArmatureBuilder
from skinning_utils import SkinnedMeshBuilder
from pose_deformation import PoseDeformer
import sys
import os

def test_single_joint_movement(joint_name: str, angle_degrees: float = 45.0):
    """Test moving a single joint and verify only expected meshes are affected."""
    
    print(f"\n{'='*80}")
    print(f"TESTING JOINT: {joint_name} ({angle_degrees}°)")
    print(f"{'='*80}")
    
    # Use the main script's infrastructure by directly calling it
    # This ensures we use the same data flow as the actual system
    from mjcf_to_glb import mjcf_to_glb
    import tempfile
    import os
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp_file:
        temp_glb_path = tmp_file.name
    
    try:
        # Create a test pose with only this joint moved
        pose_angles = {joint_name: np.radians([angle_degrees, 0, 0])}
        
        # Run the main conversion with skinning but no test poses
        # We'll add our own custom pose testing here
        print(f"Running conversion with {joint_name} at {angle_degrees}°...")
        
        # For now, let's create a minimal test by modifying the pose creation
        posed_scene = create_test_scene_for_joint(joint_name, angle_degrees)
        
        return analyze_joint_movement(posed_scene, joint_name)
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_glb_path):
            os.unlink(temp_glb_path)

def create_test_scene_for_joint(joint_name: str, angle_degrees: float):
    """Create a test scene with just the specified joint moved."""
    
    # Parse MJCF
    mjcf_file = "g1_description/g1_mjx_alt.xml"
    parser = MJCFParser(mjcf_file)
    
    # Get mesh transforms and process simplified meshes (like main script)
    mesh_transforms = parser.get_mesh_transforms()
    simplified_meshes = {}
    
    # Process meshes like in the main script
    for mesh_name, mesh_instance_list in mesh_transforms.items():
        if mesh_name in parser.meshes:
            mesh_info = parser.meshes[mesh_name]
            mesh_path = parser.mjcf_path.parent / mesh_info.file
            
            try:
                mesh = trimesh.load_mesh(str(mesh_path))
                # Simplify mesh
                simplified = mesh.simplify_quadric_decimation(face_count=mesh.faces.shape[0] // 2)
                simplified_meshes[mesh_name] = simplified
            except Exception as e:
                print(f"Warning: Could not load mesh {mesh_name}: {e}")
                continue
    
    # Build armature and skinned scene
    armature_builder = ArmatureBuilder(parser)
    armature_builder.compute_bone_positions()
    skinned_builder = SkinnedMeshBuilder(parser, armature_builder)
    
    # Create base skinned scene
    base_scene = skinned_builder.create_skinned_scene(mesh_transforms, simplified_meshes)
    
    # Create pose with single joint movement
    pose_angles = {joint_name: np.radians([angle_degrees, 0, 0])}
    
    # Apply pose
    pose_deformer = PoseDeformer(armature_builder.bones)
    posed_scene = pose_deformer.create_posed_scene(base_scene, pose_angles)
    
    return posed_scene

def analyze_joint_movement(posed_scene, joint_name: str):
    """Analyze which meshes were affected by the joint movement."""
    
    print(f"\n📊 MESH ANALYSIS:")
    print(f"{'Mesh Name':<25} {'Body':<20} {'Vertices Moved':<15} {'Max Displacement':<18} {'Expected'}")
    print("-" * 95)
    
    expected_affected_meshes = get_expected_affected_meshes(joint_name)
    
    total_meshes = 0
    correctly_affected = 0
    incorrectly_affected = 0
    
    for geom_name, geom in posed_scene.geometry.items():
        if hasattr(geom, 'metadata') and 'body_name' in geom.metadata:
            body_name = geom.metadata['body_name']
            mesh_name = geom.metadata.get('mesh_name', 'unknown')
            
            # Check if mesh was deformed by looking at vertex displacement stats
            if hasattr(geom, 'metadata') and 'deformation_stats' in geom.metadata:
                stats = geom.metadata['deformation_stats']
                vertices_moved = stats.get('vertices_moved', 0)
                max_displacement = stats.get('max_displacement', 0.0)
                
                # Determine if this mesh should be affected
                should_be_affected = any(expected_mesh in mesh_name.lower() or 
                                       expected_mesh in body_name.lower() 
                                       for expected_mesh in expected_affected_meshes)
                
                is_affected = vertices_moved > 0
                status = "✅ CORRECT" if (is_affected == should_be_affected) else "❌ WRONG"
                
                if is_affected == should_be_affected:
                    correctly_affected += 1
                else:
                    incorrectly_affected += 1
                
                print(f"{mesh_name:<25} {body_name:<20} {vertices_moved:<15} {max_displacement:<18.4f} {status}")
                total_meshes += 1
    
    print(f"\n📈 SUMMARY:")
    print(f"Total meshes analyzed: {total_meshes}")
    print(f"Correctly affected/unaffected: {correctly_affected}")
    print(f"Incorrectly affected/unaffected: {incorrectly_affected}")
    
    if incorrectly_affected == 0:
        print(f"✅ PERFECT! All meshes behaved as expected.")
    else:
        print(f"⚠️  {incorrectly_affected} meshes had unexpected behavior.")
    
    return correctly_affected, incorrectly_affected

def get_expected_affected_meshes(joint_name: str) -> list:
    """Return list of mesh/body names that should be affected by this joint."""
    
    # Define kinematic chains - which parts move when each joint moves
    kinematic_chains = {
        # Wrist joint should only affect the hand/forearm
        'left_wrist_roll_joint': ['forearm', 'wrist', 'hand'],
        'right_wrist_roll_joint': ['forearm', 'wrist', 'hand'],
        
        # Elbow joint should affect forearm + everything beyond it
        'left_elbow_joint': ['forearm', 'wrist', 'hand', 'elbow'],
        'right_elbow_joint': ['forearm', 'wrist', 'hand', 'elbow'],
        
        # Shoulder joints should affect entire arm
        'left_shoulder_pitch_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'left_shoulder_roll_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'left_shoulder_yaw_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'right_shoulder_pitch_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'right_shoulder_roll_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'right_shoulder_yaw_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        
        # Leg joints
        'left_ankle_roll_joint': ['ankle', 'foot'],
        'right_ankle_roll_joint': ['ankle', 'foot'],
        'left_knee_joint': ['ankle', 'foot', 'knee'],
        'right_knee_joint': ['ankle', 'foot', 'knee'],
    }
    
    return kinematic_chains.get(joint_name, [])

def spot_check_vertex_positions(joint_name: str, angle_degrees: float = 45.0):
    """Spot check specific vertex positions to validate displacement math."""
    
    print(f"\n🔍 SPOT-CHECKING VERTEX POSITIONS for {joint_name}")
    print("-" * 60)
    
    # Parse MJCF and build scene
    mjcf_file = "g1_description/g1_mjx_alt.xml"
    parser = MJCFParser(mjcf_file)
    
    armature_builder = ArmatureBuilder(parser)
    armature_builder.compute_bone_positions()
    skinned_builder = SkinnedMeshBuilder(parser, armature_builder)
    
    # We need mesh_transforms and simplified_meshes (from the main script pattern)
    mesh_transforms = parser.get_mesh_transforms()
    simplified_meshes = parser.simplified_meshes
    
    # Get base scene and specific mesh
    base_scene = skinned_builder.create_skinned_scene(mesh_transforms, simplified_meshes)
    
    # Find a mesh that should be affected
    target_mesh_name = None
    expected_meshes = get_expected_affected_meshes(joint_name)
    
    for geom_name, geom in base_scene.geometry.items():
        if hasattr(geom, 'metadata') and 'mesh_name' in geom.metadata:
            mesh_name = geom.metadata['mesh_name']
            if any(expected in mesh_name.lower() for expected in expected_meshes):
                target_mesh_name = geom_name
                break
    
    if not target_mesh_name:
        print(f"⚠️  No suitable mesh found for {joint_name}")
        return
    
    # Get original mesh
    original_mesh = base_scene.geometry[target_mesh_name]
    original_vertices = original_mesh.vertices.copy()
    
    print(f"Target mesh: {original_mesh.metadata.get('mesh_name', 'unknown')}")
    print(f"Original mesh vertices: {len(original_vertices)}")
    
    # Apply pose
    pose_angles = {joint_name: np.radians([angle_degrees, 0, 0])}
    pose_deformer = PoseDeformer(armature_builder.bones)
    posed_scene = pose_deformer.create_posed_scene(base_scene, pose_angles)
    
    # Get deformed mesh
    deformed_mesh = posed_scene.geometry[target_mesh_name]
    deformed_vertices = deformed_mesh.vertices
    
    # Check specific vertices
    vertex_indices = [0, len(original_vertices)//4, len(original_vertices)//2, 
                     3*len(original_vertices)//4, len(original_vertices)-1]
    
    print(f"\nVertex displacement analysis:")
    print(f"{'Index':<8} {'Original Position':<25} {'New Position':<25} {'Displacement':<12}")
    print("-" * 80)
    
    total_displacement = 0
    vertices_checked = 0
    
    for idx in vertex_indices:
        if idx < len(original_vertices):
            orig_pos = original_vertices[idx]
            new_pos = deformed_vertices[idx]
            displacement = np.linalg.norm(new_pos - orig_pos)
            
            print(f"{idx:<8} {str(orig_pos):<25} {str(new_pos):<25} {displacement:<12.4f}")
            
            total_displacement += displacement
            vertices_checked += 1
    
    avg_displacement = total_displacement / vertices_checked if vertices_checked > 0 else 0
    
    print(f"\nAverage displacement of checked vertices: {avg_displacement:.4f} units")
    
    # Validate displacement is reasonable (should be within joint's reach)
    if 0.001 < avg_displacement < 0.5:  # 1mm to 50cm seems reasonable
        print(f"✅ Displacement is reasonable for a {angle_degrees}° joint rotation")
    elif avg_displacement <= 0.001:
        print(f"⚠️  Very small displacement - may not be visible")
    else:
        print(f"❌ Displacement seems too large - possible error")

def run_kinematic_chain_tests():
    """Run complete kinematic chain validation."""
    
    print("🧪 KINEMATIC CHAIN VALIDATION TEST")
    print("="*80)
    
    # Test sequence: wrist -> elbow -> shoulder (kinematic chain order)
    test_joints = [
        ('left_wrist_roll_joint', 45),
        ('left_elbow_joint', 45), 
        ('left_shoulder_pitch_joint', 45),
        ('right_wrist_roll_joint', 45),
    ]
    
    total_correct = 0
    total_wrong = 0
    
    for joint_name, angle in test_joints:
        correct, wrong = test_single_joint_movement(joint_name, angle)
        total_correct += correct
        total_wrong += wrong
        
        # Spot check this joint
        spot_check_vertex_positions(joint_name, angle)
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Total meshes correctly handled: {total_correct}")
    print(f"Total meshes with errors: {total_wrong}")
    
    if total_wrong == 0:
        print(f"🎉 ALL TESTS PASSED! Kinematic chain is working perfectly!")
    else:
        print(f"⚠️  {total_wrong} issues found that need investigation.")

if __name__ == "__main__":
    run_kinematic_chain_tests()
