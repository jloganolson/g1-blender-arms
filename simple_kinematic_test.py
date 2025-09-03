#!/usr/bin/env python3
"""
Simple Kinematic Chain Test

Tests specific joint movements by running the existing system with single joint poses
and analyzing the results from the CLI output.
"""

import subprocess
import sys
import os
import tempfile
import re

def test_single_joint(joint_name: str, angle: float = 45):
    """Test a single joint movement and analyze the results."""
    
    print(f"\n{'='*80}")
    print(f"TESTING: {joint_name} at {angle}°")
    print(f"{'='*80}")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp_file:
        temp_glb_path = tmp_file.name
    
    try:
        # Create custom pose test by modifying the test poses
        test_pose_script = create_custom_pose_test(joint_name, angle, temp_glb_path)
        
        # Run the test
        result = subprocess.run([
            'python', test_pose_script
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"❌ Test failed with error:")
            print(result.stderr)
            return False, 0, 0
        
        # Analyze the output
        return analyze_deformation_output(result.stdout, joint_name)
        
    finally:
        # Clean up
        if os.path.exists(temp_glb_path):
            os.unlink(temp_glb_path)
        if 'test_pose_script' in locals() and os.path.exists(test_pose_script):
            os.unlink(test_pose_script)

def create_custom_pose_test(joint_name: str, angle: float, output_path: str) -> str:
    """Create a custom Python script that tests only one joint."""
    
    script_content = f'''#!/usr/bin/env python3
import sys
sys.path.append('.')
from mjcf_to_glb import mjcf_to_glb
import tempfile

# Test pose with only {joint_name}
def create_test_poses():
    """Custom test poses for kinematic validation."""
    import numpy as np
    
    test_poses = {{
        "single_joint_test": {{
            "{joint_name}": np.array([{angle}, 0, 0])  # degrees
        }}
    }}
    
    return test_poses

# Monkey patch the test poses
import skinning_utils
skinning_utils.create_test_poses = create_test_poses

# Run conversion with skinning and test poses
if __name__ == "__main__":
    mjcf_to_glb(
        mjcf_file="g1_description/g1_mjx_alt.xml",
        output_file="{output_path}",
        include_armature=False,
        include_skinning=True,
        create_test_poses_flag=True
    )
'''
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        return f.name

def analyze_deformation_output(output: str, joint_name: str) -> tuple:
    """Analyze the console output to determine which meshes were affected."""
    
    print("🔍 ANALYZING DEFORMATION RESULTS")
    print("-" * 50)
    
    # Extract deformation results from the output
    deformation_blocks = re.findall(
        r'Processing skinned mesh: (.*?)\n.*?Body: (.*?)\n.*?Mesh: (.*?)\n.*?Deformation results:\n(.*?)(?=Processing|$)', 
        output, 
        re.DOTALL
    )
    
    affected_meshes = []
    unaffected_meshes = []
    total_meshes = 0
    
    # Parse each deformation block
    for block in re.findall(r'Deformation results:\n(.*?)(?=Processing|\n\n|$)', output, re.DOTALL):
        total_meshes += 1
        
        # Look for vertices moved
        vertices_moved_match = re.search(r'Vertices moved: (\d+)', block)
        max_displacement_match = re.search(r'Maximum displacement: ([\d.]+)', block)
        
        if vertices_moved_match and max_displacement_match:
            vertices_moved = int(vertices_moved_match.group(1))
            max_displacement = float(max_displacement_match.group(1))
            
            # Get mesh info from preceding lines
            mesh_info_match = re.search(r'Processing skinned mesh: (.*?)\n.*?Body: (.*?)\n.*?Mesh: (.*?)\n', 
                                      output[:output.find(block)])
            
            if mesh_info_match:
                mesh_name = mesh_info_match.group(3)
                body_name = mesh_info_match.group(2)
                
                if vertices_moved > 0:
                    affected_meshes.append((mesh_name, body_name, vertices_moved, max_displacement))
                    print(f"✅ MOVED: {mesh_name} ({body_name}) - {vertices_moved} vertices, {max_displacement:.4f} max displacement")
                else:
                    unaffected_meshes.append((mesh_name, body_name))
                    print(f"⚪ STATIC: {mesh_name} ({body_name}) - no movement")
    
    # Determine expected behavior
    expected_meshes = get_expected_kinematic_chain(joint_name)
    
    print(f"\n📊 KINEMATIC CHAIN ANALYSIS:")
    print(f"Joint tested: {joint_name}")
    print(f"Expected affected parts: {expected_meshes}")
    print(f"Actually affected meshes: {len(affected_meshes)}")
    print(f"Static meshes: {len(unaffected_meshes)}")
    
    # Validate kinematic chain
    correct_count = 0
    error_count = 0
    
    for mesh_name, body_name, vertices_moved, max_displacement in affected_meshes:
        should_move = any(expected in mesh_name.lower() or expected in body_name.lower() 
                         for expected in expected_meshes)
        
        if should_move:
            print(f"✅ CORRECT: {mesh_name} should move (kinematic chain)")
            correct_count += 1
        else:
            print(f"❌ ERROR: {mesh_name} shouldn't move (not in kinematic chain)")
            error_count += 1
    
    for mesh_name, body_name in unaffected_meshes:
        should_move = any(expected in mesh_name.lower() or expected in body_name.lower() 
                         for expected in expected_meshes)
        
        if not should_move:
            print(f"✅ CORRECT: {mesh_name} correctly static")
            correct_count += 1
        else:
            print(f"❌ ERROR: {mesh_name} should move but didn't")
            error_count += 1
    
    # Check displacement magnitudes
    displacement_ok = True
    for mesh_name, body_name, vertices_moved, max_displacement in affected_meshes:
        if max_displacement < 0.001:
            print(f"⚠️  WARNING: {mesh_name} displacement very small ({max_displacement:.6f})")
        elif max_displacement > 0.5:
            print(f"⚠️  WARNING: {mesh_name} displacement very large ({max_displacement:.6f})")
            displacement_ok = False
        else:
            print(f"✅ GOOD: {mesh_name} displacement reasonable ({max_displacement:.4f})")
    
    success = error_count == 0 and displacement_ok
    
    print(f"\n🎯 FINAL RESULT:")
    if success:
        print(f"✅ PERFECT! Kinematic chain working correctly")
    else:
        print(f"❌ Issues found: {error_count} incorrect behaviors")
    
    return success, correct_count, error_count

def get_expected_kinematic_chain(joint_name: str) -> list:
    """Return parts that should move when this joint moves."""
    
    kinematic_chains = {
        # End effectors - should only affect immediate parts
        'left_wrist_roll_joint': ['forearm', 'wrist', 'hand'],
        'right_wrist_roll_joint': ['forearm', 'wrist', 'hand'],
        'left_ankle_roll_joint': ['ankle', 'foot'],
        'right_ankle_roll_joint': ['ankle', 'foot'],
        
        # Mid-chain joints - should affect parts beyond them
        'left_elbow_joint': ['forearm', 'wrist', 'hand', 'elbow'],
        'right_elbow_joint': ['forearm', 'wrist', 'hand', 'elbow'],
        'left_knee_joint': ['ankle', 'foot', 'knee'],
        'right_knee_joint': ['ankle', 'foot', 'knee'],
        
        # Proximal joints - should affect entire limb
        'left_shoulder_pitch_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'left_shoulder_roll_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'left_shoulder_yaw_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'right_shoulder_pitch_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'right_shoulder_roll_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'right_shoulder_yaw_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
    }
    
    return kinematic_chains.get(joint_name, [])

def run_kinematic_tests():
    """Run the kinematic chain validation tests."""
    
    print("🧪 KINEMATIC CHAIN VALIDATION")
    print("Testing individual joints to verify proper kinematic behavior")
    print("="*80)
    
    # Test sequence: end effector -> mid-chain -> proximal
    test_joints = [
        ('left_wrist_roll_joint', 45),      # End effector
        ('left_elbow_joint', 45),           # Mid-chain  
        ('left_shoulder_pitch_joint', 45),  # Proximal
        ('right_wrist_roll_joint', -45),    # Test other side
    ]
    
    total_success = 0
    total_tests = len(test_joints)
    
    for joint_name, angle in test_joints:
        success, correct, errors = test_single_joint(joint_name, angle)
        if success:
            total_success += 1
    
    print(f"\n🏆 OVERALL RESULTS:")
    print(f"Tests passed: {total_success}/{total_tests}")
    
    if total_success == total_tests:
        print("🎉 ALL KINEMATIC TESTS PASSED! The rigging system is working perfectly!")
    else:
        print(f"⚠️  {total_tests - total_success} tests failed. The kinematic chain needs fixes.")
    
    return total_success == total_tests

if __name__ == "__main__":
    # Activate virtual environment first
    if not os.environ.get('VIRTUAL_ENV'):
        print("Activating virtual environment...")
        os.system("source .venv/bin/activate")
    
    success = run_kinematic_tests()
    sys.exit(0 if success else 1)
