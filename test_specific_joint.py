#!/usr/bin/env python3
"""
Specific Joint Movement Test

Tests one joint at a time by modifying the pose definitions and running the main script.
"""

import os
import tempfile
import subprocess
import sys

def create_single_joint_pose_file(joint_name: str, angle_degrees: float):
    """Create a modified skinning_utils.py with a single joint pose."""
    
    pose_code = f'''
def create_test_poses():
    """Test poses for skinned mesh validation."""
    import numpy as np
    
    test_poses = {{
        "single_joint": {{
            "{joint_name}": np.array([{angle_degrees}, 0, 0])  # degrees
        }}
    }}
    
    return test_poses
'''
    
    return pose_code

def test_joint_movement(joint_name: str, angle_degrees: float = 45):
    """Test movement of a specific joint."""
    
    print(f"\n{'='*80}")
    print(f"TESTING JOINT: {joint_name} at {angle_degrees}°")
    print(f"{'='*80}")
    
    # Backup original skinning_utils.py
    backup_file = "skinning_utils_backup.py"
    
    try:
        # Create backup
        subprocess.run(["cp", "skinning_utils.py", backup_file], check=True)
        
        # Read original file
        with open("skinning_utils.py", "r") as f:
            original_content = f.read()
        
        # Replace the create_test_poses function
        pose_function = create_single_joint_pose_file(joint_name, angle_degrees)
        
        # Find and replace the function
        import re
        pattern = r'def create_test_poses\(\):.*?return test_poses'
        modified_content = re.sub(pattern, pose_function.strip(), original_content, flags=re.DOTALL)
        
        # Write modified file
        with open("skinning_utils.py", "w") as f:
            f.write(modified_content)
        
        # Run the main script
        print(f"Running main script with {joint_name} pose...")
        result = subprocess.run([
            "python", "mjcf_to_glb.py", 
            "--skinning", 
            "--test-poses", 
            "-o", f"test_{joint_name}.glb"
        ], 
        capture_output=True, 
        text=True,
        env={**os.environ, "PYTHONPATH": "."})
        
        if result.returncode != 0:
            print(f"❌ Script failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        # Analyze output
        return analyze_output(result.stdout, joint_name, angle_degrees)
        
    finally:
        # Restore original file
        if os.path.exists(backup_file):
            subprocess.run(["mv", backup_file, "skinning_utils.py"], check=True)
        
        # Clean up test file
        test_file = f"test_{joint_name}.glb"
        if os.path.exists(test_file):
            os.unlink(test_file)

def analyze_output(output: str, joint_name: str, angle_degrees: float) -> bool:
    """Analyze the command output to verify joint movement."""
    
    print(f"🔍 ANALYZING OUTPUT FOR {joint_name}")
    print("-" * 60)
    
    # Check if the joint was actually rotated
    if f"Applied {abs(angle_degrees):.1f}° rotation to {joint_name}" in output:
        print(f"✅ Joint {joint_name} was rotated {angle_degrees}°")
        joint_moved = True
    else:
        print(f"❌ Joint {joint_name} was NOT rotated")
        joint_moved = False
    
    # Count meshes that were deformed
    import re
    
    # Find all deformation results
    deformation_blocks = re.findall(
        r'Processing skinned mesh: (\w+)\s+Body: (\w+)\s+Mesh: (\w+).*?Deformation results:.*?Vertices moved: (\d+).*?Maximum displacement: ([\d.]+)',
        output,
        re.DOTALL
    )
    
    moved_meshes = []
    static_meshes = []
    
    for geometry, body, mesh, vertices_moved, max_displacement in deformation_blocks:
        vertices_moved = int(vertices_moved)
        max_displacement = float(max_displacement)
        
        if vertices_moved > 0:
            moved_meshes.append((mesh, body, vertices_moved, max_displacement))
            print(f"🟢 MOVED: {mesh} ({body}) - {vertices_moved} vertices, {max_displacement:.4f} displacement")
        else:
            static_meshes.append((mesh, body))
            print(f"⚪ STATIC: {mesh} ({body})")
    
    # Determine expected kinematic chain
    expected_parts = get_kinematic_chain(joint_name)
    
    print(f"\n📊 KINEMATIC ANALYSIS:")
    print(f"Expected moving parts: {expected_parts}")
    print(f"Actually moved: {len(moved_meshes)} meshes")
    print(f"Remained static: {len(static_meshes)} meshes")
    
    # Check if the right parts moved
    kinematic_correct = True
    
    for mesh, body, vertices, displacement in moved_meshes:
        should_move = any(part in mesh.lower() or part in body.lower() for part in expected_parts)
        if should_move:
            print(f"✅ CORRECT: {mesh} should move (in kinematic chain)")
        else:
            print(f"❌ ERROR: {mesh} moved but shouldn't (not in kinematic chain)")
            kinematic_correct = False
    
    for mesh, body in static_meshes:
        should_move = any(part in mesh.lower() or part in body.lower() for part in expected_parts)
        if not should_move:
            print(f"✅ CORRECT: {mesh} correctly static")
        else:
            print(f"❌ ERROR: {mesh} should move but didn't")
            kinematic_correct = False
    
    # Check displacement magnitudes are reasonable
    displacement_reasonable = True
    for mesh, body, vertices, displacement in moved_meshes:
        if displacement < 0.001:
            print(f"⚠️  {mesh}: Very small displacement ({displacement:.6f})")
        elif displacement > 0.5:
            print(f"⚠️  {mesh}: Very large displacement ({displacement:.6f})")
            displacement_reasonable = False
        else:
            print(f"✅ {mesh}: Reasonable displacement ({displacement:.4f})")
    
    # Overall assessment
    success = joint_moved and kinematic_correct and displacement_reasonable
    
    print(f"\n🎯 TEST RESULT:")
    print(f"Joint moved: {'✅' if joint_moved else '❌'}")
    print(f"Kinematic chain: {'✅' if kinematic_correct else '❌'}")
    print(f"Displacement reasonable: {'✅' if displacement_reasonable else '❌'}")
    print(f"Overall: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success

def get_kinematic_chain(joint_name: str) -> list:
    """Get the parts that should move when this joint moves."""
    
    chains = {
        'left_wrist_roll_joint': ['forearm', 'wrist', 'hand'],
        'left_elbow_joint': ['forearm', 'wrist', 'hand', 'elbow'],
        'left_shoulder_pitch_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'left_shoulder_roll_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
        'right_wrist_roll_joint': ['forearm', 'wrist', 'hand'],
        'right_elbow_joint': ['forearm', 'wrist', 'hand', 'elbow'],
        'right_shoulder_pitch_joint': ['forearm', 'wrist', 'hand', 'elbow', 'shoulder'],
    }
    
    return chains.get(joint_name, [])

def run_focused_kinematic_tests():
    """Run focused kinematic tests on key joints."""
    
    print("🧪 FOCUSED KINEMATIC CHAIN VALIDATION")
    print("Testing specific joints to verify end-effector -> proximal behavior")
    print("="*80)
    
    # Test in kinematic order: end effector -> mid chain -> proximal
    test_sequence = [
        ("left_wrist_roll_joint", 45, "End effector test"),
        ("left_elbow_joint", 45, "Mid-chain test"), 
        ("left_shoulder_pitch_joint", 45, "Proximal joint test"),
    ]
    
    results = []
    
    for joint_name, angle, description in test_sequence:
        print(f"\n🎯 {description.upper()}")
        success = test_joint_movement(joint_name, angle)
        results.append((joint_name, success))
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n🏆 FINAL RESULTS:")
    print("="*50)
    for joint, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{joint}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL KINEMATIC TESTS PASSED!")
        print("The rigging system correctly implements kinematic chains! 🚀")
    else:
        print(f"⚠️  {total-passed} test(s) failed - kinematic behavior needs fixes")
    
    return passed == total

if __name__ == "__main__":
    success = run_focused_kinematic_tests()
    sys.exit(0 if success else 1)
