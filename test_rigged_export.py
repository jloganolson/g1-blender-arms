#!/usr/bin/env python3
"""
Test Rigged Export
Test script to verify the rigged GLB export functionality and provide instructions
for testing in Blender.
"""

from pathlib import Path
from armature_exporter.gltf_armature_builder import create_rigged_waist_glb
from armature_exporter.rigged_glb_exporter import create_simple_waist_rig


def test_both_exporters():
    """Test both the simple rigged exporter and full GLTF armature builder."""
    print("🧪 Testing Rigged GLB Export Systems")
    print("=" * 50)
    
    # Test 1: Simple rigged exporter (metadata only)
    print("\n1️⃣  Testing Simple Rigged Exporter...")
    try:
        create_simple_waist_rig(output_name="test_simple_rig")
        print("✅ Simple rigged export successful")
    except Exception as e:
        print(f"❌ Simple rigged export failed: {e}")
    
    # Test 2: Full GLTF armature builder
    print("\n2️⃣  Testing Full GLTF Armature Builder...")
    try:
        create_rigged_waist_glb(output_name="test_full_rig")
        print("✅ Full GLTF armature export successful")
    except Exception as e:
        print(f"❌ Full GLTF armature export failed: {e}")
    
    # List outputs
    print("\n📁 Output Files:")
    output_dir = Path("output")
    for file in sorted(output_dir.glob("test_*")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {file.name} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps: Test in Blender")
    print_blender_test_instructions()


def print_blender_test_instructions():
    """Print detailed instructions for testing in Blender."""
    instructions = """
🔹 BLENDER TESTING INSTRUCTIONS 🔹

1. Open Blender (version 2.8+ recommended)

2. Import the rigged GLB:
   • File → Import → glTF 2.0 (.glb/.gltf)
   • Select 'test_full_rig.glb' or 'rigged_waist_robot.glb'
   • Click "Import glTF 2.0"

3. Verify the import:
   • The robot should appear in the 3D viewport
   • Check the Outliner - you should see:
     - Mesh objects (robot parts)
     - Armature object (skeleton)
     - Bone: "waist_yaw_joint"

4. Test the waist joint:
   • Select the Armature object
   • Switch to "Pose Mode" (Ctrl+Tab or mode dropdown)
   • Select the "waist_yaw_joint" bone
   • Rotate the bone (R key, then Z to constrain to Z-axis)
   • The upper body (torso, arms, head) should rotate
   • The legs should remain stationary

5. Expected behavior:
   ✅ Upper body parts follow the waist rotation
   ✅ Lower body parts stay fixed
   ✅ Smooth deformation (no tears or gaps)
   ✅ Bone appears as a proper Blender armature

6. If issues occur:
   • Check if vertex weights are applied (Weight Paint mode)
   • Verify the armature has proper parent-child relationships
   • Ensure skins are properly assigned to meshes

📄 Additional files created:
   • test_simple_rig.json - Rigging metadata for analysis
   • PNG previews - Multi-view renders of the robot

🚀 SUCCESS CRITERIA:
The waist joint should allow you to pose the robot's upper body
independently of the lower body, creating a functional rig for
animation and posing in Blender!
"""
    print(instructions)


def create_advanced_rig_example():
    """Example of how to create a more complex rig with multiple joints."""
    print("\n🔧 ADVANCED RIG EXAMPLE")
    print("To create a rig with multiple joints, modify the joint selection:")
    
    example_code = '''
# Example: Create rig with waist + shoulder joints
from armature_exporter.rigged_glb_exporter import RiggedGLBExporter
from armature_exporter.gltf_armature_builder import GLTFArmatureBuilder

exporter = RiggedGLBExporter("./g1_description/g1_mjx_alt.xml")

# Multiple joints for more complex rig
exporter.set_target_joints([
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", 
    "right_shoulder_pitch_joint"
])

# Build and export
exporter._build_bone_hierarchy()
exporter._assign_simple_weights()

builder = GLTFArmatureBuilder(exporter)
builder.build_rigged_gltf("output/multi_joint_rig.glb")
'''
    print(example_code)


if __name__ == "__main__":
    test_both_exporters()
    create_advanced_rig_example()
