#!/usr/bin/env python3
"""
Simple Export Test
Test script for the simplified rigged/unrigged GLB export system.
"""

from pathlib import Path
from armature_exporter.utils_3d import mjcf_to_glb
from armature_exporter.gltf_armature_builder import create_rigged_full_body_glb


def test_unrigged_export():
    """Test the unrigged GLB export (meshes only)."""
    print("📦 Testing Unrigged GLB Export")
    print("=" * 40)
    
    try:
        mjcf_path = "./g1_description/g1_mjx_alt.xml"
        output_path = "output/robot_unrigged.glb"
        mjcf_to_glb(mjcf_path, output_path)
        print("✅ Unrigged export successful")
    except Exception as e:
        print(f"❌ Unrigged export failed: {e}")
        import traceback
        traceback.print_exc()


def test_rigged_export():
    """Test the rigged GLB export (with armature)."""
    print("\n🦴 Testing Rigged GLB Export")
    print("=" * 40)
    
    try:
        create_rigged_full_body_glb(output_name="robot_rigged")
        print("✅ Rigged export successful")
    except Exception as e:
        print(f"❌ Rigged export failed: {e}")
        import traceback
        traceback.print_exc()


def print_usage_instructions():
    """Print instructions for using the exported models."""
    instructions = """
🎯 USAGE INSTRUCTIONS

📦 UNRIGGED GLB (robot_unrigged.glb):
   • Contains: All robot meshes in correct positions
   • Use for: Static models, reference geometry, non-animated scenes
   • Import into: Blender, Unity, web viewers, etc.

🦴 RIGGED GLB (robot_rigged.glb):
   • Contains: Full robot with 23-joint armature/skeleton
   • Use for: Animation, posing, interactive models
   • Features: Complete kinematic chains (waist→shoulders→elbows→wrists, waist→hips→knees→ankles)
   • Import into: Blender for animation work

📋 BLENDER TESTING:
   1. Open Blender
   2. File → Import → glTF 2.0 (.glb/.gltf)
   3. Select robot_rigged.glb
   4. Switch to Pose Mode (Ctrl+Tab)
   5. Select bones and rotate (R key)
   6. Test joint hierarchy and movement

🚀 SUCCESS CRITERIA:
   ✅ Unrigged: All meshes visible in correct positions
   ✅ Rigged: Functional armature with 23 controllable joints
   ✅ Rigged: Proper parent-child bone relationships
   ✅ Rigged: Smooth mesh deformation when rotating joints
"""
    print(instructions)


def list_output_files():
    """List generated output files."""
    print("\n📁 Generated Files:")
    output_dir = Path("output")
    
    for file in sorted(output_dir.glob("robot_*")):
        size_mb = file.stat().st_size / (1024 * 1024)
        file_type = "Rigged" if "rigged" in file.name else "Unrigged"
        print(f"   {file.name} ({size_mb:.1f} MB) - {file_type}")


if __name__ == "__main__":
    print("🧪 SIMPLIFIED GLB EXPORT TEST")
    print("=" * 50)
    
    # Test unrigged export
    test_unrigged_export()
    
    # Test rigged export
    test_rigged_export()
    
    # List outputs
    list_output_files()
    
    # Print usage instructions
    print_usage_instructions()
    
    print("\n" + "=" * 50)
    print("🎉 Export tests complete!")
    print("🎯 Two clean options: robot_unrigged.glb and robot_rigged.glb")
