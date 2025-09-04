#!/usr/bin/env python3
"""
Test Multi-Joint Rigging
Test script to test the new multi-joint rigging functionality, including
the waist + right shoulder pitch combination and hierarchical joint discovery.
"""

from pathlib import Path
from armature_exporter.rigged_glb_exporter import (
    create_waist_and_shoulder_rig, 
    create_hierarchical_rig
)
from armature_exporter.gltf_armature_builder import (
    create_rigged_waist_and_shoulder_glb,
    create_rigged_hierarchical_glb
)


def test_waist_and_shoulder_rig():
    """Test the waist + right shoulder pitch rig (as requested)."""
    print("🧪 Testing Waist + Right Shoulder Pitch Rig")
    print("=" * 50)
    
    # Test 1: Simple exporter (metadata-based)
    print("\n1️⃣  Testing Simple Waist + Shoulder Exporter...")
    try:
        create_waist_and_shoulder_rig(output_name="test_waist_shoulder_simple")
        print("✅ Simple waist + shoulder export successful")
    except Exception as e:
        print(f"❌ Simple waist + shoulder export failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Full GLTF armature builder
    print("\n2️⃣  Testing Full GLTF Waist + Shoulder Armature...")
    try:
        create_rigged_waist_and_shoulder_glb(output_name="test_waist_shoulder_full")
        print("✅ Full GLTF waist + shoulder export successful")
    except Exception as e:
        print(f"❌ Full GLTF waist + shoulder export failed: {e}")
        import traceback
        traceback.print_exc()


def test_hierarchical_joint_discovery():
    """Test the hierarchical joint discovery approach."""
    print("\n🌳 Testing Hierarchical Joint Discovery")
    print("=" * 50)
    
    # Test 1: Simple exporter with hierarchy discovery
    print("\n1️⃣  Testing Simple Hierarchical Exporter (5 joints max)...")
    try:
        create_hierarchical_rig(output_name="test_hierarchical_simple", max_joints=5)
        print("✅ Simple hierarchical export successful")
    except Exception as e:
        print(f"❌ Simple hierarchical export failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Full GLTF armature with hierarchy discovery
    print("\n2️⃣  Testing Full GLTF Hierarchical Armature (3 joints max)...")
    try:
        create_rigged_hierarchical_glb(output_name="test_hierarchical_full", max_joints=3)
        print("✅ Full GLTF hierarchical export successful")
    except Exception as e:
        print(f"❌ Full GLTF hierarchical export failed: {e}")
        import traceback
        traceback.print_exc()


def print_blender_test_instructions():
    """Print testing instructions for the new multi-joint rigs."""
    instructions = """
🔹 MULTI-JOINT BLENDER TESTING INSTRUCTIONS 🔹

1. Open Blender (version 2.8+ recommended)

2. Import the multi-joint rigged GLB:
   • File → Import → glTF 2.0 (.glb/.gltf)
   • Try these files:
     - 'test_waist_shoulder_full.glb' (waist + right shoulder)
     - 'test_hierarchical_full.glb' (auto-discovered joints)

3. Verify the import:
   • Check the Outliner - you should see:
     - Multiple mesh objects (robot parts)
     - Armature object (skeleton)
     - Multiple bones (waist_yaw_joint, right_shoulder_pitch_joint, etc.)

4. Test the joints:
   • Select the Armature object
   • Switch to "Pose Mode" (Ctrl+Tab)
   
   For waist joint:
   • Select "waist_yaw_joint" bone
   • Rotate around Z-axis (R + Z keys)
   • Upper body should rotate
   
   For right shoulder joint:
   • Select "right_shoulder_pitch_joint" bone
   • Rotate around Y-axis (R + Y keys)
   • Right arm should move independently

5. Expected behavior:
   ✅ Each joint moves the appropriate body parts
   ✅ Joint hierarchy is preserved (shoulder moves with waist)
   ✅ Independent control of individual joints
   ✅ Smooth deformation without artifacts

6. Advanced testing:
   • Try rotating waist first, then shoulder - shoulder should move relative to waist
   • Test joint limits and natural motion ranges
   • Check weight painting in Weight Paint mode

🚀 SUCCESS CRITERIA:
Multiple joints should provide independent control while maintaining
proper parent-child relationships in the kinematic chain!
"""
    print(instructions)


def list_output_files():
    """List all generated output files."""
    print("\n📁 Generated Output Files:")
    output_dir = Path("output")
    
    # Group files by type
    glb_files = []
    json_files = []
    png_files = []
    
    for file in sorted(output_dir.glob("test_*")):
        size_mb = file.stat().st_size / (1024 * 1024)
        file_info = f"{file.name} ({size_mb:.1f} MB)"
        
        if file.suffix == '.glb':
            glb_files.append(file_info)
        elif file.suffix == '.json':
            json_files.append(file_info)
        elif file.suffix == '.png':
            png_files.append(file_info)
    
    if glb_files:
        print("\n🎯 GLB Files (Rigged Models):")
        for file_info in glb_files:
            print(f"   {file_info}")
    
    if json_files:
        print("\n📄 JSON Files (Metadata):")
        for file_info in json_files:
            print(f"   {file_info}")
    
    if png_files:
        print("\n🖼️ PNG Files (Legacy Previews - no longer generated):")
        for file_info in png_files:
            print(f"   {file_info}")


if __name__ == "__main__":
    print("🧪 MULTI-JOINT RIGGING TEST SUITE")
    print("=" * 60)
    
    # Test the specific request: waist + right shoulder
    test_waist_and_shoulder_rig()
    
    # Test the hierarchical approach
    test_hierarchical_joint_discovery()
    
    # List all outputs
    list_output_files()
    
    # Print testing instructions
    print_blender_test_instructions()
    
    print("\n" + "=" * 60)
    print("🎉 Multi-joint rigging tests complete!")
    print("🎯 Next: Test the GLB files in Blender to verify joint functionality")
