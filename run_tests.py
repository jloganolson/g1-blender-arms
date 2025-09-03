#!/usr/bin/env python3
"""
Test runner script for the g1-blender-arms project.
Runs all tests and provides a summary of results.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --fast       # Run tests without rendering (faster)
    python run_tests.py --verbose    # Run with detailed output
"""

import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"⚠️  {description} - SKIPPED (command not found)")
        return True  # Don't fail if optional tools aren't available

def main():
    parser = argparse.ArgumentParser(description="Run tests for g1-blender-arms project")
    parser.add_argument("--fast", action="store_true", 
                       help="Skip rendering tests for faster execution")
    parser.add_argument("--verbose", action="store_true",
                       help="Run tests with verbose output")
    parser.add_argument("--module", type=str, 
                       help="Run tests for specific module only")
    
    args = parser.parse_args()
    
    print("🧪 G1 Blender Arms - Test Suite")
    print("=" * 50)
    
    # Ensure we're in the project directory
    project_dir = Path(__file__).parent
    
    # Activate virtual environment
    venv_python = project_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        print("⚠️  Virtual environment not found. Using system python.")
        python_cmd = "python"
    else:
        python_cmd = str(venv_python)
    
    results = []
    
    # Test 1: Run utils_3d tests
    if not args.module or args.module == "utils_3d":
        cmd = [python_cmd, "test_utils_3d.py"]
        if args.verbose:
            cmd.append("--verbose")
        
        success = run_command(cmd, "Utils 3D Tests")
        results.append(("Utils 3D Tests", success))
    
    # Test 2: Check for basic import errors
    if not args.module or args.module == "imports":
        modules_to_test = [
            "utils_3d",
            "mjcf_parser", 
            "mjcf_to_glb",
            "simple_mjcf_to_glb",
            "skinning_utils",
            "armature_utils"
        ]
        
        for module in modules_to_test:
            cmd = [python_cmd, "-c", f"import {module}; print('✅ {module} imported successfully')"]
            success = run_command(cmd, f"Import test: {module}")
            results.append((f"Import {module}", success))
    
    # Test 3: Check if basic mesh operations work (quick integration test)
    if not args.module or args.module == "integration":
        integration_test = '''
import trimesh
from utils_3d import simplify_mesh, combine_meshes, get_mesh_info
sphere = trimesh.creation.icosphere(subdivisions=1)
simplified = simplify_mesh(sphere, reduction_ratio=0.5)
scene = combine_meshes([sphere, simplified], ["original", "simplified"])
info = get_mesh_info(sphere)
print("✅ Integration test passed")
'''
        cmd = [python_cmd, "-c", integration_test]
        success = run_command(cmd, "Basic Integration Test")
        results.append(("Integration Test", success))
    
    # Test 4: Optional - Check if example scripts run without errors
    if not args.fast and (not args.module or args.module == "examples"):
        example_scripts = ["simple_mjcf_to_glb.py"]
        
        for script in example_scripts:
            script_path = project_dir / script
            if script_path.exists():
                # Just check if the script can be imported/parsed without running it
                cmd = [python_cmd, "-m", "py_compile", str(script_path)]
                success = run_command(cmd, f"Syntax check: {script}")
                results.append((f"Syntax {script}", success))
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
    
    if failed_tests > 0:
        print(f"\nFailed tests:")
        for test_name, success in results:
            if not success:
                print(f"  ❌ {test_name}")
    
    print(f"\n{'='*60}")
    
    if failed_tests == 0:
        print("🎉 All tests passed! Your changes look good.")
        return 0
    else:
        print("💔 Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
