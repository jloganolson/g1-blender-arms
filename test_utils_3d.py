#!/usr/bin/env python3
"""
Unit tests for utils_3d module.
Tests all functionality including mesh loading, simplification, rendering, and export.
Run with: python -m pytest test_utils_3d.py -v
Or: python test_utils_3d.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import trimesh

# Import the module we're testing
from utils_3d import (
    simplify_mesh, load_stl, combine_meshes, export_scene_to_glb, export_mesh_to_glb,
    render_mesh, render_scene, get_mesh_info, print_mesh_info, multi_stl_to_glb,
    render_scene_multiview
)


class TestUtils3D(unittest.TestCase):
    """Test cases for utils_3d module."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create test meshes
        self.sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        self.cube = trimesh.creation.box(extents=[1, 1, 1])
        self.cylinder = trimesh.creation.cylinder(radius=0.5, height=2.0)
        
        # Create a temporary STL file for testing
        self.test_stl_path = self.test_dir / "test_mesh.stl"
        self.sphere.export(str(self.test_stl_path))

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory and all contents
        shutil.rmtree(self.test_dir)

    def test_simplify_mesh(self):
        """Test mesh simplification functionality."""
        original_faces = len(self.sphere.faces)
        
        # Test with reduction ratio
        simplified = simplify_mesh(self.sphere, reduction_ratio=0.5)
        self.assertIsInstance(simplified, trimesh.Trimesh)
        self.assertLess(len(simplified.faces), original_faces)
        
        # Test with target faces
        target_faces = original_faces // 2
        simplified_target = simplify_mesh(self.sphere, target_faces=target_faces)
        self.assertIsInstance(simplified_target, trimesh.Trimesh)
        self.assertLessEqual(len(simplified_target.faces), target_faces * 1.1)  # Allow 10% tolerance

    def test_load_stl(self):
        """Test STL file loading."""
        # Test successful loading
        loaded_mesh = load_stl(self.test_stl_path)
        self.assertIsInstance(loaded_mesh, trimesh.Trimesh)
        
        # Test file not found
        with self.assertRaises(FileNotFoundError):
            load_stl(self.test_dir / "nonexistent.stl")

    def test_combine_meshes(self):
        """Test combining multiple meshes into a scene."""
        meshes = [self.sphere, self.cube, self.cylinder]
        mesh_names = ["sphere", "cube", "cylinder"]
        
        # Test with names
        scene = combine_meshes(meshes, mesh_names)
        self.assertIsInstance(scene, trimesh.Scene)
        self.assertEqual(len(scene.geometry), 3)
        self.assertIn("sphere", scene.geometry)
        self.assertIn("cube", scene.geometry)
        self.assertIn("cylinder", scene.geometry)
        
        # Test without names (auto-generated)
        scene_auto = combine_meshes(meshes)
        self.assertIsInstance(scene_auto, trimesh.Scene)
        self.assertEqual(len(scene_auto.geometry), 3)
        
        # Test empty list
        with self.assertRaises(ValueError):
            combine_meshes([])

    def test_export_mesh_to_glb(self):
        """Test exporting a single mesh to GLB format."""
        output_path = self.test_dir / "test_mesh.glb"
        
        export_mesh_to_glb(self.sphere, output_path)
        
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    def test_export_scene_to_glb(self):
        """Test exporting a scene to GLB format."""
        scene = combine_meshes([self.sphere, self.cube], ["sphere", "cube"])
        output_path = self.test_dir / "test_scene.glb"
        
        export_scene_to_glb(scene, output_path)
        
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    def test_render_mesh(self):
        """Test mesh rendering functionality."""
        output_path = self.test_dir / "test_render.png"
        
        # Test rendering with explicit save path
        render_mesh(self.sphere, title="Test Sphere", save_path=output_path)
        
        # Check if either PNG was created or PLY fallback was created
        png_exists = output_path.exists()
        ply_fallback = output_path.with_suffix('.ply').exists()
        
        self.assertTrue(png_exists or ply_fallback, 
                       "Either PNG render or PLY fallback should be created")
        
        # Test rendering without save path (auto-generated filename)
        render_mesh(self.cube, title="Auto Named Cube")
        
        # Check for auto-generated files
        auto_png = Path("auto_named_cube.png")
        auto_ply = Path("auto_named_cube.ply")
        
        try:
            self.assertTrue(auto_png.exists() or auto_ply.exists(),
                           "Auto-generated render file should exist")
        finally:
            # Clean up auto-generated files
            if auto_png.exists():
                auto_png.unlink()
            if auto_ply.exists():
                auto_ply.unlink()

    def test_render_scene(self):
        """Test scene rendering functionality."""
        scene = combine_meshes([self.sphere, self.cube], ["sphere", "cube"])
        output_path = self.test_dir / "test_scene_render.png"
        
        render_scene(scene, title="Test Scene", save_path=output_path)
        
        # Check if either PNG was created or GLB fallback was created
        png_exists = output_path.exists()
        glb_fallback = output_path.with_suffix('.glb').exists()
        
        self.assertTrue(png_exists or glb_fallback,
                       "Either PNG render or GLB fallback should be created")

    def test_get_mesh_info(self):
        """Test mesh information extraction."""
        info = get_mesh_info(self.sphere)
        
        # Check that all expected keys are present
        expected_keys = {
            'faces', 'vertices', 'volume', 'surface_area',
            'bounding_box_min', 'bounding_box_max', 'center_mass',
            'is_watertight', 'is_winding_consistent'
        }
        
        self.assertEqual(set(info.keys()), expected_keys)
        
        # Check data types
        self.assertIsInstance(info['faces'], int)
        self.assertIsInstance(info['vertices'], int)
        self.assertIsInstance(info['volume'], float)
        self.assertIsInstance(info['surface_area'], float)
        self.assertIsInstance(info['bounding_box_min'], tuple)
        self.assertIsInstance(info['bounding_box_max'], tuple)
        self.assertIsInstance(info['center_mass'], tuple)
        self.assertIsInstance(info['is_watertight'], bool)
        self.assertIsInstance(info['is_winding_consistent'], bool)

    def test_print_mesh_info(self):
        """Test mesh information printing (should not raise exceptions)."""
        # This test mainly ensures the function doesn't crash
        try:
            print_mesh_info(self.sphere, "Test Sphere")
            print_mesh_info(self.cube, "Test Cube")
        except Exception as e:
            self.fail(f"print_mesh_info raised an exception: {e}")

    def test_multi_stl_to_glb(self):
        """Test converting multiple STL files to GLB."""
        # Create multiple test STL files
        stl_paths = []
        for i, mesh in enumerate([self.sphere, self.cube, self.cylinder]):
            stl_path = self.test_dir / f"test_mesh_{i}.stl"
            mesh.export(str(stl_path))
            stl_paths.append(stl_path)
        
        output_path = self.test_dir / "combined.glb"
        
        # Test the conversion
        scene = multi_stl_to_glb(
            stl_paths, 
            output_path, 
            simplify=True, 
            reduction_ratio=0.8,
            render_preview=False  # Disable preview to avoid rendering issues in CI
        )
        
        self.assertIsInstance(scene, trimesh.Scene)
        self.assertEqual(len(scene.geometry), 3)
        self.assertTrue(output_path.exists())

    def test_mesh_properties(self):
        """Test that created meshes have expected properties."""
        # Test sphere properties
        sphere_info = get_mesh_info(self.sphere)
        self.assertGreater(sphere_info['volume'], 0)
        self.assertGreater(sphere_info['surface_area'], 0)
        self.assertTrue(sphere_info['is_watertight'])
        
        # Test cube properties
        cube_info = get_mesh_info(self.cube)
        self.assertAlmostEqual(cube_info['volume'], 1.0, places=1)  # 1x1x1 cube
        self.assertGreater(cube_info['surface_area'], 0)

    def test_error_handling(self):
        """Test error handling in various functions."""
        # Test invalid mesh paths
        with self.assertRaises(FileNotFoundError):
            load_stl("nonexistent_file.stl")
        
        # Test empty mesh list
        with self.assertRaises(ValueError):
            combine_meshes([])
        
        # Test mismatched names and meshes
        with self.assertRaises(ValueError):
            combine_meshes([self.sphere, self.cube], ["only_one_name"])



    def test_multiview_scene_rendering(self):
        """Test multi-view scene rendering functionality."""
        scene = combine_meshes([self.sphere, self.cube], ["sphere", "cube"])
        output_path = self.test_dir / "test_multiview_scene.png"
        
        render_scene_multiview(scene, title="Test Multi-View Scene", save_path=output_path)
        
        # Check if either multiview PNG or fallback GLB was created
        png_exists = output_path.exists()
        glb_fallback = output_path.with_suffix('.glb').exists()
        
        self.assertTrue(png_exists or glb_fallback,
                       "Either multi-view PNG or GLB fallback should be created")


class TestIntegration(unittest.TestCase):
    """Integration tests that test combinations of functions."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_full_pipeline(self):
        """Test a complete pipeline: create -> simplify -> combine -> export -> render."""
        # Create meshes
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        cube = trimesh.creation.box(extents=[2, 2, 2])
        
        # Simplify meshes
        simplified_sphere = simplify_mesh(sphere, reduction_ratio=0.6)
        simplified_cube = simplify_mesh(cube, reduction_ratio=0.6)
        
        # Combine into scene
        scene = combine_meshes([simplified_sphere, simplified_cube], ["sphere", "cube"])
        
        # Export to GLB
        glb_path = self.test_dir / "pipeline_test.glb"
        export_scene_to_glb(scene, glb_path)
        
        # Render scene
        render_path = self.test_dir / "pipeline_render.png"
        render_scene(scene, "Pipeline Test", render_path)
        
        # Verify outputs
        self.assertTrue(glb_path.exists())
        # Either PNG render or GLB fallback should exist
        png_exists = render_path.exists()
        glb_fallback = render_path.with_suffix('.glb').exists()
        self.assertTrue(png_exists or glb_fallback)


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestUtils3D))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TESTS SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running utils_3d unit tests...")
    print("=" * 50)
    success = run_tests()
    exit(0 if success else 1)
