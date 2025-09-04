#!/usr/bin/env python3
"""
GLB Rigged Exporter with Full Skinning Support
Exports MJCF robot models to GLB format with properly skinned meshes.
This creates a complete rigged model ready for animation in Blender.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import struct
import json
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from mjcf_parser import MJCFParser
from armature_utils import ArmatureBuilder, BoneInfo  
from skinning_utils import SkinnedMeshBuilder
from utils_3d import load_stl
from glb_validator import validate_glb_file


class GLBRiggedExporter:
    """Complete GLB exporter with mesh skinning and armature."""
    
    def __init__(self, mjcf_path: str):
        self.mjcf_path = Path(mjcf_path)
        self.parser = MJCFParser(mjcf_path)
        self.armature_builder = ArmatureBuilder(self.parser)
        self.skinning_builder = SkinnedMeshBuilder(self.parser, self.armature_builder)
        
        # Build armature
        self.bones = self.armature_builder.compute_bone_positions()
        
        # GLB data structures
        self.json_data = {
            "asset": {"version": "2.0", "generator": "MJCF GLB Rigged Exporter"},
            "scene": 0,
            "scenes": [{"name": "Scene", "nodes": []}],
            "nodes": [],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
            "materials": [],
            "skins": []
        }
        
        self.binary_buffer = bytearray()
        self.node_index_map: Dict[str, int] = {}
        self.joint_indices: Dict[str, int] = {}
        
    def export(self, output_path: str) -> None:
        """Export the complete rigged model to GLB."""
        print("🔧 Building rigged GLB model...")
        
        # 1. Create armature nodes
        print("  📦 Creating armature hierarchy...")
        root_node_idx = self._create_armature()
        
        # 2. Process and skin meshes  
        print("  🎨 Processing and skinning meshes...")
        mesh_nodes = self._create_skinned_meshes()
        
        # 3. Add mesh nodes to scene
        self.json_data["scenes"][0]["nodes"] = [root_node_idx] + mesh_nodes
        
        # 4. Finalize buffer
        self.json_data["buffers"] = [{"byteLength": len(self.binary_buffer)}]
        
        # 5. Write GLB file
        print(f"  💾 Writing GLB to {output_path}...")
        self._write_glb_file(output_path)
        
        # 6. Validate the generated GLB file
        print(f"  🔍 Validating GLB file...")
        is_valid = validate_glb_file(output_path, verbose=False)
        
        if is_valid:
            print(f"✅ Rigged GLB export complete and validated! Use File > Import > glTF 2.0 in Blender.")
        else:
            print(f"⚠️  GLB export complete but validation failed. Check the file manually.")
            print(f"    Run: python glb_validator.py {output_path} for detailed validation report.")
        
    def _create_armature(self) -> int:
        """Create the armature node hierarchy."""
        # Create joint nodes
        for bone_name, bone_info in self.bones.items():
            # Calculate relative transform to parent
            translation = bone_info.head.copy()
            if bone_info.parent and bone_info.parent in self.bones:
                parent_bone = self.bones[bone_info.parent]
                translation = bone_info.head - parent_bone.head
                
            node = {
                "name": f"{bone_name}",
                "translation": translation.tolist()
            }
            
            node_idx = len(self.json_data["nodes"])
            self.json_data["nodes"].append(node)
            self.joint_indices[bone_name] = node_idx
            
        # Set up parent-child relationships
        for bone_name, bone_info in self.bones.items():
            if bone_info.parent and bone_info.parent in self.joint_indices:
                parent_idx = self.joint_indices[bone_info.parent]
                child_idx = self.joint_indices[bone_name]
                
                if "children" not in self.json_data["nodes"][parent_idx]:
                    self.json_data["nodes"][parent_idx]["children"] = []
                self.json_data["nodes"][parent_idx]["children"].append(child_idx)
                
        # Create root armature node
        root_joints = [self.joint_indices[name] for name, bone in self.bones.items() 
                      if bone.parent is None]
        
        armature_node = {
            "name": "Armature",
            "children": root_joints
        }
        
        armature_idx = len(self.json_data["nodes"])
        self.json_data["nodes"].append(armature_node)
        
        return armature_idx
        
    def _create_skinned_meshes(self) -> List[int]:
        """Create skinned mesh nodes."""
        mesh_nodes = []
        mesh_transforms = self.parser.get_mesh_transforms()
        
        # Create a single combined mesh for simplicity
        combined_vertices = []
        combined_faces = []
        combined_weights = []
        combined_joints = []
        vertex_offset = 0
        
        # Process each mesh
        for mesh_name, transforms in mesh_transforms.items():
            mesh_path = self.parser.get_mesh_file_path(mesh_name)
            if not mesh_path or not mesh_path.exists():
                continue
                
            try:
                mesh = load_stl(mesh_path)
                if len(mesh.faces) > 5000:
                    mesh = mesh.simplify_quadric_decimation(face_count=5000)
                    
                # For each instance of this mesh
                for body_name, position, rotation_matrix, material in transforms:
                    # Transform mesh to world space
                    transform = np.eye(4)
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = position
                    
                    transformed_mesh = mesh.copy()
                    transformed_mesh.apply_transform(transform)
                    
                    # Get skinning weights for this mesh
                    skinning = self.skinning_builder.assign_vertex_weights(
                        transformed_mesh, body_name, mesh_name
                    )
                    
                    # Add to combined mesh
                    vertices = transformed_mesh.vertices
                    faces = transformed_mesh.faces + vertex_offset
                    
                    combined_vertices.append(vertices)
                    combined_faces.append(faces)
                    
                    # Process weights (max 4 influences per vertex)
                    for vertex_weights in skinning.vertex_weights:
                        # Sort by weight and take top 4
                        sorted_weights = sorted(vertex_weights, key=lambda x: x[1], reverse=True)[:4]
                        
                        joints = []
                        weights = []
                        for bone_name, weight in sorted_weights:
                            if bone_name in self.joint_indices:
                                joints.append(self.joint_indices[bone_name])
                                weights.append(weight)
                                
                        # Pad to 4 influences
                        while len(joints) < 4:
                            joints.append(0)
                            weights.append(0.0)
                            
                        # Normalize weights
                        total_weight = sum(weights)
                        if total_weight > 0:
                            weights = [w / total_weight for w in weights]
                            
                        combined_joints.append(joints)
                        combined_weights.append(weights)
                        
                    vertex_offset += len(vertices)
                    
            except Exception as e:
                print(f"    ⚠️  Error processing {mesh_name}: {e}")
                continue
                
        if not combined_vertices:
            return []
            
        # Combine all mesh data
        all_vertices = np.vstack(combined_vertices)
        all_faces = np.vstack(combined_faces)
        all_joints = np.array(combined_joints, dtype=np.uint16)
        all_weights = np.array(combined_weights, dtype=np.float32)
        
        # Create mesh primitive
        mesh_data = self._create_mesh_primitive(
            all_vertices, all_faces, all_joints, all_weights
        )
        
        # Create mesh node
        mesh_node = {
            "name": "Robot_Mesh",
            "mesh": mesh_data["mesh_idx"],
            "skin": mesh_data["skin_idx"]
        }
        
        mesh_node_idx = len(self.json_data["nodes"])
        self.json_data["nodes"].append(mesh_node)
        
        return [mesh_node_idx]
        
    def _create_mesh_primitive(self, vertices: np.ndarray, faces: np.ndarray,
                              joints: np.ndarray, weights: np.ndarray) -> Dict[str, int]:
        """Create mesh primitive with skinning data."""
        # Add vertex position accessor
        pos_data = vertices.astype(np.float32).tobytes()
        pos_view_idx = self._add_buffer_view(pos_data, 34962)  # ARRAY_BUFFER
        pos_accessor_idx = self._add_accessor(
            pos_view_idx, 0, 5126, "VEC3", len(vertices),
            vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()
        )
        
        # Add face indices accessor - use uint32 to avoid primitive restart issues
        max_index = faces.max()
        if max_index > 65534:  # Leave room for valid indices
            indices = faces.flatten().astype(np.uint32).tobytes()
            component_type = 5125  # UNSIGNED_INT
        else:
            indices = faces.flatten().astype(np.uint16).tobytes()
            component_type = 5123  # UNSIGNED_SHORT
            
        indices_view_idx = self._add_buffer_view(indices, 34963)  # ELEMENT_ARRAY_BUFFER
        indices_accessor_idx = self._add_accessor(
            indices_view_idx, 0, component_type, "SCALAR", len(faces) * 3
        )
        
        # Add joints accessor
        joints_data = joints.astype(np.uint16).tobytes()
        joints_view_idx = self._add_buffer_view(joints_data, 34962)
        joints_accessor_idx = self._add_accessor(
            joints_view_idx, 0, 5123, "VEC4", len(joints)
        )
        
        # Add weights accessor
        weights_data = weights.astype(np.float32).tobytes()
        weights_view_idx = self._add_buffer_view(weights_data, 34962)
        weights_accessor_idx = self._add_accessor(
            weights_view_idx, 0, 5126, "VEC4", len(weights)
        )
        
        # Create mesh
        mesh = {
            "name": "Robot_Mesh",
            "primitives": [{
                "attributes": {
                    "POSITION": pos_accessor_idx,
                    "JOINTS_0": joints_accessor_idx,
                    "WEIGHTS_0": weights_accessor_idx
                },
                "indices": indices_accessor_idx,
                "material": 0
            }]
        }
        
        mesh_idx = len(self.json_data["meshes"])
        self.json_data["meshes"].append(mesh)
        
        # Create skin with inverse bind matrices
        skin_idx = self._create_skin()
        
        # Add default material
        if not self.json_data["materials"]:
            self.json_data["materials"].append({
                "name": "Default",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                    "metallicFactor": 0.5,
                    "roughnessFactor": 0.5
                }
            })
            
        return {"mesh_idx": mesh_idx, "skin_idx": skin_idx}
        
    def _create_skin(self) -> int:
        """Create skin with inverse bind matrices."""
        # Calculate inverse bind matrices for each joint
        joint_list = sorted(self.joint_indices.keys(), 
                           key=lambda x: self.joint_indices[x])
        
        matrices = []
        for joint_name in joint_list:
            if joint_name in self.bones:
                bone = self.bones[joint_name]
                # Calculate the world transform for this joint
                world_transform = self._calculate_joint_world_transform(joint_name)
                
                # Inverse bind matrix is the inverse of the world transform
                try:
                    inv_bind = np.linalg.inv(world_transform)
                    # Ensure the matrix is valid (no NaN or Inf values)
                    if not np.isfinite(inv_bind).all():
                        print(f"Warning: Invalid inverse bind matrix for joint {joint_name}, using identity")
                        inv_bind = np.eye(4)
                except np.linalg.LinAlgError:
                    print(f"Warning: Singular matrix for joint {joint_name}, using identity")
                    inv_bind = np.eye(4)
                    
                matrices.append(inv_bind)
            else:
                matrices.append(np.eye(4))
                
        # Convert to bytes (glTF uses column-major matrices)
        # NumPy arrays are row-major by default, so we need to transpose
        matrices_transposed = [mat.T for mat in matrices]
        matrix_data = np.array(matrices_transposed, dtype=np.float32).tobytes()
        matrix_view_idx = self._add_buffer_view(matrix_data, None)
        matrix_accessor_idx = self._add_accessor(
            matrix_view_idx, 0, 5126, "MAT4", len(matrices)
        )
        
        # Create skin
        skin = {
            "name": "Robot_Skin",
            "inverseBindMatrices": matrix_accessor_idx,
            "joints": [self.joint_indices[name] for name in joint_list]
        }
        
        skin_idx = len(self.json_data["skins"])
        self.json_data["skins"].append(skin)
        
        return skin_idx
        
    def _calculate_joint_world_transform(self, joint_name: str) -> np.ndarray:
        """Calculate the world space transform matrix for a joint."""
        if joint_name not in self.bones:
            return np.eye(4)
            
        bone = self.bones[joint_name]
        transform = np.eye(4)
        transform[:3, 3] = bone.head
        
        # Accumulate parent transforms
        current_joint = joint_name
        while current_joint in self.bones and self.bones[current_joint].parent:
            parent_name = self.bones[current_joint].parent
            if parent_name in self.bones:
                parent_bone = self.bones[parent_name]
                parent_transform = np.eye(4)
                parent_transform[:3, 3] = parent_bone.head
                # This is simplified - in a full implementation you'd want to 
                # accumulate rotations as well, but for basic skinning this works
                current_joint = parent_name
            else:
                break
                
        return transform
        
    def _add_buffer_view(self, data: bytes, target: Optional[int] = None) -> int:
        """Add data to binary buffer and create buffer view."""
        offset = len(self.binary_buffer)
        self.binary_buffer.extend(data)
        
        view = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data)
        }
        if target is not None:
            view["target"] = target
            
        idx = len(self.json_data["bufferViews"])
        self.json_data["bufferViews"].append(view)
        return idx
        
    def _add_accessor(self, buffer_view: int, offset: int, component_type: int,
                     type_: str, count: int, min_vals: List[float] = None,
                     max_vals: List[float] = None) -> int:
        """Add accessor for buffer view."""
        accessor = {
            "bufferView": buffer_view,
            "byteOffset": offset,
            "componentType": component_type,
            "type": type_,
            "count": count
        }
        
        if min_vals is not None:
            accessor["min"] = min_vals
        if max_vals is not None:
            accessor["max"] = max_vals
            
        idx = len(self.json_data["accessors"])
        self.json_data["accessors"].append(accessor)
        return idx
        
    def _write_glb_file(self, output_path: str) -> None:
        """Write the complete GLB file."""
        # Convert JSON to bytes
        json_bytes = json.dumps(self.json_data).encode('utf-8')
        json_padding = (4 - len(json_bytes) % 4) % 4
        json_bytes += b' ' * json_padding
        
        # Pad binary buffer
        binary_padding = (4 - len(self.binary_buffer) % 4) % 4
        self.binary_buffer.extend(b'\x00' * binary_padding)
        
        # Calculate total length
        total_length = 12 + 8 + len(json_bytes) + 8 + len(self.binary_buffer)
        
        # Write GLB
        with open(output_path, 'wb') as f:
            # Header
            f.write(b'glTF')
            f.write(struct.pack('<I', 2))  # Version
            f.write(struct.pack('<I', total_length))
            
            # JSON chunk
            f.write(struct.pack('<I', len(json_bytes)))
            f.write(b'JSON')
            f.write(json_bytes)
            
            # Binary chunk
            f.write(struct.pack('<I', len(self.binary_buffer)))
            f.write(b'BIN\x00')
            f.write(self.binary_buffer)


def export_rigged_robot(mjcf_path: str = "./g1_description/g1_mjx_alt.xml",
                       output_path: str = "output/robot_fully_rigged.glb") -> None:
    """Export a fully rigged robot model to GLB."""
    exporter = GLBRiggedExporter(mjcf_path)
    exporter.export(output_path)


if __name__ == "__main__":
    export_rigged_robot()
