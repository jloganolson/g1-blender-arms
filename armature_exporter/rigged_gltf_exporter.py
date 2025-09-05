#!/usr/bin/env python3
"""
GLTF Armature Exporter
Creates GLB files with a rigged skeleton where rigid meshes are parented to bones.
This is achieved using skinning with 100% weight on a single bone for each mesh,
which is how rigid parenting to a bone is represented in GLTF.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import base64
from dataclasses import dataclass
import hashlib

try:
    from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Accessor, BufferView, Buffer
    from pygltflib import Material, Skin
    from pygltflib import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
    from pygltflib import FLOAT, UNSIGNED_INT, UNSIGNED_SHORT
    from pygltflib import JOINTS_0, WEIGHTS_0, POSITION, NORMAL
    PYGLTFLIB_AVAILABLE = True
except ImportError:
    PYGLTFLIB_AVAILABLE = False
    print("Warning: pygltflib not available. Install with: uv pip install pygltflib")

from .mjcf_parser import MJCFParser, JointInfo

@dataclass
class BoneInfo:
    """Information about a bone in the armature."""
    name: str
    parent_bone: Optional[str] = None
    joint_info: Optional[JointInfo] = None
    body_name: str = ""
    transform_matrix: np.ndarray = None
    children: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.transform_matrix is None:
            self.transform_matrix = np.eye(4)


class RiggedGLTFExporter:
    """
    Builds a rigged GLTF file by parenting meshes to bones.
    """
    
    def __init__(self, parser: MJCFParser, bones: Dict[str, BoneInfo], 
                 bone_hierarchy: List[str], mesh_instance_weights: Dict[str, Dict[str, float]]):
        if not PYGLTFLIB_AVAILABLE:
            raise ImportError("pygltflib is required. Install with: uv pip install pygltflib")
        
        self.parser = parser
        self.bones = bones
        self.bone_hierarchy = bone_hierarchy
        self.mesh_instance_weights = mesh_instance_weights
        self.gltf = GLTF2()
        
        self.buffer_data = bytearray()
        self.bone_to_node_index: Dict[str, int] = {}

    def _add_buffer_data(self, data: bytes) -> Tuple[int, int]:
        """Add data to buffer, align to 4 bytes, and return (offset, length)."""
        offset = len(self.buffer_data)
        self.buffer_data.extend(data)
        while len(self.buffer_data) % 4 != 0:
            self.buffer_data.extend(b'\\x00')
        return offset, len(data)

    def _create_buffer_view(self, buffer_index: int, byte_offset: int, 
                           byte_length: int, target: int = None) -> int:
        """Create a buffer view and return its index."""
        buffer_view = BufferView(buffer=buffer_index, byteOffset=byte_offset, byteLength=byte_length, target=target)
        self.gltf.bufferViews.append(buffer_view)
        return len(self.gltf.bufferViews) - 1

    def _create_accessor(self, buffer_view_index: int, component_type: int, 
                        count: int, type_: str, max_vals: List[float] = None, 
                        min_vals: List[float] = None) -> int:
        """Create an accessor and return its index."""
        accessor = Accessor(bufferView=buffer_view_index, componentType=component_type, count=count, type=type_, max=max_vals, min=min_vals)
        self.gltf.accessors.append(accessor)
        return len(self.gltf.accessors) - 1

    def _create_vertex_weights(self, num_vertices: int, bone_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Creates vertex weights for a mesh, binding all vertices to a single bone."""
        joints = np.zeros((num_vertices, 4), dtype=np.uint16)
        weights = np.zeros((num_vertices, 4), dtype=np.float32)
        
        if bone_name in self.bone_to_node_index:
            bone_index = self.bone_to_node_index[bone_name]
            joints[:, 0] = bone_index
            weights[:, 0] = 1.0
            
        return joints, weights

    def _add_mesh_to_gltf(self, mesh: trimesh.Trimesh, controlling_bone_name: str) -> int:
        """Adds a trimesh object to the GLTF buffer and returns the mesh index."""
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32).flatten()
        normals = mesh.vertex_normals.astype(np.float32)

        # Buffer data
        vertex_offset, vertex_len = self._add_buffer_data(vertices.tobytes())
        index_offset, index_len = self._add_buffer_data(faces.tobytes())
        normal_offset, normal_len = self._add_buffer_data(normals.tobytes())

        # Buffer views
        vertex_bv = self._create_buffer_view(0, vertex_offset, vertex_len, ARRAY_BUFFER)
        index_bv = self._create_buffer_view(0, index_offset, index_len, ELEMENT_ARRAY_BUFFER)
        normal_bv = self._create_buffer_view(0, normal_offset, normal_len, ARRAY_BUFFER)

        # Accessors
        pos_acc = self._create_accessor(vertex_bv, FLOAT, len(vertices), "VEC3", vertices.max(axis=0).tolist(), vertices.min(axis=0).tolist())
        ind_acc = self._create_accessor(index_bv, UNSIGNED_INT, len(faces), "SCALAR")
        norm_acc = self._create_accessor(normal_bv, FLOAT, len(normals), "VEC3")

        attributes = {POSITION: pos_acc, NORMAL: norm_acc}

        # Skinning data
        if controlling_bone_name:
            joints, weights_vals = self._create_vertex_weights(len(vertices), controlling_bone_name)
            
            joint_data = joints.astype(np.uint16).tobytes()
            weight_data = weights_vals.astype(np.float32).tobytes()

            joint_offset, joint_len = self._add_buffer_data(joint_data)
            weight_offset, weight_len = self._add_buffer_data(weight_data)

            joint_bv = self._create_buffer_view(0, joint_offset, joint_len, ARRAY_BUFFER)
            weight_bv = self._create_buffer_view(0, weight_offset, weight_len, ARRAY_BUFFER)

            joint_acc = self._create_accessor(joint_bv, UNSIGNED_SHORT, len(vertices), "VEC4")
            weight_acc = self._create_accessor(weight_bv, FLOAT, len(vertices), "VEC4")
            
            attributes[JOINTS_0] = joint_acc
            attributes[WEIGHTS_0] = weight_acc

        primitive = Primitive(attributes=attributes, indices=ind_acc, material=0)
        gltf_mesh = Mesh(primitives=[primitive])
        self.gltf.meshes.append(gltf_mesh)
        mesh_index = len(self.gltf.meshes) - 1
        return mesh_index

    def _create_skin(self) -> int:
        """Creates the skin for the armature."""
        joint_indices = [self.bone_to_node_index[b] for b in self.bone_hierarchy if b in self.bone_to_node_index]
        
        num_joints = len(joint_indices)
        ibm = np.tile(np.eye(4, dtype=np.float32), (num_joints, 1, 1))
        
        # In GLTF, the inverse bind matrix for a joint is the inverse of the world transform of that joint *at the time of binding*.
        # For our case, we can simplify this. Since our bone transforms are already in world space,
        # the IBM for each bone is simply the inverse of its world transform.
        for i, bone_name in enumerate(self.bone_hierarchy):
             if bone_name in self.bones:
                 bone_transform = self.bones[bone_name].transform_matrix
                 try:
                     ibm[i] = np.linalg.inv(bone_transform)
                 except np.linalg.LinAlgError:
                     print(f"Warning: Could not invert transform for bone {bone_name}")
                     # Leave as identity if not invertible

        ibm_data = ibm.tobytes()
        ibm_offset, ibm_len = self._add_buffer_data(ibm_data)
        ibm_bv = self._create_buffer_view(0, ibm_offset, ibm_len)
        ibm_acc = self._create_accessor(ibm_bv, FLOAT, num_joints, "MAT4")

        skin = Skin(joints=joint_indices, inverseBindMatrices=ibm_acc)
        self.gltf.skins.append(skin)
        return len(self.gltf.skins) - 1

    def _create_armature_nodes(self) -> List[int]:
        """Create nodes for the armature bones and return list of root node indices."""
        root_nodes = []
        for bone_name in self.bone_hierarchy:
            bone = self.bones[bone_name]
            node = Node(name=bone_name)

            if bone.parent_bone and bone.parent_bone in self.bones:
                parent_bone = self.bones[bone.parent_bone]
                bone_transform = bone.transform_matrix
                parent_transform = parent_bone.transform_matrix
                
                # Relative translation
                translation = (bone_transform[:3, 3] - parent_transform[:3, 3]).tolist()
                node.translation = translation
            else:
                # Root bone
                node.translation = bone.transform_matrix[:3, 3].tolist()
                root_nodes.append(len(self.gltf.nodes))

            self.gltf.nodes.append(node)
            node_index = len(self.gltf.nodes) - 1
            self.bone_to_node_index[bone_name] = node_index

            if bone.parent_bone and bone.parent_bone in self.bone_to_node_index:
                parent_node_index = self.bone_to_node_index[bone.parent_bone]
                parent_node = self.gltf.nodes[parent_node_index]
                if parent_node.children is None:
                    parent_node.children = []
                parent_node.children.append(node_index)
        
        return root_nodes

    def build_glb(self, output_path: str):
        """Build and save the rigged GLB file."""
        print(f"\\n=== Building Rigged GLB ===")
        print(f"Output: {output_path}")

        self.gltf.scenes = [Scene(nodes=[])]
        self.gltf.scene = 0
        self.gltf.materials = [Material(name="DefaultMaterial")]
        
        # Create bone nodes
        armature_root_nodes = self._create_armature_nodes()

        # Add armature roots to the scene
        self.gltf.scenes[0].nodes.extend(armature_root_nodes)

        # Create skin
        skin_index = self._create_skin()
        
        # Add meshes
        mesh_transforms = self.parser.get_mesh_transforms()
        for mesh_name, transforms in mesh_transforms.items():
            mesh_file_path = self.parser.get_mesh_file_path(mesh_name)
            if not mesh_file_path or not mesh_file_path.exists():
                print(f"Warning: Mesh file for {mesh_name} not found.")
                continue

            try:
                mesh = trimesh.load(str(mesh_file_path))
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            except Exception as e:
                print(f"Warning: Could not load mesh {mesh_name}: {e}")
                continue

            for i, (body_name, pos, rot, mat) in enumerate(transforms):
                instance_name = f"{body_name}_{mesh_name}"
                if len(transforms) > 1:
                    instance_name += f"_{i}"
                
                original_transform = np.eye(4)
                original_transform[:3, :3] = rot
                original_transform[:3, 3] = pos
                
                final_transform = original_transform.copy()
                controlling_bone_name = None
                
                if instance_name in self.mesh_instance_weights and self.mesh_instance_weights[instance_name]:
                    bone_name = list(self.mesh_instance_weights[instance_name].keys())[0]
                    if bone_name in self.bones:
                        controlling_bone_name = bone_name
                        bone_transform = self.bones[bone_name].transform_matrix
                        try:
                            # This is the key logic from visualize_bone_hierarchy.py
                            bone_inverse = np.linalg.inv(bone_transform)
                            final_transform = bone_inverse @ original_transform
                        except np.linalg.LinAlgError:
                            print(f"Warning: Could not invert bone transform for {bone_name}")

                mesh_index = self._add_mesh_to_gltf(mesh, controlling_bone_name)

                mesh_node = Node(name=instance_name, mesh=mesh_index)
                if controlling_bone_name:
                    mesh_node.skin = skin_index
                
                # The node transform should be the mesh's final computed transform
                # In GLTF, for a skinned mesh, this transform is applied on top of the bone's transform.
                # Since we want the mesh to be rigidly attached, this should be an identity matrix
                # if the mesh vertices are already in the bone's local frame.
                # However, our vertices are in world space, and our relative transform calculation
                # puts the mesh into the bone's local space. So, the node's transform should be identity.
                # Let's test with identity first. If that's wrong, we'll use final_transform.
                
                # After further thought, the logic from visualize_bone_hierarchy is what we want.
                # The mesh vertices are defined in their own local space.
                # The 'original_transform' brings them into world space.
                # The 'final_transform' moves them from world space into the bone's local space.
                # When the GLTF renderer applies the bone's world transform, and then this 'final_transform',
                # the mesh should appear at the correct world location.
                
                if not np.allclose(final_transform, np.eye(4)):
                    # Get translation from the transform matrix
                    trans = final_transform[:3, 3]
                    # Get quaternion from the transform matrix
                    rot_quat = trimesh.transformations.quaternion_from_matrix(final_transform)
                    
                    mesh_node.translation = trans.tolist()
                    # Convert (w, x, y, z) to (x, y, z, w) for GLTF
                    mesh_node.rotation = [rot_quat[1], rot_quat[2], rot_quat[3], rot_quat[0]]
                
                self.gltf.nodes.append(mesh_node)
                mesh_node_index = len(self.gltf.nodes) - 1
                self.gltf.scenes[0].nodes.append(mesh_node_index)

        # Finalize buffer
        if len(self.buffer_data) == 0:
            # Add some dummy data if the buffer is empty, as it's required
            self._add_buffer_data(b'\\x00\\x00\\x00\\x00')

        buffer = Buffer(byteLength=len(self.buffer_data))
        self.gltf.buffers = [buffer]
        buffer.uri = "data:application/octet-stream;base64," + base64.b64encode(self.buffer_data).decode('ascii')

        # Save GLB
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.gltf.save(str(output_path))
        print(f"✅ Rigged GLB saved: {output_path}")


def build_bone_hierarchy(parser: MJCFParser, target_joints: List[str]) -> Tuple[Dict[str, BoneInfo], List[str], Dict[str, Dict[str, float]]]:
    """
    Build bone hierarchy from MJCF joint data.
    """
    bones = {}
    mesh_instance_weights = {}
    
    for joint_name in target_joints:
        if joint_name not in parser.joints:
            print(f"Warning: Joint '{joint_name}' not found in MJCF parser.")
            continue
        joint_info = parser.joints[joint_name]
        
        body_name = None
        for bn, body in parser.bodies.items():
            if body.joint and body.joint.name == joint_name:
                body_name = bn
                break
        
        if body_name is None:
            print(f"Warning: Could not find body for joint {joint_name}")
            continue
        
        body_info = parser.bodies[body_name]
        
        parent_bone = None
        current_parent = body_info.parent
        while current_parent and parent_bone is None:
            if current_parent in parser.bodies:
                parent_body = parser.bodies[current_parent]
                if parent_body.joint and parent_body.joint.name in target_joints:
                    parent_bone = parent_body.joint.name
                    break
                current_parent = parent_body.parent
            else:
                break

        global_pos, global_rot_mat = parser.compute_global_transform(body_name)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = global_rot_mat
        transform_matrix[:3, 3] = global_pos
        
        bone = BoneInfo(
            name=joint_name,
            parent_bone=parent_bone,
            joint_info=joint_info,
            body_name=body_name,
            transform_matrix=transform_matrix
        )
        bones[joint_name] = bone
    
    bone_hierarchy = []
    visited = set()
    def visit_bone(bone_name: str):
        if bone_name in visited or bone_name not in bones:
            return
        bone = bones[bone_name]
        if bone.parent_bone and bone.parent_bone not in visited:
            visit_bone(bone.parent_bone)
        if bone_name not in visited:
            bone_hierarchy.append(bone_name)
            visited.add(bone_name)
    
    for bone_name in bones.keys():
        visit_bone(bone_name)
    
    for bone_name, bone in bones.items():
        if bone.parent_bone and bone.parent_bone in bones:
            bones[bone.parent_bone].children.append(bone_name)
    
    mesh_transforms = parser.get_mesh_transforms()
    for mesh_name, transforms in mesh_transforms.items():
        for i, (body_name, pos, rot, mat) in enumerate(transforms):
            instance_name = f"{body_name}_{mesh_name}"
            if len(transforms) > 1:
                instance_name += f"_{i}"
            
            controlling_joint = None
            current_body = body_name
            while current_body and controlling_joint is None:
                if current_body in parser.bodies:
                    body_info = parser.bodies[current_body]
                    if body_info.joint and body_info.joint.name in target_joints:
                        controlling_joint = body_info.joint.name
                        break
                    current_body = body_info.parent
                else:
                    break
            
            instance_weights = {}
            if controlling_joint:
                instance_weights[controlling_joint] = 1.0
            mesh_instance_weights[instance_name] = instance_weights
            
    return bones, bone_hierarchy, mesh_instance_weights


def create_rigged_glb(mjcf_path: str, output_path: str, target_joints: List[str]):
    """High-level function to create a rigged GLB."""
    print(f"Creating rigged GLB from {mjcf_path}")
    parser = MJCFParser(mjcf_path)
    bones, bone_hierarchy, mesh_weights = build_bone_hierarchy(parser, target_joints)
    
    exporter = RiggedGLTFExporter(parser, bones, bone_hierarchy, mesh_weights)
    exporter.build_glb(output_path)
    print(f"\\n✅ Rigged GLB creation complete: {output_path}")
