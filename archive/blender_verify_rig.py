#!/usr/bin/env python3
"""
Blender Script for Verifying and Enhancing MJCF-based Rig
Run this script in Blender after importing the GLB file to:
1. Verify the armature structure
2. Add joint constraints based on MJCF limits
3. Set up IK chains for easier animation

Usage in Blender:
1. Open Blender
2. File > Import > glTF 2.0 and select robot_fully_rigged.glb
3. Open the Scripting tab
4. Load and run this script
"""

import bpy
import json
from pathlib import Path
import numpy as np


class MJCFConstraintApplier:
    """Apply MJCF joint constraints to Blender armature."""
    
    def __init__(self):
        self.armature = None
        self.joint_limits = {}
        
    def load_joint_limits_from_mjcf(self, mjcf_info_path: str = "output/mjcf_joint_info.json"):
        """Load joint limit information exported from MJCF parser."""
        if Path(mjcf_info_path).exists():
            with open(mjcf_info_path, 'r') as f:
                self.joint_limits = json.load(f)
        else:
            print(f"Warning: {mjcf_info_path} not found. Using default constraints.")
            self._set_default_limits()
            
    def _set_default_limits(self):
        """Set reasonable default joint limits."""
        # Hip joints
        hip_pitch_limit = (-2.0, 2.0)  # ~-115° to 115°
        hip_roll_limit = (-0.5, 0.5)   # ~-30° to 30°
        hip_yaw_limit = (-0.8, 0.8)    # ~-45° to 45°
        
        # Knee
        knee_limit = (-2.5, 0.0)       # ~-140° to 0°
        
        # Ankle
        ankle_pitch_limit = (-0.9, 0.9) # ~-50° to 50°
        ankle_roll_limit = (-0.3, 0.3)  # ~-17° to 17°
        
        # Waist
        waist_yaw_limit = (-1.5, 1.5)   # ~-85° to 85°
        
        # Shoulder
        shoulder_pitch_limit = (-3.0, 3.0)  # ~-170° to 170°
        shoulder_roll_limit = (-1.5, 0.3)  # ~-85° to 17°
        shoulder_yaw_limit = (-1.5, 2.0)   # ~-85° to 115°
        
        # Elbow
        elbow_limit = (0.0, 2.6)           # ~0° to 150°
        
        # Wrist
        wrist_roll_limit = (-1.5, 1.5)     # ~-85° to 85°
        
        self.joint_limits = {
            # Left leg
            "left_hip_pitch_joint": hip_pitch_limit,
            "left_hip_roll_joint": hip_roll_limit,
            "left_hip_yaw_joint": hip_yaw_limit,
            "left_knee_joint": knee_limit,
            "left_ankle_pitch_joint": ankle_pitch_limit,
            "left_ankle_roll_joint": ankle_roll_limit,
            
            # Right leg
            "right_hip_pitch_joint": hip_pitch_limit,
            "right_hip_roll_joint": hip_roll_limit,
            "right_hip_yaw_joint": hip_yaw_limit,
            "right_knee_joint": knee_limit,
            "right_ankle_pitch_joint": ankle_pitch_limit,
            "right_ankle_roll_joint": ankle_roll_limit,
            
            # Torso
            "waist_yaw_joint": waist_yaw_limit,
            
            # Left arm
            "left_shoulder_pitch_joint": shoulder_pitch_limit,
            "left_shoulder_roll_joint": shoulder_roll_limit,
            "left_shoulder_yaw_joint": shoulder_yaw_limit,
            "left_elbow_joint": elbow_limit,
            "left_wrist_roll_joint": wrist_roll_limit,
            
            # Right arm
            "right_shoulder_pitch_joint": shoulder_pitch_limit,
            "right_shoulder_roll_joint": shoulder_roll_limit,
            "right_shoulder_yaw_joint": shoulder_yaw_limit,
            "right_elbow_joint": elbow_limit,
            "right_wrist_roll_joint": wrist_roll_limit,
        }
    
    def find_armature(self):
        """Find the imported armature object."""
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                self.armature = obj
                print(f"Found armature: {obj.name}")
                return True
        print("No armature found!")
        return False
        
    def apply_constraints(self):
        """Apply joint constraints to bones."""
        if not self.armature:
            print("No armature to apply constraints to!")
            return
            
        # Enter pose mode
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode='POSE')
        
        applied_count = 0
        for bone in self.armature.pose.bones:
            # Try to match bone name to joint name
            joint_name = None
            for jname in self.joint_limits:
                if jname.replace("_joint", "") in bone.name:
                    joint_name = jname
                    break
                    
            if joint_name and joint_name in self.joint_limits:
                limits = self.joint_limits[joint_name]
                
                # Add limit rotation constraint
                constraint = bone.constraints.new('LIMIT_ROTATION')
                constraint.use_limit_x = True
                constraint.min_x = limits[0]
                constraint.max_x = limits[1]
                constraint.owner_space = 'LOCAL'
                
                print(f"Applied constraints to {bone.name}: {limits}")
                applied_count += 1
                
        print(f"Applied constraints to {applied_count} bones")
        
        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
    def setup_ik_chains(self):
        """Set up IK chains for easier animation."""
        if not self.armature:
            return
            
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Add IK targets for feet
        for side in ['left', 'right']:
            ankle_bone_name = f"{side}_ankle_roll"
            if ankle_bone_name in self.armature.data.edit_bones:
                ankle_bone = self.armature.data.edit_bones[ankle_bone_name]
                
                # Create IK target bone
                ik_target = self.armature.data.edit_bones.new(f"{side}_foot_ik_target")
                ik_target.head = ankle_bone.tail.copy()
                ik_target.tail = ik_target.head + (ankle_bone.tail - ankle_bone.head)
                ik_target.use_deform = False
                
                # Create pole target for knee
                knee_bone_name = f"{side}_knee"
                if knee_bone_name in self.armature.data.edit_bones:
                    knee_bone = self.armature.data.edit_bones[knee_bone_name]
                    pole_target = self.armature.data.edit_bones.new(f"{side}_knee_pole")
                    pole_target.head = knee_bone.head + (knee_bone.x_axis * 0.3)
                    pole_target.tail = pole_target.head + (knee_bone.x_axis * 0.1)
                    pole_target.use_deform = False
                    
        # Add IK targets for hands
        for side in ['left', 'right']:
            wrist_bone_name = f"{side}_wrist_roll"
            if wrist_bone_name in self.armature.data.edit_bones:
                wrist_bone = self.armature.data.edit_bones[wrist_bone_name]
                
                # Create IK target bone
                ik_target = self.armature.data.edit_bones.new(f"{side}_hand_ik_target")
                ik_target.head = wrist_bone.tail.copy()
                ik_target.tail = ik_target.head + (wrist_bone.tail - wrist_bone.head)
                ik_target.use_deform = False
                
        bpy.ops.object.mode_set(mode='POSE')
        
        # Add IK constraints
        for side in ['left', 'right']:
            # Foot IK
            ankle_bone_name = f"{side}_ankle_roll"
            if ankle_bone_name in self.armature.pose.bones:
                ankle_bone = self.armature.pose.bones[ankle_bone_name]
                ik_constraint = ankle_bone.constraints.new('IK')
                ik_constraint.target = self.armature
                ik_constraint.subtarget = f"{side}_foot_ik_target"
                ik_constraint.chain_count = 4  # ankle -> knee -> hip_yaw -> hip_roll
                
                # Set pole target
                if f"{side}_knee_pole" in self.armature.pose.bones:
                    ik_constraint.pole_target = self.armature
                    ik_constraint.pole_subtarget = f"{side}_knee_pole"
                    ik_constraint.pole_angle = np.radians(-90)
                    
            # Hand IK
            wrist_bone_name = f"{side}_wrist_roll"
            if wrist_bone_name in self.armature.pose.bones:
                wrist_bone = self.armature.pose.bones[wrist_bone_name]
                ik_constraint = wrist_bone.constraints.new('IK')
                ik_constraint.target = self.armature
                ik_constraint.subtarget = f"{side}_hand_ik_target"
                ik_constraint.chain_count = 3  # wrist -> elbow -> shoulder_yaw
                
        bpy.ops.object.mode_set(mode='OBJECT')
        print("IK chains set up successfully")
        
    def verify_rig(self):
        """Print rig verification info."""
        if not self.armature:
            return
            
        print("\n=== Rig Verification ===")
        print(f"Armature: {self.armature.name}")
        print(f"Total bones: {len(self.armature.data.bones)}")
        
        # Check for deform bones
        deform_bones = [b for b in self.armature.data.bones if b.use_deform]
        print(f"Deform bones: {len(deform_bones)}")
        
        # Check mesh skinning
        skinned_meshes = []
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.modifiers:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE':
                        skinned_meshes.append(obj.name)
                        print(f"Skinned mesh: {obj.name} -> {mod.object.name}")
                        
        print(f"Total skinned meshes: {len(skinned_meshes)}")
        
        # List joint hierarchy
        print("\nJoint Hierarchy:")
        root_bones = [b for b in self.armature.data.bones if b.parent is None]
        for root in root_bones:
            self._print_bone_hierarchy(root, 0)
            
    def _print_bone_hierarchy(self, bone, indent):
        """Recursively print bone hierarchy."""
        print("  " * indent + f"- {bone.name}")
        for child in bone.children:
            self._print_bone_hierarchy(child, indent + 1)


def main():
    """Main function to run in Blender."""
    print("Starting MJCF Rig Verification and Enhancement...")
    
    applier = MJCFConstraintApplier()
    
    # Find armature
    if not applier.find_armature():
        print("Please import the GLB file first!")
        return
        
    # Load joint limits
    applier.load_joint_limits_from_mjcf()
    
    # Verify rig
    applier.verify_rig()
    
    # Apply constraints
    print("\nApplying joint constraints...")
    applier.apply_constraints()
    
    # Set up IK (optional)
    print("\nSetting up IK chains...")
    applier.setup_ik_chains()
    
    print("\n✅ Rig verification and setup complete!")
    print("You can now:")
    print("- Switch to Pose Mode to test joint movements")
    print("- Use the IK targets to animate feet and hands")
    print("- Joint limits will automatically constrain movements")


if __name__ == "__main__":
    main()
