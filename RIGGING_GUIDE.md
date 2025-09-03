# G1 Robot Rigging Guide for Blender

## Overview

This guide explains how to convert MJCF (MuJoCo XML) robot models to rigged GLB files for use in Blender. The system preserves the exact joint hierarchy, constraints, and physical properties defined in the MJCF file.

## Why GLB Format?

We chose GLB (binary glTF 2.0) over text-based formats because:
- **Complete Support**: GLB natively supports meshes, armatures, skinning, and animations
- **Blender Integration**: Excellent built-in import/export support in Blender
- **Efficiency**: Binary format is compact and loads quickly
- **Portability**: Industry-standard format works across many 3D applications
- **Preservation**: Maintains exact joint transforms and hierarchies

## System Components

### 1. MJCF Parser (`mjcf_parser.py`)
- Extracts joint definitions, positions, axes, and limits
- Builds body hierarchy with parent-child relationships
- Handles mesh transformations and material properties

### 2. Armature Builder (`armature_utils.py`)
- Converts MJCF joints to bone structures
- Calculates proper bone orientations and lengths
- Maintains kinematic chain relationships

### 3. Skinning System (`skinning_utils.py`)
- Assigns vertex weights based on proximity to joints
- Handles multi-bone influences (up to 4 per vertex)
- Ensures smooth deformations during animation

### 4. GLB Exporters
- **`glb_armature_exporter.py`**: Basic armature export
- **`glb_rigged_exporter.py`**: Full rigged model with skinned meshes

## Usage Workflow

### Step 1: Export Rigged Model
```bash
python glb_rigged_exporter.py
```
This creates `output/robot_fully_rigged.glb` with:
- Complete mesh geometry
- Armature hierarchy matching MJCF joints
- Proper vertex skinning for deformation
- Material assignments

### Step 2: Export Joint Information
```bash
python export_mjcf_joint_info.py
```
Creates `output/mjcf_joint_info.json` containing joint limits and properties.

### Step 3: Import in Blender
1. Open Blender
2. File → Import → glTF 2.0 (.glb/.gltf)
3. Select `robot_fully_rigged.glb`
4. The model imports with full armature and skinning

### Step 4: Apply Constraints (Optional)
1. Switch to Scripting tab in Blender
2. Load `blender_verify_rig.py`
3. Run the script to:
   - Apply joint rotation limits
   - Set up IK chains for animation
   - Verify rig structure

## Rigging Details

### Joint Mapping
MJCF joints are converted to Blender bones as follows:
- **Hinge joints** → Single-axis rotation constraints
- **Ball joints** → Full rotation (3 DOF)
- **Free joints** → Full 6 DOF movement
- **Slide joints** → Linear translation constraints

### Bone Structure
- Each MJCF joint becomes a bone in the armature
- Bone head = joint position
- Bone tail = points toward child joint or uses joint axis
- Leaf bones are shortened and point inward

### Skinning Weights
Vertices are weighted to bones based on:
- Primary bone (closest joint in body hierarchy)
- Influence bones (nearby joints up to 0.3m away)
- Smooth falloff using geodesic distance
- Maximum 4 bone influences per vertex

## Advanced Features

### IK Chains
The Blender script sets up inverse kinematics for:
- **Feet**: 4-bone chain from ankle to hip
- **Hands**: 3-bone chain from wrist to shoulder
- Includes pole targets for natural knee/elbow bending

### Joint Constraints
Based on MJCF definitions or defaults:
- Rotation limits enforce realistic joint ranges
- Local space constraints maintain proper axes
- Prevents impossible poses during animation

### Animation Ready
The rigged model is ready for:
- Keyframe animation in Blender
- Motion capture retargeting
- Physics simulation export
- Game engine integration

## File Structure
```
output/
├── robot_fully_rigged.glb      # Complete rigged model
├── robot_rigged_armature_vis.glb # Armature visualization
└── mjcf_joint_info.json        # Joint properties
```

## Troubleshooting

### Missing Constraints
If joint limits aren't applied:
1. Ensure `mjcf_joint_info.json` exists
2. Check bone names match joint names
3. Verify constraints in Pose Mode

### Deformation Issues
If meshes deform incorrectly:
1. Check weight painting in Blender
2. Verify influence bones are correct
3. Adjust skinning distance parameters

### Import Problems
If GLB won't import:
1. Ensure Blender 2.80+ (glTF 2.0 support)
2. Check file path is correct
3. Verify GLB file isn't corrupted

## Customization

### Modify Skinning
Edit `skinning_utils.py` to adjust:
- `max_influence_distance`: How far bones can influence vertices
- `weight_falloff_power`: Sharpness of weight transitions
- `min_weight_threshold`: Minimum weight to keep

### Change Bone Display
In `armature_utils.py`:
- Adjust bone lengths for visibility
- Change tail calculation methods
- Modify hierarchy structure

### Export Options
In `glb_rigged_exporter.py`:
- Toggle mesh decimation
- Adjust material properties
- Control vertex data included

## Next Steps

1. **Animation**: Create animations in Blender and export back to GLB
2. **Simulation**: Use the rig for physics-based animation
3. **Game Integration**: Import into Unity, Unreal, or web engines
4. **Motion Retargeting**: Apply mocap data to the rig

The rigged model maintains full fidelity to the original MJCF joint definitions while providing a flexible animation-ready asset for Blender and beyond.
