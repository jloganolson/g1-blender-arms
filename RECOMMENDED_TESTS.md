# 🧪 Recommended Tests for G1 Rigging System

Based on analysis of the overconfident status report, here are the tests you should run to validate what's actually working.

## 🚨 **Critical Tests (System Validation)**

### 1. **Fix Pose Specification First**
```bash
# Test the broken pose system
python -c "
from skinning_utils import create_test_poses
poses = create_test_poses({})
print('Pose format:', type(poses), len(poses))
for pose_name, pose_data in poses:
    print(f'{pose_name}: {type(pose_data)}')
"
```

### 2. **Test Basic Joint Rotation (Direct Method)**
```python
# Create: test_direct_pose.py
import numpy as np
from mjcf_parser import MJCFParser
from armature_utils import ArmatureBuilder
from pose_deformation import PoseDeformer

# Test single joint directly (bypass broken test system)
parser = MJCFParser("g1_description/g1_mjx_alt.xml")
armature_builder = ArmatureBuilder(parser)
armature_builder.compute_bone_positions()

# Test one joint with proper format
pose_angles = {
    "left_wrist_roll_joint": np.radians([45, 0, 0])  # RADIANS not degrees
}

pose_deformer = PoseDeformer(armature_builder.bones)
transforms = pose_deformer.apply_pose(pose_angles)
print(f"Applied transforms: {len(transforms)}")
```

### 3. **Kinematic Chain Validation Test**
```python
# Create: test_kinematic_isolation.py
# Test if ONLY the expected parts move when a joint rotates

expected_chains = {
    'left_wrist_roll_joint': ['forearm', 'wrist', 'hand'],
    'left_elbow_joint': ['forearm', 'wrist', 'hand', 'elbow'],
    'left_shoulder_pitch_joint': ['left_shoulder', 'left_elbow', 'forearm']
}

# Apply single joint, check which meshes deform
# PASS if only expected parts move, FAIL if global deformation
```

### 4. **Displacement Magnitude Test**
```bash
# Test reasonable deformation ranges
python -c "
# Check if displacements are 0.01-0.2 units as claimed
# Current evidence shows 0.004-0.108 range (reasonable)
"
```

### 5. **Vertex Weight Distribution Test**
```python
# Validate claimed '39,229 vertices' and 'proper weight distribution'
from skinning_utils import SkinnedMeshBuilder
# Check weight normalization (should sum to 1.0 per vertex)
# Check influence radius (reasonable bone-to-vertex distances)
```

## 🔧 **Functional Tests (What Actually Works)**

### 1. **MJCF Parsing Validation** ✅
```bash
python mjcf_parser.py g1_description/g1_mjx_alt.xml | grep "total_joints"
# Should show: total_joints: 24
```

### 2. **Armature Generation Check** ✅
```bash
python mjcf_to_glb.py --armature -o test_bones.glb
# Should generate clean bone visualization without protruding cylinders
```

### 3. **GLB Export Validation** ✅
```bash
python mjcf_to_glb.py --skinning -o basic_export.glb
# Should create valid GLB file loadable in 3D viewers
```

## ⚠️ **Tests That Will FAIL (Until Fixed)**

### 1. **Any Single-Joint Tests**
```bash
python test_specific_joint.py  # WILL FAIL
python simple_kinematic_test.py  # WILL FAIL
```
**Reason**: Pose specification system is broken

### 2. **Kinematic Chain Validation**
```bash
# ANY test trying to validate "only specific parts move"
```
**Reason**: Cannot specify single joints, so cannot test isolation

### 3. **End-Effector Isolation**
```bash
# Tests expecting wrist movement to NOT affect torso
```
**Reason**: All poses are multi-joint, causing global deformation

## 🎯 **Success Criteria (Realistic)**

| Component | Status | Test |
|-----------|--------|------|
| MJCF Parsing | ✅ WORKING | `python mjcf_parser.py` |
| Armature Generation | ✅ WORKING | Visual inspection of GLB |
| Vertex Weights | ❓ UNTESTED | Need weight validation script |
| **Pose Specification** | ❌ BROKEN | **Fix first** |
| **Kinematic Chains** | ❌ UNTESTABLE | Depends on pose fix |
| GLB Export | ✅ WORKING | File generation works |
| **Single Joint Control** | ❌ BROKEN | **Core issue** |

## 📋 **Action Plan**

1. **URGENT**: Fix pose specification format (tuples → dicts, degrees → radians)
2. **HIGH**: Create direct joint testing (bypass monkey patching)
3. **MEDIUM**: Validate vertex weight calculations
4. **LOW**: Performance optimization

## 💡 **Key Insights**

1. **The math works** - transformation calculations are correct
2. **The testing is broken** - pose specification never worked
3. **The writer was overconfident** - claimed validation that never occurred
4. **The core system has potential** - infrastructure is solid, just needs working pose control

Run these tests to get an **honest assessment** of what's actually working vs. what the overconfident report claimed.
