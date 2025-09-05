from armature_exporter.rigged_gltf_exporter import create_rigged_glb

full_g1_mjcf_path = "g1_description/g1_mjx_alt.xml"
existing_joints = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
]
create_rigged_glb(
    mjcf_path=full_g1_mjcf_path,
    output_path="output/robot_rigged_g1_full.glb",
    target_joints=existing_joints
)