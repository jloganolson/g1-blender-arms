# Blender ↔ MuJoCo Puppet: Quick Test

This walks you through verifying the UDP puppet pipeline with and without Blender.

## 0) Environment

```bash
source /home/logan/Projects/g1-blender-arms/.venv/bin/activate
```

## 1) MuJoCo receiver (with viewer)

Default args use the scene XML and 127.0.0.1:31001.

```bash
python deployment/udp_mujoco_receiver.py --viewer
# Optional: force EGL if needed (headless GPU)
# MUJOCO_GL=egl python deployment/udp_mujoco_receiver.py --viewer
```

You should see a MuJoCo viewer window.

## 2a) Send fake data (no Blender)

```bash
python deployment/udp_fake_sender.py --hz 30 --duration 10
# Options:
#   --joints left_elbow_joint,right_elbow_joint
#   --speed 0.5
#   --amp_scale 0.8
```

The robot should move smoothly within MJCF limits.

## 2b) Send from Blender (headless)

```bash
blender -b -P blender_scripts/puppet_panel.py -- output/robot_rigged.glb g1_description/g1_mjx_alt.xml --stream --ip 127.0.0.1 --port 31001 --hz 30
```

This imports the rigged GLB, sets constraints/drivers from MJCF, and streams sliders (defaults 0°). Use the UI variant to move sliders:

```bash
blender -P blender_scripts/puppet_panel.py -- output/robot_rigged.glb g1_description/g1_mjx_alt.xml
# In Blender: N-panel → "Puppet (MJCF)" → move sliders → Start Streaming
```

## Notes
- Receiver expects JSON: `{t, seq, joints:[{name, value_radians}]}`.
- Joint names should match MJCF joint names (exporter ensures this).
- If viewer fails to open, ensure a GL context is available or use `MUJOCO_GL=egl`.
- If MJCF floor/contact errors occur, run the receiver with `g1_description/scene_mjx_alt.xml` (default).
