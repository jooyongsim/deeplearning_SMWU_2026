# Soft Cube Goal-Pose RL

- 01_goal_pose_nonwarp.ipynb: NumPy/Gymnasium/PyTorch
- 02_goal_pose_warp.ipynb: NVIDIA Warp + vectorized PyTorch PPO
- COMPARISON_real2sim.md: comparison with kywind/real2sim-eval

Default goal: COM=(0.55, 0.25, SIDE/2), yaw=+60 deg.
Orientation is the Kabsch best-fit rotation of the deformable cube.
Table contact uses restitution + Coulomb-like friction.