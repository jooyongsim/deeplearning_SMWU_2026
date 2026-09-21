# Comparison with kywind/real2sim-eval

Inspected files:
- `sim/physics/spring_mass_warp.py`
- `sim/physics/phystwin.py`
- `cfg/physics/default.yaml`
- `experiments/utils/create_rigid_phystwin.py`

## Spring topology

This tutorial uses 8 particles and all 28 pairwise links:
12 edges + 12 face diagonals + 4 body diagonals.
The 12 edge links are actuated by changing rest length.

`real2sim-eval`/PhysTwin uses a point-cloud graph constructed by radius/KD-tree
neighbor search. Its default config uses `object_radius: 0.02` and
`object_max_neighbours: 30`. The rigid-object utility uses a denser graph with
`object_radius=0.5`, `object_max_neighbours=50`.

## Spring law

Original tutorial:
`F = k (L-L0) n`

`real2sim-eval`:
`F = clamp(exp(spring_Y), Y_min, Y_max) * (L/L0 - 1) * n`

So Real2Sim uses strain `(L/L0 - 1)` and per-spring stiffness-like parameters.
The revised notebooks default to this strain form, while the non-Warp notebook
also keeps a Hooke option.

## Damping

Both use axial dashpot damping proportional to the relative velocity projected
onto the spring direction. `real2sim-eval` additionally applies global velocity
drag `exp(-dt * drag_damping)`.

Default Real2Sim config:
- `dashpot_damping: 100`
- `drag_damping: 3`

The revised notebooks include both dashpot damping and exponential drag.

## Contact

The earlier tutorial used penalty normal force and viscous tangential damping.

`real2sim-eval` uses velocity/impulse-style collision response with:
- restitution (`collide_elas`)
- Coulomb-like tangential reduction (`collide_fric`)
- ground time-of-impact logic
- signed mesh queries + positional correction for mesh contact
- optional point self-collision via a hash grid

Default config:
- `collide_elas: 0.5`
- `collide_fric: 0.3`
- `self_collision: True`
- `collision_dist: 0.005`

The revised notebooks implement a simplified table-only version:
restitution + Coulomb-like friction + time-of-impact/projection.

## Important scale caveat

Do not copy Real2Sim numerical stiffness/damping constants directly into the
8-node 0.4 m teaching cube. The timestep, graph density, particle spacing,
mass scale, and identified physical parameters differ. The notebooks match the
**structure of the equations** and then use retuned teaching-scale constants.