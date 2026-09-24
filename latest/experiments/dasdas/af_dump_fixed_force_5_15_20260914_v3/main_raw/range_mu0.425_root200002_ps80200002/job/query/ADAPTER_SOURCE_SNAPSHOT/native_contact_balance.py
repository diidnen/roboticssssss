"""Readout adapter for the installed PhysX integrator; no physics mutations.

Subtract the simulator's actual gravity/damping/velocity-limit external load,
not idealized gravity alone. This reconstructs net environmental contact load.
Caller must supply pre-step COM velocity and any explicitly applied link force.
It is NOT a per-contact-point friction solver or a replacement AF model.
"""
import numpy as np

SOURCE = 'https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/105.1-physx-5.3.1/physx/source/lowleveldynamics/src/DyFeatherstoneArticulation.cpp'


def contact_force(reading, before_velocity_world, dt, *, applied_link_force_world=(0.,0.,0.)):
    if not np.isfinite(dt) or dt <= 0: raise ValueError('Invalid physics timestep')
    velocity = np.asarray(before_velocity_world,dtype=float)
    applied = np.asarray(applied_link_force_world,dtype=float)
    if velocity.shape != (3,) or applied.shape != (3,) or not np.isfinite(velocity).all() or not np.isfinite(applied).all():
        raise ValueError('Real pre-step velocity and explicitly applied load required')
    mass = float(reading['mass_kg'])
    gravity = np.asarray(reading['gravity_world'],dtype=float)
    damping = min(max(float(reading['linear_damping']),0.),1/dt)
    max_speed = float(reading['max_linear_velocity'])
    speed = np.linalg.norm(velocity)
    scaling = max(0.,1-max_speed/speed) if speed else 0.
    external_load = (mass*gravity+applied)*(1-damping*dt)-mass*velocity*(damping+scaling/dt)
    return mass*np.asarray(reading['com_linear_acceleration'])-np.asarray(reading['joint_force_world_n'])-external_load
