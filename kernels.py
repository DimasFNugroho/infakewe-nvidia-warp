"""kernels.py — Warp GPU kernels for Position-Based Dynamics (PBD).

Kernels
-------
kernel_integrate          PBD predict step: apply forces, advance positions
kernel_stretch_even       Stretch constraints for even-index pairs (0,1),(2,3),...
kernel_stretch_odd        Stretch constraints for odd-index  pairs (1,2),(3,4),...
kernel_bend               Bending resistance via skip-one distance constraints
kernel_update_velocity    PBD velocity correction from position delta

Even/odd splitting for stretch ensures no two threads write to the same
particle, making those passes fully conflict-free on GPU.
"""

import warp as wp


@wp.kernel
def kernel_integrate(
    pos:      wp.array(dtype=wp.vec3),
    vel:      wp.array(dtype=wp.vec3),
    prev_pos: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    gravity:  wp.vec3,
    wind:     wp.vec3,
    dt:       float,
    damping:  float,
):
    """Apply gravity + wind, store previous position, integrate forward."""
    i = wp.tid()
    if inv_mass[i] == 0.0:
        prev_pos[i] = pos[i]
        return
    prev_pos[i] = pos[i]
    vel[i]      = vel[i] * damping + (gravity + wind) * dt
    pos[i]      = pos[i] + vel[i] * dt


@wp.kernel
def kernel_stretch_even(
    pos:      wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    rest_len: float,
    stiff:    float,
):
    """Stretch constraints for even-index pairs: (0,1), (2,3), (4,5), ...
    Thread t handles pair (2t, 2t+1) — no shared particles, conflict-free."""
    t = wp.tid()
    a = t * 2
    b = a + 1
    if b >= pos.shape[0]:
        return
    pa = pos[a];  pb = pos[b]
    d    = pb - pa
    dist = wp.length(d)
    if dist < 1.0e-8:
        return
    corr = d * ((dist - rest_len) / dist) * stiff
    wa = inv_mass[a];  wb = inv_mass[b]
    wt = wa + wb
    if wt < 1.0e-8:
        return
    pos[a] = pa + corr * (wa / wt)
    pos[b] = pb - corr * (wb / wt)


@wp.kernel
def kernel_stretch_odd(
    pos:      wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    rest_len: float,
    stiff:    float,
):
    """Stretch constraints for odd-index pairs: (1,2), (3,4), (5,6), ...
    Thread t handles pair (2t+1, 2t+2) — no shared particles, conflict-free."""
    t = wp.tid()
    a = t * 2 + 1
    b = a + 1
    if b >= pos.shape[0]:
        return
    pa = pos[a];  pb = pos[b]
    d    = pb - pa
    dist = wp.length(d)
    if dist < 1.0e-8:
        return
    corr = d * ((dist - rest_len) / dist) * stiff
    wa = inv_mass[a];  wb = inv_mass[b]
    wt = wa + wb
    if wt < 1.0e-8:
        return
    pos[a] = pa + corr * (wa / wt)
    pos[b] = pb - corr * (wb / wt)


@wp.kernel
def kernel_bend(
    pos:      wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    rest_len: float,
    stiff:    float,
):
    """Bending resistance via skip-one distance constraints: (0,2), (1,3), ...
    Low stiffness + many iterations converges well despite benign write races."""
    i = wp.tid()
    j = i + 2
    if j >= pos.shape[0]:
        return
    pa = pos[i];  pb = pos[j]
    d    = pb - pa
    dist = wp.length(d)
    if dist < 1.0e-8:
        return
    corr = d * ((dist - rest_len * 2.0) / dist) * stiff
    wa = inv_mass[i];  wb = inv_mass[j]
    wt = wa + wb
    if wt < 1.0e-8:
        return
    pos[i] = pa + corr * (wa / wt)
    pos[j] = pb - corr * (wb / wt)


@wp.kernel
def kernel_update_velocity(
    pos:      wp.array(dtype=wp.vec3),
    prev_pos: wp.array(dtype=wp.vec3),
    vel:      wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    inv_dt:   float,
):
    """Derive corrected velocity from the position change (standard PBD step)."""
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    vel[i] = (pos[i] - prev_pos[i]) * inv_dt
