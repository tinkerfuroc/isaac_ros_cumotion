# SPDX-License-Identifier: Apache-2.0
"""Sizing rule for cuRobo's interpolated-trajectory buffer (``interpolation_steps``).

cuRobo interpolates every optimized trajectory into a fixed buffer of
``interpolation_steps`` rows and evaluates collision / self-collision / FK
metrics over that whole buffer, so the buffer is the planner's largest VRAM
term (batch x steps x n_spheres). The number of rows a plan actually needs is
bounded: ``calculate_traj_steps`` uses ``ceil((horizon - 1) * opt_dt /
interpolation_dt)`` and ``opt_dt`` is clamped to ``maximum_trajectory_dt``
(``calculate_dt_fixed``). Time dilation re-interpolates into a fresh,
exact-size buffer and does not touch this one.

If a plan ever exceeded the buffer, cuRobo would warn and regrow it; but with
``CUROBO_TORCH_CUDA_GRAPH_RESET`` unset (the default) the metrics CUDA graph
is only invalidated, and a *later* plan that fits the old buffer then fails
with "cuda graph is invalid". That intermittent failure is why the planner
node refuses to start with an undersized buffer instead of relying on the
regrow path.
"""
import math


def required_interpolation_steps(trajopt_tsteps: int, maximum_trajectory_dt: float,
                                 interpolation_dt: float) -> int:
    """Largest interpolated length cuRobo can produce for this configuration.

    Mirrors ``curobo.util.trajectory.calculate_traj_steps`` with ``opt_dt`` at
    its clamp ceiling ``maximum_trajectory_dt``.
    """
    if trajopt_tsteps < 2:
        raise ValueError(f'trajopt_tsteps must be >= 2, got {trajopt_tsteps}')
    if interpolation_dt <= 0.0 or maximum_trajectory_dt <= 0.0:
        raise ValueError('interpolation_dt and maximum_trajectory_dt must be > 0')
    return int(math.ceil((trajopt_tsteps - 1) * maximum_trajectory_dt / interpolation_dt))


def check_interpolation_steps(interpolation_steps: int, trajopt_tsteps: int,
                              maximum_trajectory_dt: float, interpolation_dt: float) -> int:
    """Return the required length; raise ValueError if the buffer is smaller."""
    required = required_interpolation_steps(trajopt_tsteps, maximum_trajectory_dt,
                                            interpolation_dt)
    if interpolation_steps < required:
        raise ValueError(
            f'interpolation_steps={interpolation_steps} is smaller than the '
            f'{required} rows a plan can need with num_trajopt_time_steps='
            f'{trajopt_tsteps}, maximum_trajectory_dt={maximum_trajectory_dt}, '
            f'interpolation_dt={interpolation_dt} (ceil((tsteps-1)*max_dt/dt)). '
            'An overflow regrows the buffer but leaves the metrics CUDA graph '
            'invalid, so later plans fail intermittently; raise interpolation_steps '
            '(each row costs batch x n_spheres of VRAM) or lower maximum_trajectory_dt.')
    return required
