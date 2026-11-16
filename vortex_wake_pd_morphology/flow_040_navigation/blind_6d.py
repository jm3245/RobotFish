"""Six-observation PD navigation with an internal drift estimate.

Inputs are strictly [x, y, yaw, yaw_rate, target_x, target_y]. The
controller never receives the simulated flow vector or scene metadata.
"""
from __future__ import annotations

import math
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class BlindAdaptive6DPD:
    def __init__(self, initial_obs6, *, speed=.54, drift_gain=.5,
                 estimate_tau=1.0, cross_p=1.0, cross_i=.3,
                 switch_radius=1.0):
        o = np.asarray(initial_obs6, dtype=float)
        if o.shape != (6,) or not np.isfinite(o).all():
            raise ValueError("Expected finite 6D observation")
        self.start = o[:2].copy()
        line = o[4:6] - self.start
        self.u = line / np.linalg.norm(line)
        self.n = np.array([-self.u[1], self.u[0]])
        self.speed = speed
        self.drift_gain = drift_gain
        self.alpha = .02 / (estimate_tau + .02)
        self.cross_p, self.cross_i = cross_p, cross_i
        self.switch_radius = switch_radius
        self.prev_pos = None
        self.yaw_filt = None
        self.rate_filt = 0.0
        self.flow_hat = np.zeros(2)
        self.cross_integral = 0.0
        self.turning = False
        self.near = False
        self.steps = 0

    def action(self, obs6):
        o = np.asarray(obs6, dtype=float)
        if o.shape != (6,) or not np.isfinite(o).all():
            raise ValueError("Expected finite 6D observation")
        pos, yaw, yaw_rate, target = o[:2], float(o[2]), float(o[3]), o[4:6]
        self.yaw_filt = yaw if self.yaw_filt is None else wrap(
            self.yaw_filt + .1 * wrap(yaw - self.yaw_filt))
        self.rate_filt = .9 * self.rate_filt + .1 * yaw_rate
        if self.prev_pos is not None and not self.turning and self.steps > 20:
            ground_velocity = (pos - self.prev_pos) / .02
            body_direction = np.array([math.cos(yaw), math.sin(yaw)])
            residual = ground_velocity - self.speed * body_direction
            self.flow_hat = (1 - self.alpha) * self.flow_hat + self.alpha * residual
            self.flow_hat = np.clip(self.flow_hat, -.6, .6)
        self.prev_pos = pos.copy()
        self.steps += 1

        to_target = target - pos
        distance = float(np.linalg.norm(to_target))
        bearing = math.atan2(to_target[1], to_target[0])
        if distance < self.switch_radius:
            self.near = True

        if self.near:
            d = to_target / max(distance, 1e-9)
            swim = .35 * d - self.flow_hat
            desired = math.atan2(swim[1], swim[0])
            error = wrap(desired - self.yaw_filt)
            bias = -float(np.clip(.5 * error - .08 * self.rate_filt, -.3, .3))
            return np.array([bias, -.1 if abs(error) > 1.4 else .1], np.float32), int(abs(error) > 1.4)

        target_error = wrap(bearing - self.yaw_filt)
        if self.turning:
            if abs(target_error) < .65:
                self.turning = False
                self.cross_integral = 0.0
        elif abs(target_error) > 1.4:
            self.turning = True
        if self.turning:
            return np.array([-.3 * math.copysign(1., target_error), -.1], np.float32), 1

        cross = float((pos - self.start) @ self.n)
        self.cross_integral = float(np.clip(self.cross_integral + .02 * cross, -8, 8))
        correction = float(np.clip(self.cross_p * cross + self.cross_i * self.cross_integral, -1.4, 1.4))
        swim = .4 * self.u - self.drift_gain * self.flow_hat
        desired = wrap(math.atan2(swim[1], swim[0]) - correction)
        error = wrap(desired - self.yaw_filt)
        bias = -float(np.clip(.4 * error, -.3, .3))
        return np.array([bias, .1], np.float32), 0
