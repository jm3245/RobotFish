"""Six-input heading-first PD controller; no COM-velocity dependency.

Input at every step: [fish_x, fish_y, yaw, yaw_rate, target_x, target_y].
Output: [raw steering bias B, amplitude offset delta_A].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Parameters6D:
    enter_angle: float = 1.4
    exit_angle: float = .65
    turn_bias: float = .3
    turn_delta_a: float = -.1
    cross_p: float = 1.
    cross_i: float = .3
    yaw_p: float = .4
    yaw_d: float = .15
    line_delta_a: float = .05


class HeadingFirst6DController:
    def __init__(self, initial_obs6, config: Parameters6D = Parameters6D()):
        obs = np.asarray(initial_obs6, dtype=float)
        if obs.shape != (6,) or not np.isfinite(obs).all():
            raise ValueError("Expected finite 6D [fish_x, fish_y, yaw, yaw_rate, target_x, target_y]")
        self.start = obs[:2].copy()
        self.target = obs[4:6].copy()
        line = self.target - self.start
        length = float(np.linalg.norm(line))
        if length <= 0:
            raise ValueError("Initial fish head and target must differ")
        self.u = line / length
        self.n = np.array([-self.u[1], self.u[0]])
        self.line_heading = math.atan2(self.u[1], self.u[0])
        self.g = config
        self.turning = False
        self.outer_yaw_filtered = None
        self.inner_yaw_filtered = None
        self.yaw_rate_filtered = 0.
        self.cross_integral = 0.

    def _reset_line_state(self):
        self.inner_yaw_filtered = None
        self.yaw_rate_filtered = 0.
        self.cross_integral = 0.

    def action(self, obs6):
        obs = np.asarray(obs6, dtype=float)
        if obs.shape != (6,) or not np.isfinite(obs).all():
            raise ValueError("Expected finite 6D [fish_x, fish_y, yaw, yaw_rate, target_x, target_y]")
        position = obs[:2]
        yaw, yaw_rate = float(obs[2]), float(obs[3])
        target = obs[4:6]

        self.outer_yaw_filtered = (yaw if self.outer_yaw_filtered is None else
                                   wrap(self.outer_yaw_filtered + .1 * wrap(yaw - self.outer_yaw_filtered)))
        to_target = target - position
        bearing = math.atan2(to_target[1], to_target[0])
        target_error = wrap(bearing - self.outer_yaw_filtered)
        if self.turning:
            if abs(target_error) < self.g.exit_angle:
                self.turning = False
                self._reset_line_state()
        elif abs(target_error) > self.g.enter_angle:
            self.turning = True
        if self.turning:
            bias = -math.copysign(self.g.turn_bias, target_error)
            return np.array([bias, self.g.turn_delta_a], dtype=np.float32), 1

        # These operations depend only on the six scalar inputs and controller
        # state. There is no COM velocity read, filter, gain, or diagnostic.
        if self.inner_yaw_filtered is None:
            self.inner_yaw_filtered = yaw
        else:
            self.inner_yaw_filtered = wrap(self.inner_yaw_filtered + .1 * wrap(yaw - self.inner_yaw_filtered))
        self.yaw_rate_filtered = .9 * self.yaw_rate_filtered + .1 * yaw_rate
        cross = float((position - self.start) @ self.n)
        self.cross_integral = float(np.clip(self.cross_integral + .02 * cross, -8., 8.))
        correction = float(np.clip(self.g.cross_p * cross + self.g.cross_i * self.cross_integral,
                                   -1.4, 1.4))
        desired_heading = wrap(self.line_heading - correction)
        heading_error = wrap(desired_heading - self.inner_yaw_filtered)
        bias = -float(np.clip(self.g.yaw_p * heading_error - self.g.yaw_d * self.yaw_rate_filtered,
                              -.3, .3))
        return np.array([bias, self.g.line_delta_a], dtype=np.float32), 0
