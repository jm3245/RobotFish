"""Flow-compensated PD for the existing two-action mechanical fish.

Input: [head_x, head_y, yaw, yaw_rate, target_x, target_y, flow_x, flow_y].
Output: [raw steering bias B (rad), tail amplitude offset dA (rad)].
Selected on the even-index 0.40 m/s tuning scenes only.
"""
from __future__ import annotations

import math

import numpy as np


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class FlowCompensatedPD:
    """Original heading-first 6D PD plus a fixed world-frame flow vector."""

    FLOW_GAIN = .5
    NOMINAL_SWIM_SPEED_M_S = .4
    CROSS_P = 1.0
    CROSS_I = .3
    YAW_P = .4
    YAW_D = .15
    TURN_BIAS = .3
    TURN_AMPLITUDE_OFFSET = -.1
    LINE_AMPLITUDE_OFFSET = .1

    def __init__(self, initial_obs8):
        obs = np.asarray(initial_obs8, dtype=float)
        if obs.shape != (8,) or not np.isfinite(obs).all():
            raise ValueError("Expected finite [x,y,yaw,yaw_rate,target_x,target_y,flow_x,flow_y]")
        self.start = obs[:2].copy()
        self.target = obs[4:6].copy()
        self.flow = obs[6:8].copy()
        line = self.target - self.start
        length = float(np.linalg.norm(line))
        if length <= 0:
            raise ValueError("Start and target must differ")
        self.u = line / length
        self.n = np.array([-self.u[1], self.u[0]])
        # Required fish-relative swim direction after compensating for flow.
        required_swim = self.NOMINAL_SWIM_SPEED_M_S * self.u - self.FLOW_GAIN * self.flow
        self.line_heading = math.atan2(required_swim[1], required_swim[0])
        self.turning = False
        self.outer_yaw_filtered = None
        self.inner_yaw_filtered = None
        self.yaw_rate_filtered = 0.0
        self.cross_integral = 0.0

    def _reset_line_state(self):
        self.inner_yaw_filtered = None
        self.yaw_rate_filtered = 0.0
        self.cross_integral = 0.0

    def action(self, obs8):
        obs = np.asarray(obs8, dtype=float)
        if obs.shape != (8,) or not np.isfinite(obs).all():
            raise ValueError("Expected finite 8D observation")
        position = obs[:2]
        yaw, yaw_rate = float(obs[2]), float(obs[3])
        target = obs[4:6]
        # For constant flow this equals the startup value. For a varying
        # measured flow, update the feedforward direction every control step.
        flow = obs[6:8]
        required_swim = self.NOMINAL_SWIM_SPEED_M_S * self.u - self.FLOW_GAIN * flow
        self.line_heading = math.atan2(required_swim[1], required_swim[0])
        self.outer_yaw_filtered = (yaw if self.outer_yaw_filtered is None else
                                   wrap(self.outer_yaw_filtered + .1 * wrap(yaw - self.outer_yaw_filtered)))
        to_target = target - position
        bearing = math.atan2(to_target[1], to_target[0])
        target_error = wrap(bearing - self.outer_yaw_filtered)
        if self.turning:
            if abs(target_error) < .65:
                self.turning = False
                self._reset_line_state()
        elif abs(target_error) > 1.4:
            self.turning = True
        if self.turning:
            bias = -math.copysign(self.TURN_BIAS, target_error)
            return np.array([bias, self.TURN_AMPLITUDE_OFFSET], dtype=np.float32), 1

        self.inner_yaw_filtered = (yaw if self.inner_yaw_filtered is None else
                                   wrap(self.inner_yaw_filtered + .1 * wrap(yaw - self.inner_yaw_filtered)))
        self.yaw_rate_filtered = .9 * self.yaw_rate_filtered + .1 * yaw_rate
        cross = float((position - self.start) @ self.n)
        self.cross_integral = float(np.clip(self.cross_integral + .02 * cross, -8., 8.))
        correction = float(np.clip(self.CROSS_P * cross + self.CROSS_I * self.cross_integral,
                                   -1.4, 1.4))
        desired_heading = wrap(self.line_heading - correction)
        heading_error = wrap(desired_heading - self.inner_yaw_filtered)
        bias = -float(np.clip(self.YAW_P * heading_error - self.YAW_D * self.yaw_rate_filtered,
                              -.3, .3))
        return np.array([bias, self.LINE_AMPLITUDE_OFFSET], dtype=np.float32), 0
