"""Frozen Baseline PD waveform plus explicit whole-fish COM velocity feedback."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

BASELINE = Path(__file__).resolve().parent.parent / "01_Baseline_Controller"
sys.path.insert(0, str(BASELINE))
from core import Controller as BaselineController, wrap  # noqa: E402


def fish_com_velocity_world(e) -> np.ndarray:
    """World-frame velocity of the entire fish's subtree center of mass."""
    jac = np.empty((3, e.model.nv), dtype=float)
    mujoco.mj_jacSubtreeCom(e.model, e.data, jac, e.head)
    return (jac @ e.data.qvel)[:2]


def head_origin_velocity_world(e) -> np.ndarray:
    """World-frame head body-origin velocity; diagnostic, never feedback."""
    jacp = np.empty((3, e.model.nv), dtype=float)
    jacr = np.empty((3, e.model.nv), dtype=float)
    mujoco.mj_jacBody(e.model, e.data, jacp, jacr, e.head)
    return (jacp @ e.data.qvel)[:2]


class FishVelocityController(BaselineController):
    MODES = ("body_lateral_error", "target_cross_track", "velocity_heading")

    def __init__(self, params, *, mode="body_lateral_error", gain=0.8,
                 velocity_filter=0.1, correction_limit=0.2):
        super().__init__(params)
        if mode not in self.MODES or gain < 0 or not 0 < velocity_filter <= 1 or correction_limit < 0:
            raise ValueError("Invalid velocity feedback configuration")
        self.mode = mode
        self.gain = float(gain)
        self.velocity_filter = float(velocity_filter)
        self.correction_limit = float(correction_limit)
        self.filtered_velocity = None
        self.diagnostics = {}

    def action(self, e):
        # The position, heading, wind compensation and entire waveform below
        # match frozen Baseline Controller.action. Only 'correction' is added
        # to the raw steering term before its original clamp/sign/filter.
        p = self.p
        pos = e.data.xpos[e.head]
        R = e.data.xmat[e.head].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        delta = e.target - pos[:2]
        distance = np.linalg.norm(delta)
        if self.filtered_yaw is None:
            self.filtered_yaw = yaw
        self.filtered_yaw = wrap(self.filtered_yaw + p.get("yaw_filter", .1) * wrap(yaw - self.filtered_yaw))
        self.yaw_rate = .9 * self.yaw_rate + .1 * float(e.data.qvel[5])
        wind = e.model.opt.wind[:2]
        speed = p.get("speed", .25)
        desired = speed * delta / max(distance, 1e-8) - p.get("flow_gain", 0.) * wind
        heading = math.atan2(desired[1], desired[0])
        error = wrap(heading - self.filtered_yaw)

        v_com = fish_com_velocity_world(e)
        if self.filtered_velocity is None:
            self.filtered_velocity = v_com.copy()
        else:
            alpha = self.velocity_filter
            self.filtered_velocity = (1 - alpha) * self.filtered_velocity + alpha * v_com
        target_direction = delta / max(distance, 1e-8)
        desired_ground = speed * target_direction
        velocity_error = desired_ground - self.filtered_velocity
        if self.mode == "body_lateral_error":
            normal = np.array([-math.sin(self.filtered_yaw), math.cos(self.filtered_yaw)])
            correction_unclipped = self.gain * float(np.dot(velocity_error, normal))
        elif self.mode == "target_cross_track":
            normal = np.array([-target_direction[1], target_direction[0]])
            correction_unclipped = self.gain * float(np.dot(velocity_error, normal))
        else:
            travel_speed = float(np.linalg.norm(self.filtered_velocity))
            travel_heading = math.atan2(self.filtered_velocity[1], self.filtered_velocity[0])
            target_heading = math.atan2(target_direction[1], target_direction[0])
            speed_gate = min(travel_speed / .1, 1.)
            correction_unclipped = self.gain * speed_gate * wrap(target_heading - travel_heading)
        correction = float(np.clip(correction_unclipped, -self.correction_limit, self.correction_limit))
        raw_bias = np.clip(p.get("kp", .3) * error + correction - p.get("kd", .15) * self.yaw_rate,
                           -p.get("bias_limit", .3), p.get("bias_limit", .3))
        raw_bias *= p.get("sign", 1.)
        self.last_bias = .85 * self.last_bias + .15 * raw_bias
        self.diagnostics = {
            "fish_com_vx_m_s": float(v_com[0]), "fish_com_vy_m_s": float(v_com[1]),
            "filtered_com_vx_m_s": float(self.filtered_velocity[0]),
            "filtered_com_vy_m_s": float(self.filtered_velocity[1]),
            "water_vx_m_s": float(wind[0]), "water_vy_m_s": float(wind[1]),
            "relative_vx_m_s": float(v_com[0] - wind[0]),
            "relative_vy_m_s": float(v_com[1] - wind[1]),
            "desired_ground_vx_m_s": float(desired_ground[0]),
            "desired_ground_vy_m_s": float(desired_ground[1]),
            "desired_water_relative_vx_m_s": float(desired[0]),
            "desired_water_relative_vy_m_s": float(desired[1]),
            "heading_error_rad": float(error), "velocity_correction_rad": correction,
            "raw_steering_bias_rad": float(raw_bias),
            "filtered_steering_bias_rad": float(self.last_bias),
        }

        slowdown = np.clip(distance / p.get("slow_radius", .8), p.get("min_frequency_ratio", .4), 1.)
        self.phase += 2 * np.pi * p.get("frequency", 1.5) * slowdown * e.dt
        amplitude = p.get("amplitude", .5)
        profile = p.get("head_ratio", .5) + (1 - p.get("head_ratio", .5)) * self.index / 3
        q = self.last_bias + amplitude * profile * np.sin(self.phase - self.index * p.get("lag", 1.0))
        ramp = min((e.age + 1) * e.dt / .4, 1.)
        return np.clip(q * ramp, -.75, .75)
