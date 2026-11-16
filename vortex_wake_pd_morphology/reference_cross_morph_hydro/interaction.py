"""Exact MuJoCo Warp body fluid wrenches, conservatively rebinned to eight axial regions.

The body wrench is a simulator oracle. Its assignment within one rigid segment is
an axial-overlap approximation; this environment has no pressure field and no
independent fin load. Neither flow nor morphology ID is returned to the policy.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp
from mujoco_warp._src import passive, support
from mjlab.utils.lab_api.math import quat_apply_inverse
from fish_morphologies.generate_fish import segment_lengths
from .morphology import parameters

REGIONS = 8
CHANNELS = 4
FEATURE_DIM = REGIONS * CHANNELS + 4
REFERENCE_SPEED_M_S = 0.45


class InteractionSensor:
    def __init__(self, env):
        self.env = env
        m, d = env.sim.wp_model, env.sim.wp_data
        assert m.has_fluid and bool(m.body_fluid_ellipsoid.numpy().sum()) is False, \
            'Sensor contract covers the actual inertia-box fluid model only'
        names = [env.sim.mj_model.body(i).name for i in range(env.sim.mj_model.nbody)]
        self.body_ids = [names.index('fish/head')] + [names.index(f'fish/segment_{i}') for i in range(1, env.n + 1)]
        assert len(self.body_ids) == env.n + 1
        self.wrench = wp.empty((d.nworld, m.nbody), dtype=wp.spatial_vector, device=env.sim.wp_device)
        self.projected = wp.zeros_like(d.qfrc_fluid)
        matrix = []
        lengths = []
        for cfg, _ in parameters(env.n, env.ood):
            segment = np.asarray(segment_lengths(cfg), dtype=np.float64)
            starts = np.r_[0., segment.cumsum()]
            weights = np.zeros((REGIONS, m.nbody), dtype=np.float32)
            for j, body in enumerate(self.body_ids):
                for region in range(REGIONS):
                    a = cfg.body_length * region / REGIONS
                    b = cfg.body_length * (region + 1) / REGIONS
                    weights[region, body] = max(0., min(starts[j+1], b) - max(starts[j], a)) / segment[j]
            assert np.allclose(weights.sum(axis=0)[self.body_ids], 1., atol=1e-6)
            matrix.append(weights)
            lengths.append(cfg.body_length)
        self.weights = torch.as_tensor(np.stack(matrix), device=env.device)
        self.lengths = torch.as_tensor(lengths, dtype=torch.float32, device=env.device)
        self.rho = float(m.opt.density.numpy()[0])
        assert self.rho > 0

    def body_wrenches(self) -> torch.Tensor:
        """World-frame [force xyz, COM torque xyz] for every rigid body."""
        m, d = self.env.sim.wp_model, self.env.sim.wp_data
        wp.launch(passive._fluid_force, dim=(d.nworld, m.nbody), inputs=[
            m.opt.wind, m.opt.density, m.opt.viscosity, m.body_rootid,
            m.body_geomnum, m.body_geomadr, m.body_mass, m.body_inertia,
            m.geom_type, m.geom_size, m.geom_fluid, m.body_fluid_ellipsoid,
            d.xipos, d.ximat, d.geom_xpos, d.geom_xmat, d.subtree_com, d.cvel,
        ], outputs=[self.wrench], device=self.env.sim.wp_device)
        return wp.to_torch(self.wrench)

    def check_projection(self, *, atol: float = 2e-3) -> dict:
        """Compare independent wrench projection with Warp's qfrc_fluid at one state."""
        m, d = self.env.sim.wp_model, self.env.sim.wp_data
        self.body_wrenches()
        self.projected.zero_()
        support.apply_ft(m, d, self.wrench, self.projected, False)
        actual = wp.to_torch(d.qfrc_fluid)
        projected = wp.to_torch(self.projected)
        error = (actual - projected).abs()
        ok = bool(error.max() <= atol)
        return dict(passed=ok, max_abs_error_N=float(error.max()),
                    fluid_l2_N=float(actual.norm()), actuator_l2_N=float(wp.to_torch(d.qfrc_actuator).norm()),
                    projected_l2_N=float(projected.norm()),
                    shape=list(self.body_wrenches().shape))

    def read(self) -> torch.Tensor:
        """50-Hz pre-action feature: 8x(Fx,Fy,Fz,Mz) then executed smoothed wave4."""
        env = self.env
        w = self.body_wrenches()
        morph = env.morph_ids
        weights = self.weights[morph]
        force_w = torch.einsum('erb,ebc->erc', weights, w[:, :, :3])
        torque_w = torch.einsum('erb,ebc->erc', weights, w[:, :, 3:])
        quat = env.scene['fish'].data.root_link_quat_w[:, None, :].expand(-1, REGIONS, -1)
        force_b = quat_apply_inverse(quat, force_w)
        torque_b = quat_apply_inverse(quat, torque_w)
        length = self.lengths[morph]
        force_scale = .5 * self.rho * REFERENCE_SPEED_M_S**2 * length.square()
        force_norm = force_b / force_scale[:, None, None]
        torque_norm = torque_b[:, :, 2:3] / (force_scale * length)[:, None, None]
        physical = torch.cat((force_norm, torque_norm), -1).reshape(env.num_envs, REGIONS * CHANNELS)
        executed = env.action_manager.get_term('wave').smoothed
        result = torch.cat((physical, executed), -1)
        assert result.shape == (env.num_envs, FEATURE_DIM) and torch.isfinite(result).all()
        return result
