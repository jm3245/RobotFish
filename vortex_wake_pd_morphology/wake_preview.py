"""Kinematic cylinder-wake preview; not coupled to MuJoCo fish forces.

This is a qualitative, regularized vortex approximation for experiment setup.
It is NOT the paper's IBAMR Navier-Stokes solution or its exact VS formula.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

HERE = Path(__file__).resolve().parent


def velocity(xy: np.ndarray, t: float, *, u: float = .4, diameter: float = .7,
             wavelength: float | None = None, circulation: float | None = None,
             core_radius: float | None = None) -> np.ndarray:
    """Return a sampleable 2D velocity field in m/s around a cylinder.

    A potential-flow cylinder term and alternating smooth point vortices make
    a useful visualization. It omits viscosity, no-slip and two-way coupling.
    """
    p = np.asarray(xy, dtype=float)
    if p.shape[-1] != 2:
        raise ValueError("xy must have trailing dimension 2")
    if u <= 0 or diameter <= 0:
        raise ValueError("u and diameter must be positive")
    wavelength = 4 * diameter if wavelength is None else wavelength
    circulation = 3 * u * diameter if circulation is None else circulation
    core_radius = .15 * diameter if core_radius is None else core_radius
    if wavelength <= 0 or core_radius <= 0:
        raise ValueError("wavelength and core_radius must be positive")
    a = diameter / 2
    x, y = p[..., 0], p[..., 1]
    r2 = np.maximum(x*x + y*y, a*a)
    theta = np.arctan2(y, x)
    out = np.stack((u*(1 - a*a/r2*np.cos(2*theta)),
                    -u*a*a/r2*np.sin(2*theta)), axis=-1)

    # Alternating, regularized vortices convect to the right at 0.75 U.
    # The x>0 wake placement is chosen for visualization, not CFD calibration.
    spacing = wavelength / 2
    for k in range(12):
        vx = a + .28*diameter + k*spacing + .75*u*t
        vy = (.2*diameter) * (1 if k % 2 == 0 else -1)
        gamma = circulation * (1 if k % 2 == 0 else -1)
        dx, dy = x - vx, y - vy
        d2 = dx*dx + dy*dy
        factor = gamma/(2*math.pi) * (-np.expm1(-d2/(core_radius*core_radius))) / np.maximum(d2, 1e-12)
        out[..., 0] += -factor*dy
        out[..., 1] += factor*dx
    out = np.where((x*x + y*y < a*a)[..., None], 0., out)
    return out


def gradient(xy: np.ndarray, t: float, *, step: float = .01, **kwargs) -> np.ndarray:
    """Return d(u_x,u_y)/d(x,y), shape (..., 2, 2), finite differences."""
    p = np.asarray(xy, dtype=float)
    if step <= 0:
        raise ValueError("step must be positive")
    ex, ey = np.array([step, 0.]), np.array([0., step])
    gx = (velocity(p+ex, t, **kwargs) - velocity(p-ex, t, **kwargs))/(2*step)
    gy = (velocity(p+ey, t, **kwargs) - velocity(p-ey, t, **kwargs))/(2*step)
    return np.stack((gx, gy), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", type=float, default=.4)
    parser.add_argument("--diameter", type=float, default=.7)
    parser.add_argument("--time", type=float, default=1.)
    args = parser.parse_args()
    x = np.linspace(-1.2, 5., 165)
    y = np.linspace(-1.6, 1.6, 90)
    xx, yy = np.meshgrid(x, y)
    field = velocity(np.stack((xx, yy), axis=-1), args.time,
                     u=args.flow, diameter=args.diameter)
    speed = np.linalg.norm(field, axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), layout="constrained")
    ax = axes[0]
    ax.quiver(xx[::5, ::6], yy[::5, ::6], field[::5, ::6, 0], field[::5, ::6, 1],
              speed[::5, ::6], cmap="viridis", clim=(0, max(.8, speed.max())),
              scale=14, width=.003)
    ax.set_title("Exploratory cylinder + smooth vortex field")
    ax = axes[1]
    im = ax.pcolormesh(xx, yy, speed, shading="auto", cmap="viridis",
                       vmin=0, vmax=max(.8, np.percentile(speed, 99)))
    fig.colorbar(im, ax=ax, label="Local water speed (m/s)")
    ax.set_title("Spatially varying speed; analytical preview only")
    for ax in axes:
        ax.add_patch(Circle((0, 0), args.diameter/2, color="0.2", zorder=5))
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_aspect("equal")
        ax.set_xlabel("World x (m)")
        ax.set_ylabel("World y (m)")
    fig.suptitle(f"Background flow {args.flow:.2f} m/s · cylinder diameter {args.diameter:.2f} m · t={args.time:.1f} s")
    output = HERE / "wake_field_preview.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
