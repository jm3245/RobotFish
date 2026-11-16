"""Plot reproducible upstream benchmark results from the six trial CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CAPS = (.38, .42, .75)
CONTROLLERS = ("blind_6d", "heading_first_6d")


def read_rows(controller, cap):
    with (HERE / f"{controller}_A{str(cap).replace('.', 'p')}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    x = np.arange(3)
    width = .19
    labels = {"blind_6d": "Adaptive 6D PD", "heading_first_6d": "Heading-first 6D PD"}
    for j, controller in enumerate(CONTROLLERS):
        for k, metric in enumerate(("cylinder_success", "point_success")):
            values = [sum(int(r[metric]) for r in read_rows(controller, cap)) for cap in CAPS]
            pos = x + (j * 2 + k - 1.5) * width
            ax.bar(pos, values, width, label=f"{labels[controller]} - {'cylinder' if k == 0 else 'point'}",
                   color=("#168a65" if j == 0 else "#a65842"), alpha=(1 if k == 0 else .5))
            for xx, value in zip(pos, values):
                ax.text(xx, value + 1.5, str(value), ha="center", va="bottom", fontsize=8)
    ax.set(xticks=x, xticklabels=[f"{c:.2f}" for c in CAPS], xlabel="Maximum commanded tail amplitude (rad)",
           ylabel="Arrivals / 100", ylim=(0, 108),
           title="0.4 m/s upstream only; same 100 cases; 2.5 Hz")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.savefig(HERE / "arrival_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True, layout="constrained")
    angles = [-54,-42,-30,-18,-6,6,18,30,42,54]
    distances = [1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
    for i, controller in enumerate(CONTROLLERS):
        for j, cap in enumerate(CAPS):
            arr = np.full((10, 10), np.nan)
            for row in read_rows(controller, cap):
                arr[angles.index(int(row["angle_deg"])), distances.index(float(row["distance_m"]))] = int(row["cylinder_success"])
            axes[i,j].imshow(arr, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            axes[i,j].set_title(f"{labels[controller]} | A={cap}", fontsize=9)
            axes[i,j].set_xticks(range(10), [f"{d:g}" for d in distances], rotation=50, fontsize=7)
            axes[i,j].set_yticks(range(10), [f"{a:g}" for a in angles], fontsize=7)
    fig.supxlabel("Target distance (m)")
    fig.supylabel("Target bearing (deg)")
    fig.savefig(HERE / "cylinder_success_map.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
