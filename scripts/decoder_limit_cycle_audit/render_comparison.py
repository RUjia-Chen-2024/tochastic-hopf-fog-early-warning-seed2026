"""Render the audit's latent/force-space comparison as publication-style figures."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT)
parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT)
cli = parser.parse_args()
ROOT = cli.input_dir.resolve()
OUT = cli.out_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
data = np.load(ROOT / "representative_trajectories.npz")
record = str(data["record"])
A = data["A"]
offset = data["offset"]
latent = data["latent_ensemble_mean"]
force = data["force_ensemble_mean"]
beta = float(data["beta"])
frequency = float(data["frequency_hz"])
origin_time = float(data["origin_time_s"])
cond = float(np.linalg.cond(A))
inner_radius = np.sqrt(-beta)

theta = np.linspace(0, 2 * np.pi, 800)
outer = np.column_stack((np.cos(theta), np.sin(theta)))
inner = inner_radius * outer
outer_force = outer @ A.T + offset
inner_force = inner @ A.T + offset

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)
palette = {"stable": "#2369a8", "unstable": "#e67e22", "forecast": "#20252b"}

axes[0].plot(outer[:, 0], outer[:, 1], ls="--", lw=2.5, color=palette["stable"], label="Stable frozen-coefficient cycle (r = 1)")
axes[0].plot(inner[:, 0], inner[:, 1], ls="--", lw=2.0, color=palette["unstable"], label=fr"Unstable inner cycle (r = {inner_radius:.3f})")
axes[0].plot(latent[:, 0], latent[:, 1], lw=1.8, color=palette["forecast"], alpha=0.9, label="Saved ensemble-mean forecast (5 s)")
axes[0].scatter(*latent[0], s=32, color="#2ca25f", zorder=4, label="Forecast start")
axes[0].scatter(*latent[-1], s=32, color="#c0392b", zorder=4, label="Forecast end")
axes[0].set_title("Latent Hopf state")
axes[0].set_xlabel(r"Latent coordinate $z_1$")
axes[0].set_ylabel(r"Latent coordinate $z_2$")
axes[0].set_aspect("equal", adjustable="datalim")
axes[0].grid(alpha=0.22)
axes[0].legend(loc="best", fontsize=8.3, framealpha=0.94)

axes[1].plot(outer_force[:, 0], outer_force[:, 1], ls="--", lw=2.5, color=palette["stable"], label="Mapped stable cycle")
axes[1].plot(inner_force[:, 0], inner_force[:, 1], ls="--", lw=2.0, color=palette["unstable"], label="Mapped unstable inner cycle")
axes[1].plot(force[:, 0], force[:, 1], lw=1.8, color=palette["forecast"], alpha=0.9, label="Decoded ensemble-mean forecast (5 s)")
axes[1].scatter(*force[0], s=32, color="#2ca25f", zorder=4, label="Forecast start")
axes[1].scatter(*force[-1], s=32, color="#c0392b", zorder=4, label="Forecast end")
axes[1].scatter(*offset, marker="+", s=75, linewidths=2, color="#111111", zorder=4, label="Affine-map center c")
axes[1].set_title("Original two-channel force space")
axes[1].set_xlabel("Force channel 1 (record units)")
axes[1].set_ylabel("Force channel 2 (record units)")
axes[1].set_aspect("equal", adjustable="datalim")
axes[1].grid(alpha=0.22)
axes[1].legend(loc="best", fontsize=8.3, framealpha=0.94)

fig.suptitle(
    f"Main model affine observation map: {record}\n"
    f"normal test origin {origin_time:.2f} s; frozen β={beta:.3f}, f={frequency:.3f} Hz; κ₂(A)={cond:.1f}",
    fontsize=14,
)
fig.text(
    0.5, -0.015,
    "Dashed curves are deterministic references with parameters frozen at the origin and σ=0; "
    "the solid curve is the saved 128-path ensemble mean (phase averaging can pull it toward the center; it is not one path).",
    ha="center", fontsize=9,
)
fig.savefig(OUT / "latent_vs_force_limit_cycle.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT / "latent_vs_force_limit_cycle.pdf", bbox_inches="tight")
fig.savefig(OUT / "latent_vs_force_limit_cycle.svg", bbox_inches="tight")
print("Wrote PNG, PDF, and SVG figure variants")
