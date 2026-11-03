"""Audit the trained 2x2 decoder and compare latent/force limit cycles.

Uses only Python's standard library and NumPy. PyTorch .pt checkpoint tensors
are read from their ZIP/pickle representation without importing or executing
the original model code.
"""

from __future__ import annotations

import csv
import argparse
import io
import json
import math
import pickle
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "unified_seed2026_20261027" / "source_snapshot" / "records"
OUT = Path(__file__).resolve().parent


class StorageType:
    def __init__(self, name: str):
        self.name = name


class StorageRef:
    def __init__(self, key: str, count: int, dtype: str):
        self.key, self.count, self.dtype = key, count, dtype


class TensorRef:
    def __init__(self, storage, offset, shape, strides):
        self.storage = storage
        self.offset = int(offset)
        self.shape = tuple(int(x) for x in shape)
        self.strides = tuple(int(x) for x in strides)


def rebuild_tensor(storage, offset, shape, strides, *unused):
    return TensorRef(storage, offset, shape, strides)


class SafeCheckpointUnpickler(pickle.Unpickler):
    """Allow only ordinary pickle globals needed to read a state_dict."""

    def find_class(self, module, name):
        if module == "torch._utils" and name in {
            "_rebuild_tensor_v2", "_rebuild_tensor", "_rebuild_tensor_v3"
        }:
            return rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        if module == "torch.nn.parameter" and name == "Parameter":
            return lambda data, *args: data
        return super().find_class(module, name)

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or not pid or pid[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent reference: {pid!r}")
        _, dtype, key, _location, count, *rest = pid
        dtype_name = getattr(dtype, "name", str(dtype).split(".")[-1])
        return StorageRef(str(key), int(count), dtype_name)


DTYPES = {
    "FloatStorage": "<f4",
    "DoubleStorage": "<f8",
    "HalfStorage": "<f2",
    "LongStorage": "<i8",
    "IntStorage": "<i4",
    "ShortStorage": "<i2",
    "CharStorage": "i1",
    "ByteStorage": "u1",
    "BoolStorage": "?",
}


def read_checkpoint(path: Path):
    with zipfile.ZipFile(path) as archive:
        pickle_name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        prefix = pickle_name[: -len("data.pkl")]
        payload = SafeCheckpointUnpickler(io.BytesIO(archive.read(pickle_name))).load()
        data_names = set(archive.namelist())

        def tensor_array(value):
            if not isinstance(value, TensorRef):
                return np.asarray(value)
            dtype = DTYPES.get(value.storage.dtype)
            if dtype is None:
                raise TypeError(f"Unsupported checkpoint tensor dtype: {value.storage.dtype}")
            data_path = f"{prefix}data/{value.storage.key}"
            if data_path not in data_names:
                raise KeyError(f"Missing tensor storage {data_path} in {path}")
            with zipfile.ZipFile(path) as storage_archive:
                raw = storage_archive.read(data_path)
            base = np.frombuffer(raw, dtype=dtype)
            # PyTorch records tensor strides in elements, not bytes.
            element_strides = value.strides
            view = np.lib.stride_tricks.as_strided(
                base[value.offset :], shape=value.shape,
                strides=tuple(s * np.dtype(dtype).itemsize for s in element_strides),
            )
            return np.array(view, copy=True)

        return payload, tensor_array


def finite_summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {
        "min": float(a.min()), "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)), "q75": float(np.quantile(a, 0.75)),
        "max": float(a.max()),
    }


def extract_record(folder: Path):
    checkpoint, read_tensor = read_checkpoint(folder / "model.pt")
    state = checkpoint["state_dict"]
    weight = read_tensor(state["decoder.weight"]).astype(float)
    bias_norm = read_tensor(state["decoder.bias"]).astype(float)
    mean = np.asarray(checkpoint["mean"], dtype=float)
    scale = np.asarray(checkpoint["scale"], dtype=float)
    if weight.shape != (2, 2) or mean.shape != (2,) or scale.shape != (2,):
        raise ValueError(f"Unexpected decoder/normalization shapes in {folder.name}")

    # nn.Linear implements q_normalized = W z + b; the training targets were
    # standardized per observed channel: q_raw = scale * q_norm + mean.
    A = np.diag(scale) @ weight
    offset = mean + scale * bias_norm
    svals_w = np.linalg.svd(weight, compute_uv=False)
    svals_a = np.linalg.svd(A, compute_uv=False)
    det_w, det_a = float(np.linalg.det(weight)), float(np.linalg.det(A))

    pred = np.load(folder / "predictions.npz")
    origins = pred["origin_time"]
    betas = pred["current_parameters"][:, 0]
    all_frequency = np.asarray(pred["current_parameters"][:, 1], dtype=float)
    # Select an origin near this record's median predicted current beta.
    target_beta = float(np.median(betas))
    origin_idx = int(np.argmin(np.abs(betas - target_beta)))
    force_mean = np.asarray(pred["mean"][origin_idx], dtype=float)
    latent_mean = np.linalg.solve(A, (force_mean - offset).T).T
    params = np.asarray(pred["current_parameters"][origin_idx], dtype=float)

    # Verify that the affine inverse recovers a force path to numerical precision.
    decoded_roundtrip = latent_mean @ A.T + offset
    residual = float(np.max(np.abs(decoded_roundtrip - force_mean)))
    beta = float(params[0])
    if not -1.0 < beta < 0.0:
        raise ValueError(f"Expected bistable beta in (-1,0), got {beta} in {folder.name}")

    return {
        "record": folder.name,
        "weight": weight, "bias_norm": bias_norm, "mean": mean, "scale": scale,
        "A": A, "offset": offset, "singular_values_W": svals_w,
        "singular_values_A": svals_a, "det_W": det_w, "det_A": det_a,
        "cond_W": float(svals_w.max() / svals_w.min()),
        "cond_A": float(svals_a.max() / svals_a.min()),
        "origin_time": float(origins[origin_idx]), "beta": beta,
        "frequency_hz": float(params[1]), "sigma": float(params[2]),
        "inner_radius": math.sqrt(-beta), "outer_radial_eigenvalue": -2 * (1 + beta),
        "outer_floquet_multiplier": math.exp(-2 * (1 + beta) / float(params[1])),
        "force_mean": force_mean, "latent_mean": latent_mean,
        "roundtrip_max_abs_error": residual,
        "all_beta_origins": np.asarray(betas, dtype=float),
        "all_frequency_origins": all_frequency,
    }


def csv_write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_text(x, y, value, size=14, anchor="middle", weight="normal", color="#243142"):
    import html
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}">{html.escape(str(value))}</text>')


def make_figure(r, target: Path):
    W, H = 1180, 560
    panels = [(70, 90, 475, 390), (660, 90, 475, 390)]
    idx = np.linspace(0, 2 * np.pi, 721)
    beta, ru = r["beta"], r["inner_radius"]
    A, off = r["A"], r["offset"]
    circle = np.column_stack((np.cos(idx), np.sin(idx)))
    inner = ru * circle
    mapped_outer = circle @ A.T + off
    mapped_inner = inner @ A.T + off
    paths = [r["latent_mean"], r["force_mean"]]

    def limits(items, pad=0.12):
        all_points = np.concatenate([x for x in items if len(x)], axis=0)
        low, high = all_points.min(0), all_points.max(0)
        span = np.maximum(high - low, 1e-6)
        # Use common x/y scale in each panel so geometric distortion is visible.
        side = max(span) * (1 + 2 * pad)
        center = (low + high) / 2
        return np.array([center[0] - side / 2, center[1] - side / 2]), np.array([center[0] + side / 2, center[1] + side / 2])

    def draw_panel(plot, curves, labels, title, xlabel, ylabel, colors):
        x, y, w, h = plot
        lo, hi = limits([points for points, *_ in curves])
        scale = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
        ox = x + (w - (hi[0] - lo[0]) * scale) / 2
        oy = y + (h - (hi[1] - lo[1]) * scale) / 2

        def point(p):
            return (ox + (p[0] - lo[0]) * scale, oy + h - (p[1] - lo[1]) * scale)

        elements = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff" stroke="#9aa7b3"/>']
        # Light grid and numeric ticks.
        for fraction in (0.25, 0.5, 0.75):
            gx, gy = x + w * fraction, y + h * fraction
            elements.append(f'<line x1="{gx}" y1="{y}" x2="{gx}" y2="{y+h}" stroke="#e7ebef"/>')
            elements.append(f'<line x1="{x}" y1="{gy}" x2="{x+w}" y2="{gy}" stroke="#e7ebef"/>')
        for points, style, label, color in curves:
            mapped = [point(p) for p in points]
            coords = " ".join(f"{px:.2f},{py:.2f}" for px, py in mapped)
            dash = ' stroke-dasharray="7 5"' if style == "dash" else ""
            width = 2.5 if style != "solid" else 2.2
            elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"{dash}/>')
        elements.append(svg_text(x + w / 2, y - 22, title, 18, weight="bold"))
        elements.append(svg_text(x + w / 2, y + h + 42, xlabel, 13))
        elements.append(svg_text(x - 40, y + h / 2, ylabel, 13))
        # Compact legend.
        ly = y + h - 12 * len(curves) - 12
        for j, (_, style, label, color) in enumerate(curves):
            yy = ly + j * 19
            dash = ' stroke-dasharray="7 5"' if style == "dash" else ""
            elements.append(f'<line x1="{x+12}" y1="{yy}" x2="{x+42}" y2="{yy}" stroke="{color}" stroke-width="2.5"{dash}/>')
            elements.append(svg_text(x + 49, yy + 5, label, 11, anchor="start"))
        return "\n".join(elements)

    latent_curves = [
        (circle, "dash", "Stable cycle r=1", "#1167b1"),
        (inner, "dash", f"Unstable cycle r={ru:.3f} (frozen beta)", "#e67e22"),
        (paths[0], "solid", "Predicted ensemble-mean path (5 s)", "#20242a"),
    ]
    force_curves = [
        (mapped_outer, "dash", "Mapped stable cycle", "#1167b1"),
        (mapped_inner, "dash", "Mapped unstable cycle", "#e67e22"),
        (paths[1], "solid", "Decoded ensemble-mean path (5 s)", "#20242a"),
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(W / 2, 30, f"Latent and force-space limit cycles — {r['record']}", 21, weight="bold"),
        svg_text(W / 2, 56, f"A condition number={r['cond_A']:.3g}; det(A)={r['det_A']:.4g}; frozen beta={beta:.3f}; f={r['frequency_hz']:.3f} Hz", 13),
        draw_panel(panels[0], latent_curves, None, "Latent Hopf state", "Latent coordinate z₁", "Latent coordinate z₂", None),
        draw_panel(panels[1], force_curves, None, "Original two-channel force space", "Force channel 1 (record units)", "Force channel 2 (record units)", None),
        svg_text(W / 2, 548, "The solid path is the decoded ensemble mean; dashed curves are frozen-coefficient deterministic limit-cycle references.", 12),
        "</svg>",
    ]
    target.write_text("\n".join(elements), encoding="utf-8")


def main():
    records = [extract_record(p) for p in sorted(CHECKPOINTS.iterdir()) if (p / "model.pt").is_file()]
    if len(records) != 21:
        raise RuntimeError(f"Expected 21 primary per-record checkpoints, found {len(records)}")

    rows = []
    for r in records:
        A = r["A"]
        row = {
            "record": r["record"],
            "det_W_normalized": r["det_W"], "cond_W_normalized": r["cond_W"],
            "W_singular_max": r["singular_values_W"][0], "W_singular_min": r["singular_values_W"][-1],
            "det_A_raw": r["det_A"], "cond_A_raw": r["cond_A"],
            "A11": A[0, 0], "A12": A[0, 1], "A21": A[1, 0], "A22": A[1, 1],
            "offset_channel1": r["offset"][0], "offset_channel2": r["offset"][1],
            "A_singular_max": r["singular_values_A"][0], "A_singular_min": r["singular_values_A"][-1],
            "beta_near_record_median": r["beta"], "inner_unstable_radius": r["inner_radius"],
            "outer_radial_eigenvalue_per_s": r["outer_radial_eigenvalue"],
            "outer_floquet_multiplier_per_cycle": r["outer_floquet_multiplier"],
            "representative_origin_time_s": r["origin_time"],
            "force_latent_inverse_roundtrip_max_abs": r["roundtrip_max_abs_error"],
        }
        rows.append(row)

    fields = list(rows[0])
    csv_write(OUT / "decoder_invertibility_all21.csv", rows, fields)
    # Select the model whose physical-space condition number is nearest the cohort median.
    median_cond = float(np.median([r["cond_A"] for r in records]))
    representative = min(records, key=lambda r: abs(r["cond_A"] - median_cond))
    make_figure(representative, OUT / "latent_vs_force_limit_cycle.svg")
    np.savez_compressed(
        OUT / "representative_trajectories.npz",
        record=representative["record"], A=representative["A"], offset=representative["offset"],
        latent_ensemble_mean=representative["latent_mean"],
        force_ensemble_mean=representative["force_mean"],
        beta=representative["beta"], frequency_hz=representative["frequency_hz"],
        sigma=representative["sigma"], origin_time_s=representative["origin_time"],
    )

    cond_w = [r["cond_W"] for r in records]
    cond_a = [r["cond_A"] for r in records]
    min_sv_w = [r["singular_values_W"][-1] for r in records]
    min_sv_a = [r["singular_values_A"][-1] for r in records]
    radial_eigs = [r["outer_radial_eigenvalue"] for r in records]
    floquet = [r["outer_floquet_multiplier"] for r in records]
    betas = [r["beta"] for r in records]
    all_origin_betas = np.concatenate([r["all_beta_origins"] for r in records])
    all_origin_frequencies = np.concatenate([r["all_frequency_origins"] for r in records])
    all_origin_radial_eigs = -2 * (1 + all_origin_betas)
    all_origin_floquet = np.exp(-2 * (1 + all_origin_betas) / all_origin_frequencies)
    worst_condition = max(records, key=lambda r: r["cond_A"])
    summary = {
        "source": "unified_seed2026_20261027/source_snapshot per-record model.pt checkpoints",
        "n_records": len(records),
        "observation_map": "q_raw = A_raw z + c; A_raw = diag(scale) @ decoder.weight; c = mean + scale * decoder.bias",
        "all_21_decoder_matrices_invertible_numerically": all(x > 0 for x in min_sv_w),
        "minimum_singular_value_normalized_W": finite_summary(min_sv_w),
        "condition_number_normalized_W": finite_summary(cond_w),
        "minimum_singular_value_raw_A": finite_summary(min_sv_a),
        "condition_number_raw_A": finite_summary(cond_a),
        "condition_number_raw_A_over_100_count": int(sum(x > 100 for x in cond_a)),
        "condition_number_raw_A_over_200_count": int(sum(x > 200 for x in cond_a)),
        "determinant_normalized_W": finite_summary([r["det_W"] for r in records]),
        "determinant_raw_A": finite_summary([r["det_A"] for r in records]),
        "fraction_negative_det": float(np.mean([r["det_A"] < 0 for r in records])),
        "median_current_beta_by_record": finite_summary(betas),
        "current_beta_over_all_normal_test_origins": finite_summary(all_origin_betas),
        "outer_radial_eigenvalue_per_s": finite_summary(radial_eigs),
        "outer_floquet_multiplier_per_cycle": finite_summary(floquet),
        "outer_radial_eigenvalue_over_all_normal_test_origins_per_s": finite_summary(all_origin_radial_eigs),
        "outer_floquet_multiplier_over_all_normal_test_origins": finite_summary(all_origin_floquet),
        "normal_test_origins_with_frozen_beta_in_minus1_0": int(np.sum((all_origin_betas > -1) & (all_origin_betas < 0))),
        "representative_record": representative["record"],
        "representative_condition_number_raw_A": representative["cond_A"],
        "representative_A_raw": representative["A"].tolist(),
        "representative_offset_raw": representative["offset"].tolist(),
        "representative_force_ellipse_semiaxes": representative["singular_values_A"].tolist(),
        "worst_condition_record": worst_condition["record"],
        "worst_condition_number_raw_A": worst_condition["cond_A"],
        "worst_condition_force_ellipse_semiaxes": worst_condition["singular_values_A"].tolist(),
        "worst_condition_inverse_small_axis_gain": float(1 / worst_condition["singular_values_A"][-1]),
        "representative_local_frozen_parameters": {
            "beta": representative["beta"], "frequency_hz": representative["frequency_hz"],
            "sigma": representative["sigma"], "origin_time_s": representative["origin_time"],
        },
        "deterministic_local_stability": {
            "origin": "stable focus for beta < 0, eigenvalue real part beta",
            "inner_cycle": "unstable for -1 < beta < 0, radius sqrt(-beta)",
            "outer_cycle": "stable for -1 < beta < 0, radius 1",
            "outer_radial_eigenvalue_per_s": representative["outer_radial_eigenvalue"],
            "outer_floquet_multiplier_per_cycle": representative["outer_floquet_multiplier"],
            "affine_invariance": "An invertible A conjugates the vector fields; eigenvalues and Floquet multipliers are unchanged.",
        },
        "inverse_roundtrip_max_abs_error_all21": max(r["roundtrip_max_abs_error"] for r in records),
        "caveats": [
            "The ellipse is the affine image of the deterministic frozen-coefficient latent cycle; the main model has time-varying parameters and stochastic forcing.",
            "The plotted solid trajectory is the saved 128-path decoded ensemble mean, inverted through the same affine map; individual paths were not saved in this compact source snapshot.",
            "This checks numerical invertibility and geometric/stability consequences, not physiological identification of the latent axes.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = f"""# Main-model decoder invertibility and limit-cycle geometry

## Result

Across {len(records)} record-specific main-model checkpoints, the 2×2 trained decoder is full rank numerically: every determinant is nonzero and every minimum singular value is positive. However, it is not well-conditioned. In normalized coordinates, the condition number ranges from {summary['condition_number_normalized_W']['min']:.4g} to {summary['condition_number_normalized_W']['max']:.4g}; in original force units it ranges from {summary['condition_number_raw_A']['min']:.4g} to {summary['condition_number_raw_A']['max']:.4g}, with {summary['condition_number_raw_A_over_100_count']} of 21 records above 100 and {summary['condition_number_raw_A_over_200_count']} above 200. Thus inversion is possible, but the relative sensitivity can be as high as κ₂; the worst-conditioned record is {worst_condition['record']} (κ₂={worst_condition['cond_A']:.2f}, inverse gain along its compressed axis {summary['worst_condition_inverse_small_axis_gain']:.2f} latent units per force unit; force-space ellipse semiaxes {worst_condition['singular_values_A'][0]:.3f}, {worst_condition['singular_values_A'][1]:.3f}).

The physical-space mapping is

`q_raw = A_raw z + c`, `A_raw = diag(scale) @ decoder.weight`, `c = mean + scale * decoder.bias`.

The returned inverse map reconstructs all saved ensemble-mean force paths with maximum absolute round-trip error {summary['inverse_roundtrip_max_abs_error_all21']:.3g} in record units.

## Shape and stability

For fixed coefficients and no noise, the latent radial equation is `dr/dt = r [ beta + (1-beta) r^2 - r^4 ]`. For `-1 < beta < 0`, it has a stable outer cycle `r=1` and an unstable inner cycle `r=sqrt(-beta)`. A full-rank affine decoder maps both circles to ellipses, centered at `c`. The singular values of `A_raw` give the ellipse semiaxes, and their ratio is the ellipse aspect ratio.

For the representative median-condition model `{representative['record']}`, the selected normal test origin is t={representative['origin_time']:.2f} s with frozen local coefficients beta={representative['beta']:.4f}, f={representative['frequency_hz']:.4f} Hz, sigma={representative['sigma']:.4f}. The force-space outer-cycle semiaxes are {representative['singular_values_A'][0]:.3f} and {representative['singular_values_A'][1]:.3f} record units (aspect ratio {representative['cond_A']:.2f}); its center is ({representative['offset'][0]:.2f}, {representative['offset'][1]:.2f}). The outer cycle's radial eigenvalue is {representative['outer_radial_eigenvalue']:.5f} s⁻¹ and its transverse Floquet multiplier per cycle is {representative['outer_floquet_multiplier']:.5f}, both indicating local attraction. These quantities are identical after transforming to force coordinates because `A_raw` gives a smooth invertible conjugacy. The ellipse changes Euclidean distances and the apparent direction-dependent contraction, but not asymptotic stability or Floquet multipliers. Across the saved normal test origins, the frozen-beta reference remains in `(-1,0)` for {summary['normal_test_origins_with_frozen_beta_in_minus1_0']} origins; the pooled beta range is {summary['current_beta_over_all_normal_test_origins']['min']:.3f} to {summary['current_beta_over_all_normal_test_origins']['max']:.3f}. The pooled outer radial eigenvalue spans {summary['outer_radial_eigenvalue_over_all_normal_test_origins_per_s']['min']:.3f} to {summary['outer_radial_eigenvalue_over_all_normal_test_origins_per_s']['max']:.3f} s⁻¹, while all transverse Floquet multipliers remain below 1 (range {summary['outer_floquet_multiplier_over_all_normal_test_origins']['min']:.3g}–{summary['outer_floquet_multiplier_over_all_normal_test_origins']['max']:.3f}).

## Interpretation limit

The stability calculation freezes the coefficients at one origin and sets diffusion to zero. The actual model predicts time-varying beta/frequency/noise and simulates an SDE, so its realized forecasts are not closed deterministic limit cycles; stochastic paths can fluctuate or switch between the two radial basins. The solid line in the figure is the saved 128-path ensemble-mean forecast over five seconds. The dashed circles/ellipses are the corresponding local deterministic reference cycles. In an invertible affine map, isotropic latent diffusion becomes anisotropic/correlated in force coordinates with covariance proportional to `A_raw A_rawᵀ`.

This is evidence that the trained observation map retains both latent directions and allows the latent orbit to be visualized in the recorded two-channel space. It does not establish that the latent coordinates or learned vector field uniquely identify physiology.

## Files

- `decoder_invertibility_all21.csv`: determinant, singular values, condition numbers, effective raw-space matrices, local stability values.
- `latent_vs_force_limit_cycle.svg`: latent phase-plane reference cycles and corresponding decoded force-space ellipses, with one saved ensemble-mean forecast.
- `representative_trajectories.npz`: the affine map and representative mean paths.
- `summary.json`: machine-readable pooled audit and caveats.
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    print(json.dumps({
        "n_records": len(records), "representative": representative["record"],
        "cond_W": summary["condition_number_normalized_W"],
        "cond_A": summary["condition_number_raw_A"],
        "min_singular_value_W": summary["minimum_singular_value_normalized_W"],
        "min_singular_value_A": summary["minimum_singular_value_raw_A"],
        "max_roundtrip_error": summary["inverse_roundtrip_max_abs_error_all21"],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINTS,
                        help="Directory containing one subdirectory per record, with model.pt and predictions.npz")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    cli = parser.parse_args()
    CHECKPOINTS = cli.checkpoint_root.resolve()
    OUT = cli.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    main()
