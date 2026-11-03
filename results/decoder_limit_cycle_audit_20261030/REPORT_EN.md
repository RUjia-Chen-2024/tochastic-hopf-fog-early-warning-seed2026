# Main-model decoder invertibility and limit-cycle geometry

## Result

Across 21 record-specific main-model checkpoints, the 2×2 trained decoder is full rank numerically: every determinant is nonzero and every minimum singular value is positive. However, it is not well-conditioned. In normalized coordinates, the condition number ranges from 22.52 to 434.6; in original force units it ranges from 22.55 to 434.6, with 6 of 21 records above 100 and 1 above 200. Thus inversion is possible, but the relative sensitivity can be as high as κ₂; the worst-conditioned record is Pt_32_FD32_1 (κ₂=434.59, inverse gain along its compressed axis 3.86 latent units per force unit; force-space ellipse semiaxes 112.631, 0.259).

The physical-space mapping is

`q_raw = A_raw z + c`, `A_raw = diag(scale) @ decoder.weight`, `c = mean + scale * decoder.bias`.

The returned inverse map reconstructs all saved ensemble-mean force paths with maximum absolute round-trip error 2.84e-14 in record units.

## Shape and stability

For fixed coefficients and no noise, the latent radial equation is `dr/dt = r [ beta + (1-beta) r^2 - r^4 ]`. For `-1 < beta < 0`, it has a stable outer cycle `r=1` and an unstable inner cycle `r=sqrt(-beta)`. A full-rank affine decoder maps both circles to ellipses, centered at `c`. The singular values of `A_raw` give the ellipse semiaxes, and their ratio is the ellipse aspect ratio.

For the representative median-condition model `Pt_18_FD18_5`, the selected normal test origin is t=68.74 s with frozen local coefficients beta=-0.9483, f=1.1901 Hz, sigma=0.3884. The force-space outer-cycle semiaxes are 59.588 and 0.864 record units (aspect ratio 68.99); its center is (74.72, 79.18). The outer cycle's radial eigenvalue is -0.10333 s⁻¹ and its transverse Floquet multiplier per cycle is 0.91684, both indicating local attraction. These quantities are identical after transforming to force coordinates because `A_raw` gives a smooth invertible conjugacy. The ellipse changes Euclidean distances and the apparent direction-dependent contraction, but not asymptotic stability or Floquet multipliers. Across the saved normal test origins, the frozen-beta reference remains in `(-1,0)` for 8264 origins; the pooled beta range is -0.979 to -0.029. The pooled outer radial eigenvalue spans -1.943 to -0.043 s⁻¹, while all transverse Floquet multipliers remain below 1 (range 4.3e-09–0.959).

## Interpretation limit

The stability calculation freezes the coefficients at one origin and sets diffusion to zero. The actual model predicts time-varying beta/frequency/noise and simulates an SDE, so its realized forecasts are not closed deterministic limit cycles; stochastic paths can fluctuate or switch between the two radial basins. The solid line in the figure is the saved 128-path ensemble-mean forecast over five seconds. The dashed circles/ellipses are the corresponding local deterministic reference cycles. In an invertible affine map, isotropic latent diffusion becomes anisotropic/correlated in force coordinates with covariance proportional to `A_raw A_rawᵀ`.

This is evidence that the trained observation map retains both latent directions and allows the latent orbit to be visualized in the recorded two-channel space. It does not establish that the latent coordinates or learned vector field uniquely identify physiology.

## Files

- `decoder_invertibility_all21.csv`: determinant, singular values, condition numbers, effective raw-space matrices, local stability values.
- `latent_vs_force_limit_cycle.svg`: latent phase-plane reference cycles and corresponding decoded force-space ellipses, with one saved ensemble-mean forecast.
- `representative_trajectories.npz`: the affine map and representative mean paths.
- `summary.json`: machine-readable pooled audit and caveats.
