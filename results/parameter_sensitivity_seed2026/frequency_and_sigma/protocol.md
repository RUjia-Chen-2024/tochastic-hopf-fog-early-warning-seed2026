# Paired frequency and noise-strength replay

- 21 records; 8,264 normal test origins; 91 test events.
- Each forecast uses one cycle of input history, a 5-s stochastic Hopf rollout, 128 paths, and paired Brownian increments.
- The alarm score is the chance of a new cycle anomaly within 2 s. Alarm occurs after the score is above 0.80 at two consecutive 0.05-s origins.
- Frequency intervention: hold each origin's `frequency(0)` across that origin's 5-s rollout; retain beta(s), sigma(s), initial state, decoder, and the same Brownian increments.
- Sigma intervention: hold each origin's `sigma(0)` across that origin's 5-s rollout; retain beta(s), frequency(s), initial state, and decoder. The same standard Brownian increments `dW` are multiplied by sigma(s) in the baseline and sigma(0) in the counterfactual, so this freezes noise intensity without turning noise off.
- For each path, first future accepted peak uses the same peak calibration, threshold, prominence, minimum distance, and cycle-age filter as the event-label code.
- Baseline risks reproduced the existing seed-2026 evaluation exactly (risk MAE 0 for every record). This is a post-fit sensitivity analysis, not a refit, parameter-identification test, or causal/physiological claim.
