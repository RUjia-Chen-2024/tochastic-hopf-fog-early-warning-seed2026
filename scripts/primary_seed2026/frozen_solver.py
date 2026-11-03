"""Exact solver and IO functions used in the archived seed-2026 execution."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch

def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, data):
    if not data:
        return
    fields = list(dict.fromkeys(k for r in data for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(data)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=True), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def simulate(model, inputs, draws, multiplier):
    # Same operation order as the frozen source forward; only sigma is scaled.
    h = model.encoder(inputs.flatten(1))
    z0 = 1.5 * torch.tanh(model.initial(h))
    p0 = model.param_head(h)
    branch = model.branch(h).reshape(len(inputs), 3, 32)
    trunk = model.trunk(model.tau).reshape(model.horizon, 3, 32)
    delta = torch.einsum('bcp,tcp->btc', branch, trunk) / np.sqrt(32)
    curve = model.physical_parameters(p0[:, None, :] + model.tau[None, :, :] * delta)
    z = z0[:, None, :].expand(-1, draws, -1)
    substeps = max(1, int(np.ceil(model.dt / .01)))
    ds = model.dt / substeps
    fine = []
    for k in range(model.horizon):
        beta = curve[:, k, 0, None, None]
        freq = curve[:, k, 1, None, None]
        sigma = curve[:, k, 2, None, None] * multiplier
        for _ in range(substeps):
            drift = model.drift(z, beta, freq)
            noise = torch.randn_like(z) * np.sqrt(ds) * sigma
            z = z + ds * drift / (1 + ds * torch.linalg.vector_norm(drift, dim=-1, keepdim=True)) + noise
            fine.append(z)
    latent = torch.stack(fine, 2)
    decoded = model.decoder(latent)
    return decoded, decoded[:, :, substeps-1::substeps], curve, ds

