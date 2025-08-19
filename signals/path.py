# file: signals/path.py
"""
Deterministic synthetic price paths for driving exogenous "fair value" or news shocks.

We provide two generators:

1) OUPath: Ornstein–Uhlenbeck mean-reverting process around mu with Gaussian noise.
2) RegimePath: Two-state regime-switching drift/vol moves with optional jump shocks.

Both classes precompute a path of length `steps` with a fixed seed, so that
multiple agents can read `value_at(t)` without advancing shared state.

Usage
-----
p = OUPath(p0=100.0, mu=100.0, theta=0.05, sigma=0.5, steps=5000, seed=42)
ref = p.value_at(t)  # for any integer tick t in [0, steps)

Notes
-----
- These paths are **signals** (e.g., perceived fundamental value), not the traded price.
  You may add a "SignalTakerAgent" that nudges the market price toward this reference.
- Research simulator only. Not investment advice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class OUParams:
    mu: float = 100.0       # long-run mean
    theta: float = 0.05     # mean reversion speed
    sigma: float = 0.5      # diffusion vol per tick


class OUPath:
    def __init__(self, p0: float, params: OUParams, steps: int, seed: int = 42) -> None:
        self.p0 = float(p0)
        self.params = params
        self.steps = int(steps)
        self.seed = int(seed)
        self._path: List[float] = self._generate()

    def _generate(self) -> List[float]:
        rng = np.random.default_rng(self.seed)
        x = np.empty(self.steps, dtype=float)
        x[0] = self.p0
        mu, theta, sigma = self.params.mu, self.params.theta, self.params.sigma
        for t in range(1, self.steps):
            # Discrete-time OU: x_t+1 = x_t + theta*(mu - x_t) + sigma*eps
            eps = rng.normal(0.0, 1.0)
            x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * eps
        return x.tolist()

    def value_at(self, t: int) -> float:
        t = max(0, min(int(t), self.steps - 1))
        return self._path[t]


@dataclass
class RegimeParams:
    mu_low: float = 0.0
    mu_high: float = 0.1
    vol_low: float = 0.3
    vol_high: float = 1.2
    p_switch: float = 0.005   # probability to switch regime per tick
    jump_prob: float = 0.001  # probability of jump per tick
    jump_scale: float = 3.0   # jump magnitude (stds)


class RegimePath:
    """
    Additive regime-switching process on top of base price p0.

    price_{t+1} = price_t * (1 + drift_regime + vol_regime * N(0,1)) + jump
    """
    def __init__(self, p0: float, params: RegimeParams, steps: int, seed: int = 123) -> None:
        self.p0 = float(p0)
        self.params = params
        self.steps = int(steps)
        self.seed = int(seed)
        self._path: List[float] = self._generate()

    def _generate(self) -> List[float]:
        rng = np.random.default_rng(self.seed)
        p = np.empty(self.steps, dtype=float)
        p[0] = self.p0
        high = False
        for t in range(1, self.steps):
            if rng.random() < self.params.p_switch:
                high = not high
            mu = self.params.mu_high if high else self.params.mu_low
            vol = self.params.vol_high if high else self.params.vol_low
            ret = mu + vol * rng.normal(0.0, 1.0)
            jump = 0.0
            if rng.random() < self.params.jump_prob:
                jump = self.params.jump_scale * vol * rng.normal(0.0, 1.0)
            p[t] = max(1e-6, p[t - 1] * (1.0 + ret) + jump)
        return p.tolist()

    def value_at(self, t: int) -> float:
        t = max(0, min(int(t), self.steps - 1))
        return self._path[t]
