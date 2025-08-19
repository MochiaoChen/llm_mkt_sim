# file: benchmarks/scenarios.py
from __future__ import annotations
"""
Scenario factory for assembling agent rosters with reproducible IDs.

Usage
-----
from benchmarks.scenarios import build_agents
agents = build_agents(scenario="small", seed=42, llm_interval=20)

Design
------
- Deterministic agent IDs per run so logs are easy to analyze.
- Override knobs (n_llm, n_mm, n_mom, n_noise) to customize mixes.
- Sensible parameter defaults for each agent family.

Scenarios
---------
small:
  - 2 LLM decision agents
  - 5 MarketMakers
  - 5 Momentum
  - 10 Noise

medium:
  - 6 LLM decision agents
  - 20 MarketMakers
  - 20 Momentum
  - 40 Noise

custom:
  - start from "small" and apply overrides; or set exact counts via overrides.
"""
import random
from typing import Dict, List, Optional

from agents.base import Agent
from agents.baseline import MarketMakerAgent, MomentumAgent, NoiseAgent
from agents.llm_agent import LLMDecisionAgent


def _seq_names(prefix: str, n: int) -> List[str]:
    return [f"{prefix}{i+1:03d}" for i in range(n)]


def _mk_llm_agents(n: int, interval: int, seed: int) -> List[Agent]:
    # LLMDecisionAgent is low-frequency; interval controls refresh cadence.
    names = _seq_names("LLM_", n)
    return [LLMDecisionAgent(agent_id=name, interval=interval) for name in names]


def _mk_mm_agents(n: int, seed: int) -> List[Agent]:
    # Slight heterogeneity in base_size/spread driven by seed
    rng = random.Random(seed + 101)
    names = _seq_names("MM_", n)
    agents: List[Agent] = []
    for i, name in enumerate(names):
        base_size = 0.8 + 0.4 * rng.random()     # ~[0.8, 1.2]
        base_spread = 0.0008 + 0.0008 * rng.random()  # ~[8,16] bps
        agents.append(MarketMakerAgent(agent_id=name, base_size=base_size, base_spread=base_spread))
    return agents


def _mk_mom_agents(n: int, seed: int) -> List[Agent]:
    rng = random.Random(seed + 202)
    names = _seq_names("MOM_", n)
    agents: List[Agent] = []
    for i, name in enumerate(names):
        lookback = rng.choice([3, 5, 8, 13])
        size = rng.choice([0.5, 1.0, 1.5])
        agents.append(MomentumAgent(agent_id=name, lookback=lookback, size=size))
    return agents


def _mk_noise_agents(n: int, seed: int) -> List[Agent]:
    rng = random.Random(seed + 303)
    names = _seq_names("NOI_", n)
    agents: List[Agent] = []
    for i, name in enumerate(names):
        size = rng.choice([0.2, 0.3, 0.5])
        prob = rng.uniform(0.03, 0.12)  # submit on ~3%-12% of ticks
        agents.append(NoiseAgent(agent_id=name, size=size, prob=prob))
    return agents


def build_agents(
    scenario: str = "small",
    seed: int = 42,
    llm_interval: int = 20,
    overrides: Optional[Dict[str, Optional[int]]] = None,
) -> List[Agent]:
    """
    Build a list of agents for a given scenario.

    Parameters
    ----------
    scenario : {"small","medium","custom"}
        Preset roster size. "custom" starts from "small" and then applies overrides.
    seed : int
        Controls heterogeneity and reproducibility.
    llm_interval : int
        Decision refresh cadence for LLM agents (in ticks).
    overrides : dict
        Keys: n_llm, n_mm, n_mom, n_noise. Any None values are ignored.

    Returns
    -------
    List[Agent]
    """
    overrides = overrides or {}

    if scenario not in {"small", "medium", "custom"}:
        raise ValueError("scenario must be one of: small | medium | custom")

    # Base sizes
    if scenario == "small" or scenario == "custom":
        n_llm, n_mm, n_mom, n_noise = 2, 5, 5, 10
    elif scenario == "medium":
        n_llm, n_mm, n_mom, n_noise = 6, 20, 20, 40

    # Apply overrides if provided
    for k in ("n_llm", "n_mm", "n_mom", "n_noise"):
        v = overrides.get(k)
        if isinstance(v, int) and v >= 0:
            if k == "n_llm":
                n_llm = v
            elif k == "n_mm":
                n_mm = v
            elif k == "n_mom":
                n_mom = v
            elif k == "n_noise":
                n_noise = v

    roster: List[Agent] = []
    roster.extend(_mk_llm_agents(n_llm, interval=llm_interval, seed=seed))
    roster.extend(_mk_mm_agents(n_mm, seed=seed))
    roster.extend(_mk_mom_agents(n_mom, seed=seed))
    roster.extend(_mk_noise_agents(n_noise, seed=seed))

    return roster
