# file: tests/test_simulator_smoke.py
from __future__ import annotations

from orchestration import Simulator
from benchmarks import build_agents


def test_simulator_runs_small():
    agents = build_agents(scenario="small", seed=123, llm_interval=10)
    sim = Simulator(agents=agents, initial_price=100.0, seed=123, fee_bps=1.0, snapshot_interval=5)
    sim.run(steps=300, seed=123)
    trades_df, accounts_df, snaps_df = sim.to_dataframes()

    # Basic sanity checks
    assert not trades_df.empty
    assert not accounts_df.empty
    assert not snaps_df.empty

    # All agent ids present in timeline
    agent_ids = set(a.agent_id for a in agents)
    assert agent_ids.issubset(set(accounts_df["agent"].unique()))
