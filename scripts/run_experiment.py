# file: scripts/run_experiment.py
from __future__ import annotations

"""
CLI to run LLM Market Sim experiments.

Examples
--------
# Small, fast dry run (no real LLM needed)
python -m scripts.run_experiment --scenario small --steps 2000 --seed 42

# Medium with LLM intent every 20 ticks and plots
python -m scripts.run_experiment --scenario medium --steps 5000 --llm-interval 20 --plots

Outputs
-------
<outdir>/
  events.parquet        # trade timeline
  accounts.parquet      # per-tick per-agent equity/inventory
  snapshots.parquet     # coarse book snapshots (top-of-book)
  config.json           # the run configuration
  plots/                # optional diagnostic pngs
"""

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import print as rprint
from rich.table import Table

from orchestration.simulator import Simulator
# Provided next in benchmarks/scenarios.py
from benchmarks.scenarios import build_agents  # type: ignore


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _mk_outdir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    (p / "plots").mkdir(exist_ok=True)
    return p


def _preview(df: pd.DataFrame, name: str, n: int = 5) -> None:
    tbl = Table(title=f"{name} (head {n})")
    if df.empty:
        rprint(f"[yellow]{name} is empty[/yellow]")
        return
    for c in df.columns[:10]:
        tbl.add_column(str(c))
    for _, row in df.head(n).iterrows():
        tbl.add_row(*[str(row[c]) for c in df.columns[:10]])
    rprint(tbl)


@app.command()
def main(
    scenario: str = typer.Option("small", help="Preset scenario: small|medium|custom"),
    steps: int = typer.Option(2000, help="Number of simulator ticks."),
    seed: int = typer.Option(42, help="Random seed."),
    llm_interval: int = typer.Option(20, help="LLM decision refresh interval (ticks)."),
    initial_price: float = typer.Option(100.0, help="Initial mid price for bootstrapping."),
    fee_bps: float = typer.Option(1.0, help="Fee in basis points (0.01% = 1 bp)."),
    tick_size: Optional[float] = typer.Option(None, help="Optional price tick size."),
    snapshot_interval: int = typer.Option(10, help="Record book snapshot every N ticks."),
    outdir: str = typer.Option("outputs/run", help="Output directory."),
    plots: bool = typer.Option(False, "--plots/--no-plots", help="Whether to render quick plots."),
    n_llm: Optional[int] = typer.Option(None, help="Override number of LLM agents."),
    n_mm: Optional[int] = typer.Option(None, help="Override number of MarketMaker agents."),
    n_mom: Optional[int] = typer.Option(None, help="Override number of Momentum agents."),
    n_noise: Optional[int] = typer.Option(None, help="Override number of Noise agents."),
):
    """
    Run an experiment and write Parquet/CSV + optional diagnostic plots.
    """
    outp = _mk_outdir(outdir)

    # Build agents from scenario factory (next file will define this)
    agents = build_agents(
        scenario=scenario,
        seed=seed,
        llm_interval=llm_interval,
        overrides=dict(n_llm=n_llm, n_mm=n_mm, n_mom=n_mom, n_noise=n_noise),
    )
    rprint(f"[green]Built {len(agents)} agents[/green] for scenario: [bold]{scenario}[/bold]")

    # Simulator
    sim = Simulator(
        agents=agents,
        initial_price=initial_price,
        seed=seed,
        fee_bps=fee_bps,
        tick_size=tick_size,
        snapshot_interval=snapshot_interval,
    )
    rprint(f"[cyan]Running {steps} ticks...[/cyan]")
    sim.run(steps=steps, seed=seed)

    # Export
    trades_df, accounts_df, snaps_df = sim.to_dataframes()
    trades_path = outp / "events.parquet"
    accounts_path = outp / "accounts.parquet"
    snaps_path = outp / "snapshots.parquet"

    trades_df.to_parquet(trades_path, index=False)
    accounts_df.to_parquet(accounts_path, index=False)
    snaps_df.to_parquet(snaps_path, index=False)

    # Also small CSV heads for quick peeking
    trades_df.head(1000).to_csv(outp / "events_head.csv", index=False)
    accounts_df.head(2000).to_csv(outp / "accounts_head.csv", index=False)
    snaps_df.head(1000).to_csv(outp / "snapshots_head.csv", index=False)

    # Save run config
    cfg = dict(
        scenario=scenario,
        steps=steps,
        seed=seed,
        llm_interval=llm_interval,
        initial_price=initial_price,
        fee_bps=fee_bps,
        tick_size=tick_size,
        snapshot_interval=snapshot_interval,
        overrides=dict(n_llm=n_llm, n_mm=n_mm, n_mom=n_mom, n_noise=n_noise),
        outdir=str(outp),
    )
    with open(outp / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Console previews
    _preview(trades_df, "Trades")
    _preview(accounts_df, "Accounts")
    _preview(snaps_df, "Snapshots")

    # Optional plots
    if plots:
        try:
            from eval.metrics import compute_basic_metrics  # type: ignore
            from eval.plots import quick_diagnostics  # type: ignore

            metrics = compute_basic_metrics(trades_df, accounts_df, snaps_df)
            rprint("[magenta]Metrics (basic):[/magenta]")
            for k, v in metrics.items():
                rprint(f"  {k}: {v}")

            quick_diagnostics(trades_df, accounts_df, snaps_df, outdir=outp / "plots")
            rprint(f"[green]Plots written to {outp / 'plots'}[/green]")
        except Exception as e:
            rprint(f"[yellow]Plot/metrics generation failed: {e}[/yellow]")

    rprint(f"[bold green]Done. Outputs in {outp}[/bold green]")


if __name__ == "__main__":
    app()
