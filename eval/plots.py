# file: eval/plots.py
from __future__ import annotations
"""
Quick diagnostic plots for LLM Market Sim runs.

This module intentionally keeps plotting minimal and dependency-light (matplotlib only).
All functions save PNGs into the provided output directory.

Functions
---------
quick_diagnostics(trades_df, accounts_df, snaps_df, outdir)
  - mid price over time
  - spread over time
  - average bid/ask depth over time
  - equity curves per agent (thinned to avoid huge PNGs)
  - inventory histogram per agent (faceted vertically if many)

Notes
-----
These plots are for eyeballing runs. For paper-ready figures, you'll likely export
aggregated CSVs and style them in your publication pipeline.
"""
import math
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(d: Path | str) -> Path:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mid(snaps_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if snaps_df.empty or "mid" not in snaps_df:
        return None
    p = outdir / "price_mid.png"
    plt.figure(figsize=(9, 3))
    snaps_df = snaps_df.sort_values("time")
    plt.plot(snaps_df["time"], snaps_df["mid"])
    plt.title("Mid Price")
    plt.xlabel("time")
    plt.ylabel("mid")
    _savefig(p)
    return p


def plot_spread(snaps_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if snaps_df.empty or "spread" not in snaps_df:
        return None
    p = outdir / "spread.png"
    plt.figure(figsize=(9, 3))
    snaps_df = snaps_df.sort_values("time")
    plt.plot(snaps_df["time"], snaps_df["spread"])
    plt.title("Bid-Ask Spread")
    plt.xlabel("time")
    plt.ylabel("spread")
    _savefig(p)
    return p


def plot_depth(snaps_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if snaps_df.empty or "bid_depth" not in snaps_df or "ask_depth" not in snaps_df:
        return None
    p = outdir / "depth.png"
    plt.figure(figsize=(9, 3))
    snaps_df = snaps_df.sort_values("time")
    plt.plot(snaps_df["time"], snaps_df["bid_depth"], label="bid_depth")
    plt.plot(snaps_df["time"], snaps_df["ask_depth"], label="ask_depth")
    plt.title("Top-of-Book Depth")
    plt.xlabel("time")
    plt.ylabel("units")
    plt.legend()
    _savefig(p)
    return p


def plot_equity_curves(accounts_df: pd.DataFrame, outdir: Path, max_agents: int = 30) -> Optional[Path]:
    if accounts_df.empty:
        return None
    p = outdir / "equity_curves.png"
    plt.figure(figsize=(10, 6))
    acc = accounts_df.copy()
    agents = acc["agent"].unique().tolist()
    # If too many agents, sample a subset for readability
    if len(agents) > max_agents:
        agents = agents[:max_agents]
        acc = acc[acc["agent"].isin(agents)]
    for aid, g in acc.groupby("agent"):
        g = g.sort_values("time")
        plt.plot(g["time"], g["equity"], alpha=0.8, label=str(aid))
    plt.title("Equity Curves (subset if many agents)")
    plt.xlabel("time")
    plt.ylabel("equity")
    if len(agents) <= 15:
        plt.legend(ncol=2)
    _savefig(p)
    return p


def plot_inventory_hist(accounts_df: pd.DataFrame, outdir: Path, per_fig: int = 12) -> list[Path]:
    paths: list[Path] = []
    if accounts_df.empty:
        return paths
    acc = accounts_df.copy()
    agents = sorted(acc["agent"].unique().tolist())
    # Aggregate last inventory per agent
    last_inv = acc.sort_values("time").groupby("agent")["inventory"].last().reset_index()

    # Facet into multiple panels if many agents
    n = len(agents)
    pages = math.ceil(n / per_fig)
    for pg in range(pages):
        chunk = last_inv.iloc[pg * per_fig : (pg + 1) * per_fig]
        if chunk.empty:
            continue
        rows = math.ceil(len(chunk) / 3)
        cols = min(3, len(chunk))
        plt.figure(figsize=(cols * 3.2, rows * 2.6))
        for i, (_, row) in enumerate(chunk.iterrows(), start=1):
            plt.subplot(rows, cols, i)
            # Use per-agent time-series histogram if you prefer; here we show last inventory as a stem.
            plt.stem([0], [row["inventory"]], use_line_collection=True)
            plt.title(str(row["agent"]))
            plt.xticks([])
            plt.ylabel("inventory")
        p = outdir / f"inventory_page{pg+1}.png"
        _savefig(p)
        paths.append(p)
    return paths


def quick_diagnostics(
    trades_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    snaps_df: pd.DataFrame,
    outdir: Path | str,
) -> dict[str, list[str] | str]:
    """
    Produce a minimal set of PNGs for quick inspection.

    Returns
    -------
    dict: mapping figure name -> path(s)
    """
    outp = _ensure_dir(outdir)
    results: dict[str, list[str] | str] = {}

    mp = plot_mid(snaps_df, outp)
    if mp:
        results["price_mid"] = str(mp)

    sp = plot_spread(snaps_df, outp)
    if sp:
        results["spread"] = str(sp)

    dp = plot_depth(snaps_df, outp)
    if dp:
        results["depth"] = str(dp)

    eq = plot_equity_curves(accounts_df, outp)
    if eq:
        results["equity_curves"] = str(eq)

    inv_paths = plot_inventory_hist(accounts_df, outp)
    if inv_paths:
        results["inventory_pages"] = [str(p) for p in inv_paths]

    return results
