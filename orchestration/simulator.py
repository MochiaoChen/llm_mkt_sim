# file: orchestration/simulator.py
"""
Event-driven simulator that ties Agents to the OrderBook and keeps accounts.

Responsibilities
----------------
- Maintain a single-asset limit order book.
- Query each agent every tick for an OrderRequest (or None).
- Submit orders, record trades, update accounts with fees and PnL.
- Provide observation bundle (top-of-book snapshot + agent account + signals).
- Periodically record book snapshots and account timeline for evaluation.

Design notes
------------
- Single asset, single venue for MVP.
- Deterministic pseudo-time: integer ticks (0..steps-1).
- Simple fee model: both sides pay fee_bps of notional.
- Bootstraps initial liquidity around `initial_price` to get a usable mid.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import math
import random
import os

import numpy as np

from exchange.order_book import (
    OrderBook,
    Order,
    Trade,
    Side,
    OrderType,
    TimeInForce,
)
from agents.base import Agent, AccountState, Observation, OrderRequest


class IdGen:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.seq = 0

    def next(self) -> str:
        self.seq += 1
        return f"{self.prefix}{self.seq}"


class Simulator:
    """
    Orchestrates agents, order book, and accounting.

    Public attributes (filled after run):
        trades_log: List[dict]           # trade timeline
        accounts_timeline: List[dict]    # per-tick per-agent equity/inventory
        snapshots: List[dict]            # coarse book snapshots (top-of-book)
    """

    def __init__(
        self,
        agents: List[Agent],
        initial_price: float = 100.0,
        seed: int = 42,
        fee_bps: float = 1.0,  # 1 bp = 0.01% → this is 1.0 bps = 0.01%
        tick_size: Optional[float] = None,
        snapshot_interval: int = 10,
        bootstrap_qty: float = 50.0,
        bootstrap_spread: float = 0.01,  # ±1%
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.rng = random.Random(seed)

        self.book = OrderBook(tick_size=tick_size)
        self.agents: List[Agent] = agents
        self.agent_ids = [a.agent_id for a in agents]
        self.accounts: Dict[str, AccountState] = {
            a.agent_id: AccountState(agent_id=a.agent_id, cash=0.0, inventory=0.0) for a in agents
        }
        self.fee_rate = float(fee_bps) * 1e-4
        self.snapshot_interval = snapshot_interval

        self.order_id_gen = IdGen("O")
        self.initial_price = float(initial_price)
        self.bootstrap_qty = float(bootstrap_qty)
        self.bootstrap_spread = float(bootstrap_spread)

        # logs
        self.trades_log: List[dict] = []
        self.accounts_timeline: List[dict] = []
        self.snapshots: List[dict] = []

    # -------------------------
    # Bootstrapping liquidity
    # -------------------------
    def _bootstrap_book(self, now: float) -> None:
        mid = self.initial_price
        bid_px = mid * (1.0 - self.bootstrap_spread)
        ask_px = mid * (1.0 + self.bootstrap_spread)
        # System orders with agent_id="SYS"
        bid = Order(
            order_id=self.order_id_gen.next(),
            agent_id="SYS",
            side=Side.BUY,
            type=OrderType.LIMIT,
            qty=self.bootstrap_qty,
            price=bid_px,
            timestamp=now,
            tif=TimeInForce.GTC,
        )
        ask = Order(
            order_id=self.order_id_gen.next(),
            agent_id="SYS",
            side=Side.SELL,
            type=OrderType.LIMIT,
            qty=self.bootstrap_qty,
            price=ask_px,
            timestamp=now,
            tif=TimeInForce.GTC,
        )
        self.book.submit(bid)
        self.book.submit(ask)
        # last trade anchor for marking
        self.book.last_trade_price = mid

    # -------------------------
    # Accounting helpers
    # -------------------------
    def _apply_fill(self, trade: Trade, buy_agent: str, sell_agent: str) -> None:
        px, qty = trade.price, trade.qty
        notional = px * qty
        fee = notional * self.fee_rate

        # Buyer pays cash and increases inventory
        if buy_agent in self.accounts:
            acct_b = self.accounts[buy_agent]
            acct_b.cash -= notional + fee
            acct_b.inventory += qty
            acct_b.fees_paid += fee

        # Seller receives cash and decreases inventory
        if sell_agent in self.accounts:
            acct_s = self.accounts[sell_agent]
            acct_s.cash += notional - fee
            acct_s.inventory -= qty
            acct_s.fees_paid += fee

    def _mark_to_mid(self, mid: Optional[float]) -> None:
        if mid is None:
            return
        for acct in self.accounts.values():
            acct.last_mark_price = mid

    # -------------------------
    # Observation builder
    # -------------------------
    def _obs_for(self, agent_id: str, now: float) -> Observation:
        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()
        spread = None
        mid = None
        if best_bid is not None and best_ask is not None:
            spread = max(0.0, best_ask - best_bid)
            mid = (best_bid + best_ask) / 2.0
        elif self.book.last_trade_price is not None:
            mid = self.book.last_trade_price

        # depth: top-of-book quantities (coarse)
        bid_depth = 0.0
        ask_depth = 0.0
        snap = self.book.snapshot(n_levels=1)
        if snap["bids"]:
            bid_depth = snap["bids"][0][1]
        if snap["asks"]:
            ask_depth = snap["asks"][0][1]

        return Observation(
            time=now,
            mid=mid,
            spread=spread,
            best_bid=best_bid,
            best_ask=best_ask,
            last_trade=self.book.last_trade_price,
            account=self.accounts[agent_id],
            signals={},  # hook for external signals/news
            bid_depth=bid_depth,
            ask_depth=ask_depth,
        )

    # -------------------------
    # Core step: handle orders
    # -------------------------
    def _submit_from_request(self, agent_id: str, req: OrderRequest, now: float) -> Tuple[List[Trade], List[str], str]:
        order = Order(
            order_id=self.order_id_gen.next(),
            agent_id=agent_id,
            side=req.side,
            type=req.type,
            qty=req.qty,
            price=req.price,
            timestamp=now,
            tif=req.tif,
        )
        trades, acks = self.book.submit(order)
        return trades, acks, order.order_id

    # -------------------------
    # Run loop
    # -------------------------
    def run(self, steps: int, seed: Optional[int] = None) -> None:
        """Run the simulation for a number of ticks (0..steps-1)."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # agents reset (optional)
        for a in self.agents:
            a.reset(seed)

        # Bootstrap book so agents have a mid
        self._bootstrap_book(now=0.0)

        for t in range(steps):
            now = float(t)

            # top-of-book to mark accounts
            snap0 = self.book.snapshot(n_levels=1)
            mid0 = snap0.get("mid")
            self._mark_to_mid(mid0)

            # Query each agent
            for agent in self.agents:
                obs = self._obs_for(agent.agent_id, now)
                out = agent.on_observation(obs)
                if out is None:
                    continue
                # Submit to book
                trades, acks, oid = self._submit_from_request(agent.agent_id, out, now)

                # Process trades & accounting
                for tr in trades:
                    # Need to decode buy/sell agent ids: they are embedded in order ids via book index
                    # The OrderBook gives us only order ids; we can't map to agent here without index.
                    # For MVP we log agents using a local map: derive via last active orders? Instead,
                    # we carry agent id in Order; OrderBook doesn't expose it back in Trade.
                    # To keep code minimal, we *encode* agent id into order_id prefix: O###-<agent_id>
                    pass  # replaced below

            # The above `pass` is a placeholder; we need a way to recover agent ids for fills.
            # Workaround: extend OrderBook to include original Order objects in a local lookup.
            # Instead of modifying OrderBook API, we keep a local map from order_id -> agent_id
            # within this Simulator step. Implemented below by re-running loop with tracking.

        # ------------- Re-run properly with fill tracking -------------
        # The simplistic loop above highlighted the need for order_id -> agent_id mapping
        # for accounting. Implement the full run loop with a persistent mapping.

        # Reset logs & accounts and run again properly
        self.trades_log.clear()
        self.accounts_timeline.clear()
        self.snapshots.clear()
        # Reset book and accounts
        self.book = OrderBook(tick_size=self.book.tick_size)
        for k in list(self.accounts.keys()):
            self.accounts[k] = AccountState(agent_id=k, cash=0.0, inventory=0.0, realized_pnl=0.0, fees_paid=0.0)

        # order->agent mapping for accounting
        order_owner: Dict[str, str] = {}

        # Bootstrap again
        self._bootstrap_book(now=0.0)

        for a in self.agents:
            a.reset(seed)

        for t in range(steps):
            now = float(t)

            # mark to mid
            snap = self.book.snapshot(n_levels=1)
            self._mark_to_mid(snap.get("mid"))

            # agent actions
            for agent in self.agents:
                obs = self._obs_for(agent.agent_id, now)
                req = agent.on_observation(obs)
                if req is None:
                    continue
                # submit and track owner
                ord_id = f"{self.order_id_gen.next()}"
                order = Order(
                    order_id=ord_id,
                    agent_id=agent.agent_id,
                    side=req.side,
                    type=req.type,
                    qty=req.qty,
                    price=req.price,
                    timestamp=now,
                    tif=req.tif,
                )
                order_owner[ord_id] = agent.agent_id
                trades, acks = self.book.submit(order)

                # apply fills
                for tr in trades:
                    # The OrderBook returns buy_order_id / sell_order_id, we map them to agent ids.
                    buyer = order_owner.get(tr.buy_order_id, "SYS")
                    seller = order_owner.get(tr.sell_order_id, "SYS")
                    self._apply_fill(tr, buyer, seller)
                    self.book.last_trade_price = tr.price  # update last trade

                    # log trade
                    self.trades_log.append(
                        {
                            "time": now,
                            "price": tr.price,
                            "qty": tr.qty,
                            "buy_agent": buyer,
                            "sell_agent": seller,
                            "trade_id": tr.trade_id,
                        }
                    )

            # periodic snapshot
            if (t % self.snapshot_interval) == 0:
                s = self.book.snapshot(n_levels=5)
                self.snapshots.append(
                    {
                        "time": now,
                        "mid": s.get("mid"),
                        "spread": s.get("spread"),
                        "best_bid": s["bids"][0][0] if s["bids"] else None,
                        "best_ask": s["asks"][0][0] if s["asks"] else None,
                        "bid_depth": s["bids"][0][1] if s["bids"] else 0.0,
                        "ask_depth": s["asks"][0][1] if s["asks"] else 0.0,
                    }
                )

            # record accounts timeline (per agent)
            for aid, acct in self.accounts.items():
                self.accounts_timeline.append(
                    {
                        "time": now,
                        "agent": aid,
                        "cash": acct.cash,
                        "inventory": acct.inventory,
                        "equity": acct.equity(),
                        "fees_paid": acct.fees_paid,
                    }
                )

    # -------------------------
    # Export helpers
    # -------------------------
    def to_dataframes(self):
        """Return (trades_df, accounts_df, snaps_df) as pandas DataFrames."""
        import pandas as pd

        trades_df = pd.DataFrame(self.trades_log)
        accounts_df = pd.DataFrame(self.accounts_timeline)
        snaps_df = pd.DataFrame(self.snapshots)
        return trades_df, accounts_df, snaps_df
