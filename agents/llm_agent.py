# file: agents/llm_agent.py
"""
LLM-driven decision agent.

This agent queries an LLM (real or heuristic fallback) at a lower frequency
(e.g., once every N ticks) to produce a high-level trading Intent. The
Intent is then translated into concrete orders by an ExecutionPolicy.

Design
------
- LLMDecisionAgent: manages when to call LLM, stores current Intent.
- ExecutionPolicy: maps Intent into OrderRequest(s) over multiple ticks.
- FallbackHeuristic: deterministic "toy LLM" used if no API key available.

Notes
-----
This is an MVP; the prompt schema is kept minimal. In practice you may
inject richer state, role personas, or external tools (news, factors).
"""
from __future__ import annotations

import random
import os
from typing import Optional

from agents.base import Agent, Observation, Intent, OrderRequest
from exchange.order_book import Side, OrderType, TimeInForce


# ================
# Execution Policy
# ================

class ExecutionPolicy:
    """
    Simple TWAP-like executor for Intents.

    - Each tick, compute remaining target delta vs. current inventory.
    - Spread the delta evenly over remaining horizon.
    - Place aggressive or passive orders depending on urgency.
    """

    def __init__(self) -> None:
        self.active_intent: Optional[Intent] = None
        self.ticks_left: int = 0

    def set_intent(self, intent: Intent) -> None:
        self.active_intent = intent
        self.ticks_left = intent.time_horizon

    def step(self, obs: Observation) -> Optional[OrderRequest]:
        if self.active_intent is None or self.ticks_left <= 0:
            return None
        inv = obs.account.inventory
        target = self.active_intent.target_position
        delta = target - inv
        if abs(delta) < 1e-6:
            return None
        # qty per tick
        slice_qty = delta / self.ticks_left
        side = Side.BUY if slice_qty > 0 else Side.SELL
        qty = abs(slice_qty)
        # price logic
        if obs.mid is None:
            return None
        if self.active_intent.urgency > 0.7:
            # cross spread (market order)
            order = OrderRequest(side=side, type=OrderType.MARKET, qty=qty)
        else:
            # passive limit order near best
            px = obs.mid * (1 - 0.001 if side == Side.BUY else 1 + 0.001)
            order = OrderRequest(side=side, type=OrderType.LIMIT, qty=qty, price=px, tif=TimeInForce.GTC)
        self.ticks_left -= 1
        return order


# ====================
# Fallback Heuristic LLM
# ====================

class FallbackHeuristic:
    """
    Cheap deterministic pseudo-LLM.

    Logic:
    - Bias to mean revert: if inventory too positive, suggest selling down.
    - Else random long/short toggle with small target.
    """

    def generate_intent(self, obs: Observation) -> Intent:
        inv = obs.account.inventory
        if inv > 2:
            return Intent(target_position=0.0, urgency=0.8, note="Reduce long exposure")
        elif inv < -2:
            return Intent(target_position=0.0, urgency=0.8, note="Reduce short exposure")
        else:
            # random tilt
            tgt = random.choice([-1.0, 0.0, 1.0])
            return Intent(target_position=tgt, urgency=0.5, note="Heuristic tilt")


# ====================
# LLMDecisionAgent
# ====================

class LLMDecisionAgent(Agent):
    """
    LLM-based decision agent.

    - Every `interval` ticks, calls an LLM (real or fallback) to get an Intent.
    - Stores Intent and uses ExecutionPolicy to gradually place orders.
    """

    def __init__(self, agent_id: str, interval: int = 20) -> None:
        super().__init__(agent_id)
        self.interval = interval
        self.executor = ExecutionPolicy()
        # In MVP, always use fallback. (Extend to real LLM API with API key detection)
        self.llm = FallbackHeuristic()
        self._last_intent_tick: int = -999

    def on_observation(self, obs: Observation) -> Optional[OrderRequest]:
        # check if it's time to refresh intent
        if int(obs.time) - self._last_intent_tick >= self.interval:
            intent = self.llm.generate_intent(obs)
            self.executor.set_intent(intent)
            self._last_intent_tick = int(obs.time)
        return self.executor.step(obs)
