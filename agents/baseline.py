# file: agents/baseline.py
"""
Baseline heuristic agents for comparison against LLM-driven agents.

Implemented strategies:
- MarketMakerAgent: posts bids/asks around mid, rebalances inventory.
- MomentumAgent: trades in direction of recent price trend.
- NoiseAgent: submits random buy/sell orders with small size.

These are intentionally simple but configurable so we can benchmark
market quality with and without LLM agents.
"""
from __future__ import annotations

import random
import math
from typing import Optional

from agents.base import Agent, Observation, OrderRequest
from exchange.order_book import Side, OrderType, TimeInForce


class MarketMakerAgent(Agent):
    """
    Simple inventory-based market maker.

    Logic:
    - Posts one bid and one ask around mid price.
    - Spread scales with inventory imbalance.
    - Order size = base_size (configurable).
    """

    def __init__(self, agent_id: str, base_size: float = 1.0, base_spread: float = 0.001) -> None:
        super().__init__(agent_id)
        self.base_size = base_size
        self.base_spread = base_spread

    def on_observation(self, obs: Observation) -> Optional[OrderRequest]:
        if obs.mid is None:
            return None
        inv = obs.account.inventory
        # widen spread if inventory heavy
        spread_adj = self.base_spread * (1 + abs(inv))
        # choose side to lean: if inv >0, lean sell; if inv <0, lean buy
        if inv > 0:
            side = Side.SELL
            price = obs.mid * (1 + spread_adj)
        else:
            side = Side.BUY
            price = obs.mid * (1 - spread_adj)
        return OrderRequest(side=side, type=OrderType.LIMIT, qty=self.base_size, price=price, tif=TimeInForce.GTC)


class MomentumAgent(Agent):
    """
    Trades with short-term momentum.

    Logic:
    - Compare last_trade vs mid (or vs previous mid).
    - If price rising, go long; if falling, go short.
    """

    def __init__(self, agent_id: str, lookback: int = 5, size: float = 1.0) -> None:
        super().__init__(agent_id)
        self.lookback = lookback
        self.size = size
        self.history: list[float] = []

    def on_observation(self, obs: Observation) -> Optional[OrderRequest]:
        if obs.mid is None:
            return None
        self.history.append(obs.mid)
        if len(self.history) < self.lookback + 1:
            return None
        prev = self.history[-self.lookback - 1]
        if obs.mid > prev:
            return OrderRequest(side=Side.BUY, type=OrderType.MARKET, qty=self.size)
        elif obs.mid < prev:
            return OrderRequest(side=Side.SELL, type=OrderType.MARKET, qty=self.size)
        return None


class NoiseAgent(Agent):
    """
    Random buy/sell with small size.

    Useful as a background "retail noise flow".
    """

    def __init__(self, agent_id: str, size: float = 0.5, prob: float = 0.1) -> None:
        super().__init__(agent_id)
        self.size = size
        self.prob = prob

    def on_observation(self, obs: Observation) -> Optional[OrderRequest]:
        if random.random() < self.prob:
            side = random.choice([Side.BUY, Side.SELL])
            return OrderRequest(side=side, type=OrderType.MARKET, qty=self.size)
        return None
