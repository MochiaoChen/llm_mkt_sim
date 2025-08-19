# file: agents/base.py
"""
Agent interfaces and shared data structures.

This layer defines:
- Intent schema that LLM (or heuristic) agents produce at low frequency.
- Abstract Agent base class with hooks used by the simulator.
- Simple portfolio/account state for PnL & risk tracking.

Design
------
Agents receive:
  - current time (tick)
  - market observations (book snapshot, last trade, own inventory, cash)
  - optional external signals (e.g., news/factors)

They may emit:
  - low-level orders (limit/market) to the exchange, or
  - a high-level Intent that execution policies translate into orders.

For MVP we keep both options:
  - Baseline agents (MM/Momentum/Noise) place orders directly.
  - LLMDecisionAgent outputs Intent; an ExecutionPolicy maps to orders.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator

from exchange.order_book import Side, OrderType, TimeInForce


# =========================
# Portfolio / Account state
# =========================

@dataclass
class AccountState:
    agent_id: str
    cash: float = 0.0
    inventory: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    # For evaluation convenience (updated by simulator)
    last_mark_price: Optional[float] = None

    def equity(self) -> float:
        mark = self.last_mark_price if self.last_mark_price is not None else 0.0
        return self.cash + self.inventory * mark

    def __repr__(self) -> str:
        return (f"Account(agent={self.agent_id}, cash={self.cash:.2f}, inv={self.inventory:.4f}, "
                f"realized={self.realized_pnl:.2f}, fees={self.fees_paid:.4f})")


# =========================
# High-level Intent schema
# =========================

class Intent(BaseModel):
    """
    High-level decision abstraction the LLM agent produces.

    Semantics
    ---------
    - target_position: desired inventory after execution (absolute, not delta).
    - urgency: 0..1 controls aggressiveness (execution may choose market vs passive).
    - max_spread: willingness to cross spread (price tolerance in % of mid or absolute).
    - time_horizon: over how many ticks to work the order (executor schedules TWAP/POV).
    - note: optional rationale for auditing (ignored by executor).

    Execution layer will translate Intent to specific orders over subsequent ticks.
    """
    target_position: float = Field(..., description="Absolute desired inventory (units).")
    urgency: float = Field(0.5, ge=0.0, le=1.0, description="Aggressiveness 0..1.")
    max_spread: float = Field(0.0025, ge=0.0, description="Relative (e.g., 0.002=20bps) or abs if >1.0.")
    time_horizon: int = Field(20, ge=1, description="Ticks to work the order.")
    note: Optional[str] = Field(None, description="Optional audit rationale.")

    @field_validator("max_spread")
    @classmethod
    def _sanitize_spread(cls, v: float) -> float:
        # tiny negative due to FP? clamp.
        return max(0.0, float(v))


# =========================
# Low-level Order request
# =========================

class OrderRequest(BaseModel):
    """
    Concrete order request emitted by agents (baseline or executor).

    This mirrors exchange.Order minus runtime fields.
    """
    side: Side
    type: OrderType
    qty: float
    price: Optional[float] = None
    tif: TimeInForce = TimeInForce.GTC

    @field_validator("qty")
    @classmethod
    def _pos_qty(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("qty must be positive")
        return float(v)


# =========================
# Observation bundle
# =========================

@dataclass
class Observation:
    time: float
    mid: Optional[float]
    spread: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    last_trade: Optional[float]
    account: AccountState
    signals: Dict[str, Any] = field(default_factory=dict)
    # Optional top-of-book depth for crude liquidity sense (units)
    bid_depth: Optional[float] = None
    ask_depth: Optional[float] = None


# =========================
# Agent interface
# =========================

class Agent(ABC):
    """
    Abstract Agent.

    Lifecycle:
      - reset(seed): called once before simulation
      - on_observation(obs) -> Optional[OrderRequest] | Optional[Intent]
      - on_fill(price, qty, side, fee): update internal state if needed
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def reset(self, seed: Optional[int] = None) -> None:
        """Optional reproducibility hook."""
        return None

    @abstractmethod
    def on_observation(self, obs: Observation) -> Optional[OrderRequest] | Optional[Intent]:
        """Return an order or an intent; or None to do nothing at this tick."""
        raise NotImplementedError

    def on_fill(self, price: float, qty: float, side: Side, fee: float) -> None:
        """Receive execution reports if the simulator chooses to forward them."""
        return None
