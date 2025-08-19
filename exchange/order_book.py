# file: exchange/order_book.py
"""
Event-driven limit order book (price-time priority).

Features
--------
- Supports LIMIT and MARKET orders, partial fills, cancels, IOC/FOK.
- Price-time priority via FIFO queues per price level.
- Snapshot helpers for spread/depth/midprice; coarse book export for evaluation.
- Deterministic matching with explicit timestamps (provided by simulator).

Dependencies
------------
- sortedcontainers (SortedDict) for price ladder ordering.

Notes
-----
- Time is a monotonic float or int (ticks). The simulator is responsible for
  supplying strictly non-decreasing timestamps.
- Prices are floats; for production you should quantize to ticks. We provide an
  optional tick_size argument for rounding.
- This module is intentionally self-contained so it can be unit-tested without
  the rest of the framework.

Disclaimer
----------
Research simulator only. Not investment advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
from sortedcontainers import SortedDict


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(str, Enum):
    GTC = "gtc"   # Good till cancel
    IOC = "ioc"   # Immediate or cancel (unfilled remainder cancels)
    FOK = "fok"   # Fill or kill (must fully fill immediately)


@dataclass
class Order:
    order_id: str
    agent_id: str
    side: Side
    type: OrderType
    qty: float
    timestamp: float
    price: Optional[float] = None
    tif: TimeInForce = TimeInForce.GTC

    # runtime fields
    remaining: float = field(init=False)
    is_active: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit order must have price.")
        if self.qty <= 0:
            raise ValueError("Order qty must be positive.")
        if self.type == OrderType.MARKET:
            self.price = None
        self.remaining = float(self.qty)


@dataclass
class Trade:
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    price: float
    qty: float
    timestamp: float


@dataclass
class CancelReport:
    order_id: str
    reason: str
    timestamp: float
    remaining: float


class OrderBook:
    """
    Price-time priority order book with event-driven matching.

    Public API (most common):
        - submit(order: Order) -> (trades, acks)
        - cancel(order_id: str, now: float) -> Optional[CancelReport]
        - best_bid(), best_ask(), midprice(), spread()
        - snapshot(n_levels: int) -> dict
    """

    def __init__(self, tick_size: Optional[float] = None) -> None:
        # Bids: highest price first -> use reverse-sorted view
        self.bids: SortedDict[float, Deque[Order]] = SortedDict()
        # Asks: lowest price first
        self.asks: SortedDict[float, Deque[Order]] = SortedDict()
        # Index for cancels/lookup
        self._order_index: Dict[str, Tuple[float, Deque[Order]]] = {}
        # Stats
        self.last_trade_price: Optional[float] = None
        self.trade_seq: int = 0
        self.tick_size = tick_size

    # ---------------------- helpers ----------------------

    def _quantize(self, price: float) -> float:
        if self.tick_size:
            return round(round(price / self.tick_size) * self.tick_size, 10)
        return price

    def _ladder(self, side: Side) -> SortedDict[float, Deque[Order]]:
        return self.bids if side == Side.BUY else self.asks

    def _opposite_ladder(self, side: Side) -> SortedDict[float, Deque[Order]]:
        return self.asks if side == Side.BUY else self.bids

    def _append_order(self, order: Order) -> None:
        if order.type != OrderType.LIMIT:
            return
        assert order.price is not None
        px = self._quantize(order.price)
        ladder = self._ladder(order.side)
        q = ladder.get(px)
        if q is None:
            q = deque()
            ladder[px] = q
        q.append(order)
        self._order_index[order.order_id] = (px, q)

    def _pop_empty_level(self, ladder: SortedDict[float, Deque[Order]], px: float) -> None:
        q = ladder.get(px)
        if q is not None and not q:
            del ladder[px]

    def _gen_trade_id(self) -> str:
        self.trade_seq += 1
        return f"T{self.trade_seq}"

    # ---------------------- public API ----------------------

    def best_bid(self) -> Optional[float]:
        if not self.bids:
            return None
        return next(reversed(self.bids.keys()))

    def best_ask(self) -> Optional[float]:
        if not self.asks:
            return None
        return next(iter(self.asks.keys()))

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return max(0.0, ba - bb)

    def midprice(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def total_depth(self, side: Side) -> float:
        ladder = self._ladder(side)
        return sum(o.remaining for _, q in ladder.items() for o in q)

    def snapshot(self, n_levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Return top-N price levels and quantities for bids/asks."""
        bids, asks = [], []

        # Asks: ascending
        for i, (px, q) in enumerate(self.asks.items()):
            if i >= n_levels:
                break
            asks.append((px, sum(o.remaining for o in q)))

        # Bids: descending
        for i, (px, q) in enumerate(reversed(self.bids.items())):
            if i >= n_levels:
                break
            bids.append((px, sum(o.remaining for o in q)))

        return {"bids": bids, "asks": asks, "mid": self.midprice(), "spread": self.spread()}

    def submit(self, order: Order) -> Tuple[List[Trade], List[str]]:
        """
        Submit an order and perform matching.

        Returns:
            trades: list of Trade generated by this submission
            acks:   list of textual acknowledgements (e.g., "accepted", "rejected-fok")
        """
        trades: List[Trade] = []
        acks: List[str] = []

        if order.type == OrderType.LIMIT and order.price is not None:
            order.price = self._quantize(order.price)

        # FOK feasibility quick check
        if order.tif == TimeInForce.FOK:
            if not self._can_fok_fill(order):
                acks.append("rejected-fok")
                return trades, acks

        # Match against opposite ladder
        if order.side == Side.BUY:
            trades.extend(self._execute_buy(order))
        else:
            trades.extend(self._execute_sell(order))

        # Handle residuals per TIF
        if order.remaining > 0:
            if order.type == OrderType.LIMIT and order.tif == TimeInForce.GTC:
                self._append_order(order)
                acks.append("accepted-resting")
            elif order.tif == TimeInForce.IOC:
                # cancel remainder silently
                acks.append("accepted-ioc-partial-cancel")
            elif order.tif == TimeInForce.FOK:
                # Should not happen due to pre-check; but if partial fill occurred, roll back? For simplicity, forbid partial by pre-check.
                acks.append("rejected-fok-partial")
            else:
                # MARKET with remainder: cancel remainder
                acks.append("accepted-market-partial-cancel")
        else:
            acks.append("filled")

        return trades, acks

    def cancel(self, order_id: str, now: float, reason: str = "user-cancel") -> Optional[CancelReport]:
        ref = self._order_index.get(order_id)
        if ref is None:
            return None
        px, q = ref
        # find and remove
        removed: Optional[Order] = None
        tmp = deque()
        while q:
            o = q.popleft()
            if o.order_id == order_id and o.is_active:
                removed = o
                o.is_active = False
                break
            tmp.append(o)
        # restore others
        while tmp:
            q.appendleft(tmp.pop())
        if removed is None:
            # already filled or canceled
            self._pop_empty_level(self._ladder(Side.BUY), px)  # harmless calls
            self._pop_empty_level(self._ladder(Side.SELL), px)
            return None

        self._pop_empty_level(self._ladder(removed.side), px)
        del self._order_index[order_id]
        return CancelReport(order_id=order_id, reason=reason, timestamp=now, remaining=removed.remaining)

    # ---------------------- matching internals ----------------------

    def _execute_buy(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        # Best ask first (ascending prices)
        while order.remaining > 1e-12 and self.asks:
            best_ask = next(iter(self.asks.keys()))
            # Price check for LIMIT
            if order.type == OrderType.LIMIT and order.price is not None and best_ask > order.price:
                break  # cannot cross
            # deque at best ask
            q = self.asks[best_ask]
            while order.remaining > 1e-12 and q:
                maker = q[0]
                fill_qty = min(order.remaining, maker.remaining)
                trade_px = best_ask
                trade = self._fill_pair(buy=order, sell=maker, price=trade_px, qty=fill_qty, ts=order.timestamp)
                trades.append(trade)
                if maker.remaining <= 1e-12:
                    maker.is_active = False
                    q.popleft()
                    self._remove_from_index_if_empty(maker, best_ask, q)
            if not q:
                del self.asks[best_ask]
            # MARKET order continues; LIMIT will loop until price condition breaks
            if order.type == OrderType.LIMIT and order.price is not None:
                # Loop condition handles break if cannot cross next level
                pass

        return trades

    def _execute_sell(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        # Best bid first (descending prices)
        while order.remaining > 1e-12 and self.bids:
            best_bid = next(reversed(self.bids.keys()))
            if order.type == OrderType.LIMIT and order.price is not None and best_bid < order.price:
                break
            q = self.bids[best_bid]
            while order.remaining > 1e-12 and q:
                maker = q[0]
                fill_qty = min(order.remaining, maker.remaining)
                trade_px = best_bid
                trade = self._fill_pair(buy=maker, sell=order, price=trade_px, qty=fill_qty, ts=order.timestamp)
                trades.append(trade)
                if maker.remaining <= 1e-12:
                    maker.is_active = False
                    q.popleft()
                    self._remove_from_index_if_empty(maker, best_bid, q)
            if not q:
                del self.bids[best_bid]
        return trades

    def _fill_pair(self, buy: Order, sell: Order, price: float, qty: float, ts: float) -> Trade:
        buy.remaining -= qty
        sell.remaining -= qty
        self.last_trade_price = price
        t = Trade(
            trade_id=self._gen_trade_id(),
            buy_order_id=buy.order_id,
            sell_order_id=sell.order_id,
            price=price,
            qty=qty,
            timestamp=ts,
        )
        return t

    def _remove_from_index_if_empty(self, o: Order, px: float, q: Deque[Order]) -> None:
        if o.order_id in self._order_index and o.remaining <= 1e-12:
            # Safe to delete; keep level cleanup to caller
            del self._order_index[o.order_id]

    def _can_fok_fill(self, order: Order) -> bool:
        """Dry-run check: is there enough opposite liquidity at acceptable prices to fully fill?"""
        needed = order.qty
        if order.side == Side.BUY:
            # Consume asks from best upward
            for px, q in self.asks.items():
                if order.type == OrderType.LIMIT and order.price is not None and px > order.price:
                    break
                avail = sum(o.remaining for o in q)
                needed -= avail
                if needed <= 1e-12:
                    return True
            return False
        else:
            for px, q in reversed(self.bids.items()):
                if order.type == OrderType.LIMIT and order.price is not None and px < order.price:
                    break
                avail = sum(o.remaining for o in q)
                needed -= avail
                if needed <= 1e-12:
                    return True
            return False

    # ---------------------- utilities for tests/exports ----------------------

    def level_sizes(self) -> Dict[str, int]:
        return {"bids": sum(len(q) for q in self.bids.values()), "asks": sum(len(q) for q in self.asks.values())}

    def dump_active_orders(self) -> List[Tuple[str, str, str, float, Optional[float], float]]:
        """
        For debugging: return [(order_id, agent_id, side, remaining, price, ts), ...]
        """
        out: List[Tuple[str, str, str, float, Optional[float], float]] = []
        for px, q in self.asks.items():
            for o in q:
                if o.is_active and o.remaining > 1e-12:
                    out.append((o.order_id, o.agent_id, o.side.value, o.remaining, px, o.timestamp))
        for px, q in self.bids.items():
            for o in q:
                if o.is_active and o.remaining > 1e-12:
                    out.append((o.order_id, o.agent_id, o.side.value, o.remaining, px, o.timestamp))
        return out
