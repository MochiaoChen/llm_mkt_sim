# file: tests/test_order_book.py
from __future__ import annotations

import math
from exchange.order_book import OrderBook, Order, Side, OrderType, TimeInForce


def test_limit_match_and_spread():
    ob = OrderBook()
    # Seed the book
    ob.submit(Order("B1", "A", Side.BUY, OrderType.LIMIT, qty=10, price=99.0, timestamp=0.0))
    ob.submit(Order("S1", "B", Side.SELL, OrderType.LIMIT, qty=10, price=101.0, timestamp=0.0))
    assert ob.best_bid() == 99.0
    assert ob.best_ask() == 101.0
    assert math.isclose(ob.spread(), 2.0)

    # Cross with market buy, should trade at best ask 101
    trades, acks = ob.submit(Order("MB", "X", Side.BUY, OrderType.MARKET, qty=3, timestamp=1.0))
    assert len(trades) == 1
    assert trades[0].price == 101.0
    assert trades[0].qty == 3
    assert "filled" in acks or "accepted-market-partial-cancel" in acks


def test_ioc_and_fok():
    ob = OrderBook()
    # Only a small ask available
    ob.submit(Order("S1", "B", Side.SELL, OrderType.LIMIT, qty=2, price=100.0, timestamp=0.0))

    # IOC larger than available: partial then cancel remainder
    trades, acks = ob.submit(
        Order("B_IOC", "A", Side.BUY, OrderType.LIMIT, qty=5, price=101.0, timestamp=1.0, tif=TimeInForce.IOC)
    )
    assert sum(t.qty for t in trades) == 2
    assert any("partial" in a for a in acks)

    # FOK larger than available: reject
    trades, acks = ob.submit(
        Order("B_FOK", "A", Side.BUY, OrderType.LIMIT, qty=5, price=101.0, timestamp=2.0, tif=TimeInForce.FOK)
    )
    assert len(trades) == 0
    assert "rejected-fok" in acks
