
"""
DOGEUSDT DCA/Grid (No SL) — Bybit Unified Trading (v5) worker
- Long-only, DCA safety orders on % drops, close all at small TP from avg.
- Avoids opposite-direction positions (no short). One-Way mode enforced when possible.
- Idempotent orders via orderLinkId, qty rounded to exchange step, leverage set.
- Retries with exponential backoff on transient errors.
- Reads BYBIT_API_KEY/BYBIT_API_SECRET (or fallback API_KEY/API_SECRET).

Requirements:
  pip install pybit

ENV:
  BYBIT_API_KEY / API_KEY
  BYBIT_API_SECRET / API_SECRET
"""

import os, time, math, uuid, logging
from dataclasses import dataclass
from typing import Optional, Callable, Any
from pybit.unified_trading import HTTP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

@dataclass
class Config:
    symbol: str = "DOGEUSDT"
    category: str = "linear"
    leverage: int = 3
    base_order_usdt: float = 10.0
    safety_orders: int = 12
    step_pct: float = 0.015   # 1.5% grid step
    volume_scale: float = 1.4
    tp_pct: float = 0.008     # 0.8% take-profit from avg entry
    max_position_usdt: float = 500.0
    poll_sec: float = 3.0
    recv_window: int = 60000  # 60s like in t1.py

def with_retry(fn: Callable[..., Any], *args, **kwargs):
    """Retry helper with exponential backoff (5 tries)."""
    tries = kwargs.pop("_tries", 5)
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait = 1.2 * (2 ** i)
            logging.warning("API call failed (attempt %d/%d): %s; retrying in %.1fs",
                            i+1, tries, e, wait)
            time.sleep(wait)
    # last try
    return fn(*args, **kwargs)

class DogeDCA:
    def __init__(self, http: HTTP, cfg: Config):
        self.http = http
        self.cfg = cfg

        # internal state
        self.level = -1
        self.size_usdt = cfg.base_order_usdt
        self.next_buy_price: Optional[float] = None
        self.avg_entry: Optional[float] = None
        self.pos_qty: float = 0.0

        # read instrument filters and set leverage/mode
        info = with_retry(self.http.get_instruments_info, category=cfg.category, symbol=cfg.symbol)
        lot_size_filter = info["result"]["list"][0]["lotSizeFilter"]
        self.qty_step = float(lot_size_filter["qtyStep"])
        self.min_qty = float(lot_size_filter["minOrderQty"])

        # Switch to One-Way (avoid opposite direction) — ignore if already one-way or not supported
        try:
            with_retry(self.http.switch_position_mode, category=cfg.category, symbol=cfg.symbol, mode="OneWay")
            logging.info("Position mode: One-Way")
        except Exception as e:
            logging.warning("Cannot switch to One-Way mode (may already be One-Way): %s", e)

        # Set leverage
        try:
            with_retry(self.http.set_leverage, category=cfg.category, symbol=cfg.symbol,
                       buyLeverage=str(cfg.leverage), sellLeverage=str(cfg.leverage))
            logging.info("Leverage set to %dx", cfg.leverage)
        except Exception as e:
            logging.warning("Cannot set leverage: %s", e)

    # ---------- helpers ----------
    def round_qty(self, q: float) -> float:
        step = self.qty_step
        # floor to step, then ensure >= min_qty if > 0
        q = math.floor(q / step) * step
        if q > 0 and q < self.min_qty:
            q = self.min_qty
        return q

    def last_price(self) -> float:
        r = with_retry(self.http.get_tickers, category=self.cfg.category, symbol=self.cfg.symbol)
        return float(r["result"]["list"][0]["lastPrice"])

    def has_short(self) -> bool:
        """Detect if any short exposure exists; skip longs to avoid opposite direction."""
        try:
            r = with_retry(self.http.get_positions, category=self.cfg.category, symbol=self.cfg.symbol)
            lst = r.get("result", {}).get("list", [])
            for p in lst:
                side = (p.get("side") or "").lower()
                sz = float(p.get("size") or p.get("cumSize") or 0)
                if side == "sell" or sz < 0:
                    return True
        except Exception as e:
            logging.warning("get_positions failed: %s", e)
        return False

    def market_buy(self, qty: float):
        qty = self.round_qty(qty)
        if qty <= 0:
            logging.info("Qty < min; skip buy.")
            return
        link_id = str(uuid.uuid4())
        with_retry(self.http.place_order,
                   category=self.cfg.category,
                   symbol=self.cfg.symbol,
                   side="Buy",
                   orderType="Market",
                   qty=str(qty),
                   reduceOnly=False,
                   orderLinkId=link_id)
        self.pos_qty += qty
        logging.info("BUY market %s qty=%s link=%s", self.cfg.symbol, qty, link_id)

    def market_sell_all(self):
        if self.pos_qty <= 0:
            return
        qty = self.round_qty(self.pos_qty)
        link_id = str(uuid.uuid4())
        with_retry(self.http.place_order,
                   category=self.cfg.category,
                   symbol=self.cfg.symbol,
                   side="Sell",
                   orderType="Market",
                   qty=str(qty),
                   reduceOnly=True,   # close long only
                   orderLinkId=link_id)
        logging.info("SELL market (close) %s qty=%s link=%s", self.cfg.symbol, qty, link_id)
        self.pos_qty = 0.0

    # ---------- core logic ----------
    def open_base_if_flat(self, price: float):
        # Avoid opposite direction exposure (no short allowed)
        if self.has_short():
            logging.warning("Detected short exposure; skip opening long to avoid opposite directions.")
            return

        if self.pos_qty == 0:
            notional = self.cfg.base_order_usdt
            if notional > self.cfg.max_position_usdt:
                logging.warning("Base order exceeds max position cap. Skipping.")
                return
            qty = notional / price
            self.market_buy(qty)
            self.avg_entry = price
            self.level = 0
            self.size_usdt = self.cfg.base_order_usdt
            self.next_buy_price = price * (1 - self.cfg.step_pct)
            logging.info("Opened base: avg=%.6f next_buy=%.6f", self.avg_entry, self.next_buy_price)

    def add_safety_if_needed(self, price: float):
        if self.next_buy_price is None or self.level + 1 >= self.cfg.safety_orders:
            return
        current_notional = self.pos_qty * price
        next_usdt = self.size_usdt * self.cfg.volume_scale
        if price <= self.next_buy_price and current_notional + next_usdt <= self.cfg.max_position_usdt + 1e-9:
            qty = next_usdt / price
            # execute buy
            prev_qty = self.pos_qty
            self.market_buy(qty)
            new_qty = self.pos_qty
            filled = max(0.0, new_qty - prev_qty)
            if filled > 0:
                # update avg entry
                self.avg_entry = ((self.avg_entry * prev_qty) + price * filled) / new_qty
                self.level += 1
                self.size_usdt = next_usdt
                self.next_buy_price = price * (1 - self.cfg.step_pct)
                logging.info("Added safety L%d avg=%.6f next_buy=%.6f", self.level, self.avg_entry, self.next_buy_price)

    def tp_if_reached(self, price: float):
        if self.pos_qty > 0 and price >= self.avg_entry * (1 + self.cfg.tp_pct):
            logging.info("TP hit: price=%.6f avg=%.6f", price, self.avg_entry)
            self.market_sell_all()
            # reset state
            self.level = -1
            self.size_usdt = self.cfg.base_order_usdt
            self.next_buy_price = None
            self.avg_entry = None

    def loop(self):
        while True:
            try:
                price = self.last_price()
            except Exception as e:
                logging.warning("Price fetch failed: %s", e)
                time.sleep(self.cfg.poll_sec)
                continue

            self.open_base_if_flat(price)
            self.tp_if_reached(price)
            self.add_safety_if_needed(price)

            time.sleep(self.cfg.poll_sec)

def main():
    key = os.environ.get("BYBIT_API_KEY") or os.environ.get("API_KEY")
    sec = os.environ.get("BYBIT_API_SECRET") or os.environ.get("API_SECRET")
    if not key or not sec:
        raise SystemExit("Set BYBIT_API_KEY/BYBIT_API_SECRET (hoặc API_KEY/API_SECRET)")

    http = HTTP(api_key=key, api_secret=sec, recv_window=Config.recv_window)
    bot = DogeDCA(http, Config())
    bot.loop()

if __name__ == "__main__":
    main()
