
"""
DOGEUSDT DCA/Grid (No SL) — Bybit Unified Trading (v5) skeleton.
- Long-only, adds safety orders on % drops, closes entire position at small TP from avg.
- NO STOP LOSS. USE AT YOUR OWN RISK.
- Requires: pip install pybit (v5)
- Env: BYBIT_API_KEY, BYBIT_API_SECRET
"""

import os, time, math, logging
from dataclasses import dataclass
from typing import Optional
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
    step_pct: float = 0.015   # 1.5% between safety orders
    volume_scale: float = 1.4
    tp_pct: float = 0.008     # 0.8% TP from avg
    max_position_usdt: float = 500.0
    poll_sec: float = 3.0

class DogeDCA:
    def __init__(self, http: HTTP, cfg: Config):
        self.http = http
        self.cfg = cfg
        self.level = -1
        self.size_usdt = cfg.base_order_usdt
        self.next_buy_price: Optional[float] = None
        self.avg_entry: Optional[float] = None
        self.pos_qty: float = 0.0

        # market info
        info = self.http.get_instruments_info(category=cfg.category, symbol=cfg.symbol)
        lot_size_filter = info["result"]["list"][0]["lotSizeFilter"]
        self.qty_step = float(lot_size_filter["qtyStep"])
        self.min_qty = float(lot_size_filter["minOrderQty"])

        # set leverage
        try:
            self.http.set_leverage(category=cfg.category, symbol=cfg.symbol,
                                   buyLeverage=str(cfg.leverage), sellLeverage=str(cfg.leverage))
        except Exception as e:
            logging.warning("Cannot set leverage: %s", e)

    def round_qty(self, q: float) -> float:
        step = self.qty_step
        return max(self.min_qty, math.floor(q / step) * step)

    def last_price(self) -> float:
        r = self.http.get_tickers(category=self.cfg.category, symbol=self.cfg.symbol)
        return float(r["result"]["list"][0]["lastPrice"])

    def open_base_if_flat(self, price: float):
        if self.pos_qty == 0:
            qty = self.round_qty(self.cfg.base_order_usdt / price)
            if qty * price > self.cfg.max_position_usdt:
                logging.warning("Base order exceeds max position cap. Skipping.")
                return
            self.market_buy(qty)
            self.avg_entry = price
            self.level = 0
            self.size_usdt = self.cfg.base_order_usdt
            self.next_buy_price = price * (1 - self.cfg.step_pct)
            logging.info("Opened base: qty=%.0f avg=%.6f next_buy=%.6f", qty, self.avg_entry, self.next_buy_price)

    def market_buy(self, qty: float):
        qty = self.round_qty(qty)
        if qty < self.min_qty:
            logging.info("Qty < min, skip buy.")
            return
        self.http.place_order(
            category=self.cfg.category,
            symbol=self.cfg.symbol,
            side="Buy",
            orderType="Market",
            qty=str(qty),
            reduceOnly=False
        )
        self.pos_qty += qty

    def market_sell_all(self):
        if self.pos_qty <= 0:
            return
        qty = self.round_qty(self.pos_qty)
        self.http.place_order(
            category=self.cfg.category,
            symbol=self.cfg.symbol,
            side="Sell",
            orderType="Market",
            qty=str(qty),
            reduceOnly=True
        )
        self.pos_qty = 0.0

    def add_safety_if_needed(self, price: float):
        if self.next_buy_price is None or self.level + 1 >= self.cfg.safety_orders:
            return
        current_notional = self.pos_qty * price
        next_usdt = self.size_usdt * self.cfg.volume_scale
        if price <= self.next_buy_price and current_notional + next_usdt <= self.cfg.max_position_usdt + 1e-9:
            qty = self.round_qty(next_usdt / price)
            self.market_buy(qty)
            # update avg
            self.avg_entry = ((self.avg_entry * (self.pos_qty - qty)) + price * qty) / self.pos_qty
            self.level += 1
            self.size_usdt = next_usdt
            self.next_buy_price = price * (1 - self.cfg.step_pct)
            logging.info("Added safety L%d qty=%.0f avg=%.6f next_buy=%.6f", self.level, qty, self.avg_entry, self.next_buy_price)

    def tp_if_reached(self, price: float):
        if self.pos_qty > 0 and price >= self.avg_entry * (1 + self.cfg.tp_pct):
            logging.info("TP hit: price=%.6f avg=%.6f", price, self.avg_entry)
            self.market_sell_all()
            self.level = -1; self.size_usdt = self.cfg.base_order_usdt
            self.next_buy_price = None; self.avg_entry = None

    def loop(self):
        while True:
            try:
                price = self.last_price()
            except Exception as e:
                logging.warning("Price fetch failed: %s", e); time.sleep(self.cfg.poll_sec); continue

            self.open_base_if_flat(price)
            self.tp_if_reached(price)
            self.add_safety_if_needed(price)

            time.sleep(self.cfg.poll_sec)

def main():
    key = os.environ.get("BYBIT_API_KEY")
    sec = os.environ.get("BYBIT_API_SECRET")
    if not key or not sec:
        raise SystemExit("Set BYBIT_API_KEY and BYBIT_API_SECRET in environment")

    http = HTTP(api_key=key, api_secret=sec)
    bot = DogeDCA(http, Config())
    bot.loop()

if __name__ == "__main__":
    main()
