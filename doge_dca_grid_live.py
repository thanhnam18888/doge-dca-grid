
"""
DOGEUSDT Flip-on-TP — Win-Only Close (Aggressive Preset, Throttled Logs)
-------------------------------------------------------------------------
- Always-in-market: LONG <-> SHORT; flip on TP.
- "Win-only": close only when net profit >= min_profit_usd (after taker fees). No loss-closes.
- TP default: Limit PostOnly (maker) reduceOnly. Detect fills by syncing positions.
- RSI gating (your rule): RSI_D > 70 ⇒ prefer SHORT (pause LONG); else prefer LONG (pause SHORT).
- Cooldown after TP: 10 minutes.
- DCA against adverse moves with limits; widen step when deep.
- Funding guard for SHORT (stricter): pause SHORT open/DCA if funding < -0.06%/8h.
- Extra logging: liqPrice & adlRank if available from Bybit.
- Throttled logs: Heartbeat every 30s, RiskView every 120s (configurable).

ENV: BYBIT_API_KEY/BYBIT_API_SECRET (or API_KEY/API_SECRET)
Needs: pip install pybit
"""

import os, time, math, uuid, logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Tuple, List
from pybit.unified_trading import HTTP

# -------- logging setup (honor LOG_LEVEL env if set) --------
_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _level, logging.INFO),
                    format="%(asctime)s %(levelname)s: %(message)s")

# ---------------------- config (Aggressive) ----------------------

@dataclass
class LongCfg:
    base_usdt: float = 24.0
    step_pct: float = 0.055          # 5.5%
    volume_scale: float = 1.18
    max_dca: int = 7
    tp_pct_floor: float = 0.01       # 1% min raw TP
    max_position_usdt: float = 175.0

@dataclass
class ShortCfg:
    base_usdt: float = 10.0
    step_pct: float = 0.075          # 7.5%
    volume_scale: float = 1.15
    max_dca: int = 3
    tp_pct_floor: float = 0.01       # 1%
    max_position_usdt: float = 85.0

@dataclass
class RiskCfg:
    leverage: float = 2.0
    taker_fee: float = 0.0006
    maker_fee: float = 0.0002
    min_profit_usd: float = 0.10
    close_requires_net_profit: bool = True
    widen_step_level: int = 3        # widen when DCA level >= 3
    widen_step_add: float = 0.03     # add +3% to step when deep

@dataclass
class GuardCfg:
    cooldown_seconds: int = 600      # 10 minutes
    rsi_period: int = 14
    rsi_refresh_sec: int = 1800
    short_funding_pause: float = -0.0006  # pause SHORT if funding < -0.06%/8h
    prefer_short_rsi: float = 70.0        # RSI_D > 70 => prefer SHORT

@dataclass
class TPModeCfg:
    tp_order_mode: str = "limit_postonly"  # default: maker; alt: "market_conditional"

@dataclass
class LogCfg:
    heartbeat_sec: int = int(os.environ.get("LOG_HEARTBEAT_SEC", "30"))  # status every 30s
    riskview_sec: int = int(os.environ.get("LOG_RISKVIEW_SEC", "120"))   # liq/adl every 120s

@dataclass
class Config:
    symbol: str = "DOGEUSDT"
    category: str = "linear"
    poll_sec: float = 3.0
    recv_window: int = 60000
    long: LongCfg = field(default_factory=LongCfg)
    short: ShortCfg = field(default_factory=ShortCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    guard: GuardCfg = field(default_factory=GuardCfg)
    tp_mode: TPModeCfg = field(default_factory=TPModeCfg)
    log: LogCfg = field(default_factory=LogCfg)

# ---------------------- utils ----------------------

def with_retry(fn: Callable[..., Any], *args, **kwargs):
    tries = kwargs.pop("_tries", 5)
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait = 1.2 * (2 ** i)
            logging.warning("API call failed (attempt %d/%d): %s; retrying in %.1fs", i+1, tries, e, wait)
            time.sleep(wait)
    return fn(*args, **kwargs)

def calc_rsi_from_closes(closes: List[float], period: int) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = []; losses = []
    for i in range(1, period+1):
        diff = closes[-i] - closes[-i-1]
        gains.append(max(diff, 0.0)); losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# ---------------------- bot ----------------------

class DogeFlipAggressiveWinOnly:
    def __init__(self, http: HTTP, cfg: Config):
        self.http = http; self.cfg = cfg

        # state
        self.side = "long"                  # "long" | "short"
        self.cooldown_until = 0.0
        self.pos_qty = 0.0
        self.avg_entry: Optional[float] = None
        self.level = -1
        self.size_usdt = 0.0
        self.next_price: Optional[float] = None
        self.tp_order_id: Optional[str] = None
        self.entry_fees_paid_usd = 0.0

        # log throttling timers
        self._t_last_hb = 0.0
        self._t_last_risk = 0.0

        # instrument
        info = with_retry(self.http.get_instruments_info, category=cfg.category, symbol=cfg.symbol)
        inst = info["result"]["list"][0]
        lot = inst["lotSizeFilter"]; pricef = inst["priceFilter"]
        self.qty_step = float(lot["qtyStep"]); self.min_qty = float(lot["minOrderQty"])
        self.tick_size = float(pricef["tickSize"])

        # one-way & leverage
        try:
            with_retry(self.http.switch_position_mode, category=cfg.category, symbol=cfg.symbol, mode=0)
            logging.info("Position mode One-Way")
        except Exception as e:
            logging.warning("Cannot switch One-Way (ok): %s", e)
        try:
            with_retry(self.http.set_leverage, category=cfg.category, symbol=cfg.symbol,
                       buyLeverage=str(cfg.risk.leverage), sellLeverage=str(cfg.risk.leverage))
            logging.info("Leverage set to %sx", cfg.risk.leverage)
        except Exception as e:
            logging.warning("Cannot set leverage: %s", e)

        # RSI cache
        self._rsi_ts = 0.0; self._rsi_daily = None

    # helpers
    def round_qty(self, q: float) -> float:
        q = math.floor(q / self.qty_step) * self.qty_step
        if q > 0 and q < self.min_qty: q = self.min_qty
        return q

    def round_px(self, px: float) -> float:
        return math.floor(px / self.tick_size) * self.tick_size

    def last_mark(self) -> Tuple[float, float]:
        r = with_retry(self.http.get_tickers, category=self.cfg.category, symbol=self.cfg.symbol)
        item = r["result"]["list"][0]
        last = float(item["lastPrice"]); mark = float(item.get("markPrice", last))
        return last, mark

    def get_daily_rsi(self) -> Optional[float]:
        now = time.time()
        if now - self._rsi_ts < self.cfg.guard.rsi_refresh_sec and self._rsi_daily is not None:
            return self._rsi_daily
        try:
            r = with_retry(self.http.get_kline, category="linear", symbol=self.cfg.symbol, interval="D", limit=120)
            lst = sorted(r["result"]["list"], key=lambda x: int(x[0]))
            closes = [float(x[4]) for x in lst]
            rsi_d = calc_rsi_from_closes(closes, self.cfg.guard.rsi_period)
            self._rsi_daily = rsi_d; self._rsi_ts = now
            return rsi_d
        except Exception as e:
            logging.warning("RSI fetch failed: %s", e)
            return None

    def fetch_funding_rate(self) -> Optional[float]:
        try:
            r = with_retry(self.http.get_tickers, category=self.cfg.category, symbol=self.cfg.symbol)
            item = r["result"]["list"][0]
        except Exception:
            return None
        fr = item.get("fundingRate")
        try:
            return float(fr) if fr is not None else None
        except Exception:
            return None

    # sync live position (maker TP detection) + get liq/adl
    def sync_position(self):
        try:
            r = with_retry(self.http.get_positions, category=self.cfg.category, symbol=self.cfg.symbol)
            lst = r.get("result", {}).get("list", [])
            size = 0.0; avg = None; liq = None; adl = None
            for p in lst:
                sz = float(p.get("size") or 0.0)
                if sz > 0:
                    size = sz
                    avg = float(p.get("avgPrice") or 0.0)
                    liq = p.get("liqPrice")
                    try:
                        liq = float(liq) if liq is not None else None
                    except Exception:
                        liq = None
                    adl = p.get("adlRankIndicator") or p.get("adlRank") or p.get("positionIdx")
                    break
            # detect TP fill
            if size == 0.0 and self.pos_qty > 0.0:
                logging.info("Detected position closed on exchange (likely TP filled).")
                self._on_tp_filled_flip()
            else:
                # refresh local state
                self.pos_qty = size
                self.avg_entry = avg if size > 0 else None
                # throttled risk view
                if size > 0:
                    now = time.time()
                    if now - self._t_last_risk >= self.cfg.log.riskview_sec:
                        logging.info("RiskView liqPrice=%s adlRank=%s",
                                     f"{liq:.6f}" if liq is not None else "NA", str(adl))
                        self._t_last_risk = now
        except Exception as e:
            logging.warning("sync_position failed: %s", e)

    # fees & pnl
    def _add_entry_fee(self, notional: float):
        self.entry_fees_paid_usd += notional * self.cfg.risk.taker_fee

    def expected_net_pnl(self, exit_price: float) -> float:
        if self.pos_qty <= 0 or self.avg_entry is None: return 0.0
        if self.side == "long":
            gross = (exit_price - self.avg_entry) * self.pos_qty
        else:
            gross = (self.avg_entry - exit_price) * self.pos_qty
        exit_fee = exit_price * self.pos_qty * self.cfg.risk.taker_fee
        return gross - self.entry_fees_paid_usd - exit_fee

    def dynamic_step(self) -> float:
        add = self.cfg.risk.widen_step_add if self.level >= self.cfg.risk.widen_step_level else 0.0
        base = self.cfg.long.step_pct if self.side == "long" else self.cfg.short.step_pct
        return base + add

    def calc_tp_target_win_only(self) -> Optional[float]:
        if self.pos_qty <= 0 or self.avg_entry is None: return None
        qty = self.pos_qty; fee = self.cfg.risk.taker_fee; minp = self.cfg.risk.min_profit_usd
        if self.side == "long":
            rhs = (self.entry_fees_paid_usd + minp) / max(qty, 1e-9)
            P = (self.avg_entry + rhs) / (1.0 - fee)
            P_floor = self.avg_entry * (1.0 + self.cfg.long.tp_pct_floor)
            target = max(P, P_floor)
        else:
            rhs = (self.entry_fees_paid_usd + minp) / max(qty, 1e-9)
            P = (self.avg_entry - rhs) / (1.0 + fee)
            P_floor = self.avg_entry * (1.0 - self.cfg.short.tp_pct_floor)
            target = min(P, P_floor)
        return self.round_px(target)

    # TP orders
    def cancel_tp(self):
        if not self.tp_order_id: return
        try:
            with_retry(self.http.cancel_order, category=self.cfg.category, symbol=self.cfg.symbol, orderId=self.tp_order_id)
            logging.info("Cancelled TP %s", self.tp_order_id)
        except Exception as e:
            logging.warning("Cancel TP failed: %s", e)
        finally:
            self.tp_order_id = None

    def place_tp(self):
        trg = self.calc_tp_target_win_only()
        if trg is None or self.pos_qty <= 0: return
        qty = self.round_qty(self.pos_qty)
        self.cancel_tp()
        link = str(uuid.uuid4())
        if self.cfg.tp_mode.tp_order_mode == "limit_postonly":
            side = "Sell" if self.side == "long" else "Buy"
            order = with_retry(self.http.place_order,
                               category=self.cfg.category, symbol=self.cfg.symbol,
                               side=side, orderType="Limit", qty=str(qty),
                               price=str(trg), reduceOnly=True, timeInForce="PostOnly",
                               closeOnTrigger=False, orderLinkId=link)
            self.tp_order_id = order.get("result", {}).get("orderId")
            logging.info("TP (maker) %s: price=%.6f qty=%s id=%s", self.side, trg, qty, self.tp_order_id)
        else:
            side = "Sell" if self.side == "long" else "Buy"
            trig_dir = 1 if self.side == "long" else 2
            order = with_retry(self.http.place_order,
                               category=self.cfg.category, symbol=self.cfg.symbol,
                               side=side, orderType="Market", qty=str(qty),
                               triggerDirection=trig_dir, triggerPrice=str(trg),
                               reduceOnly=True, closeOnTrigger=True, orderLinkId=link)
            self.tp_order_id = order.get("result", {}).get("orderId")
            logging.info("TP (conditional) %s: trigger=%.6f qty=%s id=%s", self.side, trg, qty, self.tp_order_id)

    # order ops
    def mkt_buy(self, qty: float, price: float):
        qty = self.round_qty(qty); 
        if qty <= 0: return
        notional = qty * price
        link = str(uuid.uuid4())
        with_retry(self.http.place_order, category=self.cfg.category, symbol=self.cfg.symbol,
                   side="Buy", orderType="Market", qty=str(qty), reduceOnly=False, orderLinkId=link)
        self.pos_qty += qty; self._add_entry_fee(notional)
        logging.info("BUY qty=%s link=%s", qty, link)

    def mkt_sell(self, qty: float, price: float):
        qty = self.round_qty(qty); 
        if qty <= 0: return
        notional = qty * price
        link = str(uuid.uuid4())
        with_retry(self.http.place_order, category=self.cfg.category, symbol=self.cfg.symbol,
                   side="Sell", orderType="Market", qty=str(qty), reduceOnly=False, orderLinkId=link)
        self.pos_qty += qty; self._add_entry_fee(notional)
        logging.info("SELL qty=%s link=%s", qty, link)

    def _on_tp_filled_flip(self):
        # called when exchange closed position (limit maker TP filled)
        self._start_cooldown()
        self._flip_side()
        self._reset_position_state()

    def close_long_market_guarded(self, price: float):
        if self.pos_qty <= 0: return
        if self.cfg.risk.close_requires_net_profit and self.expected_net_pnl(price) < self.cfg.risk.min_profit_usd:
            logging.info("Skip close LONG — net pnl not positive yet."); return
        qty = self.round_qty(self.pos_qty); link = str(uuid.uuid4())
        with_retry(self.http.place_order, category=self.cfg.category, symbol=self.cfg.symbol,
                   side="Sell", orderType="Market", qty=str(qty), reduceOnly=True, orderLinkId=link)
        logging.info("CLOSE LONG qty=%s link=%s", qty, link)
        self._start_cooldown(); self._flip_side(); self._reset_position_state()

    def close_short_market_guarded(self, price: float):
        if self.pos_qty <= 0: return
        if self.cfg.risk.close_requires_net_profit and self.expected_net_pnl(price) < self.cfg.risk.min_profit_usd:
            logging.info("Skip close SHORT — net pnl not positive yet."); return
        qty = self.round_qty(self.pos_qty); link = str(uuid.uuid4())
        with_retry(self.http.place_order, category=self.cfg.category, symbol=self.cfg.symbol,
                   side="Buy", orderType="Market", qty=str(qty), reduceOnly=True, orderLinkId=link)
        logging.info("CLOSE SHORT qty=%s link=%s", qty, link)
        self._start_cooldown(); self._flip_side(); self._reset_position_state()

    def _reset_position_state(self):
        self.pos_qty = 0.0; self.avg_entry = None; self.level = -1
        self.size_usdt = 0.0; self.next_price = None; self.entry_fees_paid_usd = 0.0
        self.cancel_tp()

    # core: open/DCA and TP+flip
    def maybe_open_or_dca(self, price: float):
        # funding guard for short
        if self.side == "short":
            fr = self.fetch_funding_rate()
            if fr is not None and fr < self.cfg.guard.short_funding_pause:
                logging.warning("Funding too negative (%.5f). Pause opening/DCA SHORT.", fr)
                return

        dyn_step = self.dynamic_step()
        if self.side == "long":
            if self.pos_qty == 0:
                if self._notional(price) > self.cfg.long.max_position_usdt:
                    logging.warning("Base long exceeds cap; skip"); return
                qty = self.cfg.long.base_usdt / price
                self.mkt_buy(qty, price)
                self.avg_entry = price; self.level = 0; self.size_usdt = self.cfg.long.base_usdt
                self.next_price = price * (1 - dyn_step)
                self.place_tp()
                logging.info("Open LONG avg=%.6f next=%.6f", self.avg_entry, self.next_price); return
            if self.level < self.cfg.long.max_dca and price <= self.next_price:
                next_usdt = self.size_usdt * self.cfg.long.volume_scale
                if self._notional(price) + next_usdt <= self.cfg.long.max_position_usdt + 1e-9:
                    prev_qty = self.pos_qty
                    self.mkt_buy(next_usdt / price, price)
                    filled = max(0.0, self.pos_qty - prev_qty)
                    if filled > 0:
                        self.avg_entry = ((self.avg_entry * prev_qty) + price * filled) / self.pos_qty
                        self.level += 1; self.size_usdt = next_usdt
                        self.next_price = price * (1 - self.dynamic_step())
                        self.place_tp()
                        logging.info("DCA LONG L%d avg=%.6f next=%.6f", self.level, self.avg_entry, self.next_price)
        else:
            if self.pos_qty == 0:
                if self._notional(price) > self.cfg.short.max_position_usdt:
                    logging.warning("Base short exceeds cap; skip"); return
                qty = self.cfg.short.base_usdt / price
                self.mkt_sell(qty, price)
                self.avg_entry = price; self.level = 0; self.size_usdt = self.cfg.short.base_usdt
                self.next_price = price * (1 + dyn_step)
                self.place_tp()
                logging.info("Open SHORT avg=%.6f next=%.6f", self.avg_entry, self.next_price); return
            if self.level < self.cfg.short.max_dca and price >= self.next_price:
                next_usdt = self.size_usdt * self.cfg.short.volume_scale
                if self._notional(price) + next_usdt <= self.cfg.short.max_position_usdt + 1e-9:
                    prev_qty = self.pos_qty
                    self.mkt_sell(next_usdt / price, price)
                    filled = max(0.0, self.pos_qty - prev_qty)
                    if filled > 0:
                        self.avg_entry = ((self.avg_entry * prev_qty) + price * filled) / self.pos_qty
                        self.level += 1; self.size_usdt = next_usdt
                        self.next_price = price * (1 + self.dynamic_step())
                        self.place_tp()
                        logging.info("DCA SHORT L%d avg=%.6f next=%.6f", self.level, self.avg_entry, self.next_price)

    def check_tp_and_flip(self, price: float, mark: float):
        # If tp_mode is maker limit, rely on exchange fill + sync_position
        if self.cfg.tp_mode.tp_order_mode == "limit_postonly":
            return
        # else market-conditional fallback (not default)
        trg = self.calc_tp_target_win_only()
        if self.pos_qty <= 0 or self.avg_entry is None or trg is None: return
        px = max(price, mark) if self.side == "long" else min(price, mark)
        if self.side == "long":
            if px >= trg:
                self.close_long_market_guarded(px)
        else:
            if px <= trg:
                self.close_short_market_guarded(px)

    def _start_cooldown(self):
        self.cooldown_until = time.time() + self.cfg.guard.cooldown_seconds
        logging.info("Cooldown for %d seconds", self.cfg.guard.cooldown_seconds)

    def _flip_side(self):
        rsi_d = self.get_daily_rsi()
        if rsi_d is not None and rsi_d > self.cfg.guard.prefer_short_rsi:
            self.side = "short"
            logging.info("Next side=SHORT (RSI_D=%.1f > %.1f).", rsi_d, self.cfg.guard.prefer_short_rsi)
        else:
            self.side = "long"
            logging.info("Next side=LONG  (RSI_D=%s <= %.1f).",
                         f"{rsi_d:.1f}" if rsi_d is not None else "NA", self.cfg.guard.prefer_short_rsi)

    def _notional(self, price: float) -> float:
        return self.pos_qty * price

    def _reset_position_state(self):
        self.pos_qty = 0.0; self.avg_entry = None; self.level = -1
        self.size_usdt = 0.0; self.next_price = None; self.entry_fees_paid_usd = 0.0
        self.cancel_tp()

    # main loop
    def loop(self):
        while True:
            try:
                last, mark = self.last_mark()
                price = (last + mark) / 2.0
            except Exception as e:
                logging.warning("Price fetch fail: %s", e); time.sleep(self.cfg.poll_sec); continue

            # always sync first to catch maker TP fills
            self.sync_position()

            if time.time() < self.cooldown_until:
                time.sleep(self.cfg.poll_sec); continue

            self.maybe_open_or_dca(price)
            self.check_tp_and_flip(price, mark)

            # throttled heartbeat
            now = time.time()
            if now - self._t_last_hb >= self.cfg.log.heartbeat_sec:
                pnl = None
                if self.avg_entry and self.pos_qty > 0:
                    pnl = self.expected_net_pnl(price)
                rsi_d = self.get_daily_rsi()
                logging.info("HB side=%s pos=%.0f avg=%.6f px=%.6f pnl≈%s rsiD=%s L=%d",
                             self.side, self.pos_qty, self.avg_entry or 0.0, price,
                             f"{pnl:.2f}" if pnl is not None else "NA",
                             f"{rsi_d:.1f}" if rsi_d is not None else "NA",
                             self.level)
                self._t_last_hb = now

            time.sleep(self.cfg.poll_sec)

# ---------------------- main ----------------------

def main():
    key = os.environ.get("BYBIT_API_KEY") or os.environ.get("API_KEY")
    sec = os.environ.get("BYBIT_API_SECRET") or os.environ.get("API_SECRET")
    if not key or not sec:
        raise SystemExit("Set BYBIT_API_KEY/BYBIT_API_SECRET (hoặc API_KEY/API_SECRET)")

    cfg = Config()  # Aggressive defaults baked in + throttled logs
    http = HTTP(api_key=key, api_secret=sec, recv_window=cfg.recv_window)
    bot = DogeFlipAggressiveWinOnly(http, cfg)
    bot.loop()

if __name__ == "__main__":
    main()
