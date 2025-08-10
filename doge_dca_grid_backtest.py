
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def backtest_dca_grid(csv_path,
                      base_order_usdt=10.0,
                      safety_orders=12,
                      step_pct=0.015,
                      volume_scale=1.4,
                      tp_pct=0.008,
                      max_position_usdt=500.0,
                      start_equity=1000.0,
                      plot_path=None,
                      trades_csv=None,
                      summary_json=None):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    close_col = None
    for c in ['close','closing price','close_price','c']:
        if c in df.columns:
            close_col = c; break
    time_col = None
    for c in ['timestamp','time','open time','open_time','date']:
        if c in df.columns:
            time_col = c; break
    if close_col is None:
        raise ValueError("No close column in CSV")
    if time_col is not None:
        try:
            t = pd.to_datetime(df[time_col])
        except Exception:
            try:
                t = pd.to_datetime(df[time_col], unit='ms')
            except Exception:
                t = pd.RangeIndex(len(df), name='bar')
    else:
        t = pd.RangeIndex(len(df), name='bar')
    df = pd.DataFrame({'time': t, 'close': df[close_col].astype(float)}).dropna().reset_index(drop=True)

    equity = start_equity
    equity_curve = []
    pos_qty = 0.0
    avg_entry = None
    level = -1
    next_buy_price = None
    last_fill_price = None
    size_usdt = base_order_usdt

    trades = []
    max_dd = 0.0; peak = equity
    deepest_level = 0; largest_pos_usdt = 0.0

    def qty_from_usdt(usdt, price):
        return usdt / price

    for _, row in df.iterrows():
        price = float(row['close']); tm = row['time']
        unrealized = (price - avg_entry) * pos_qty if pos_qty else 0.0
        eq_now = equity + unrealized
        equity_curve.append((tm, eq_now))
        if eq_now > peak: peak = eq_now
        dd = eq_now - peak
        if dd < max_dd: max_dd = dd

        if pos_qty == 0:
            qty = qty_from_usdt(base_order_usdt, price)
            if base_order_usdt <= max_position_usdt:
                pos_qty += qty; avg_entry = price; level = 0
                size_usdt = base_order_usdt
                last_fill_price = price
                next_buy_price = last_fill_price * (1 - step_pct)
                largest_pos_usdt = max(largest_pos_usdt, pos_qty * price)
                deepest_level = max(deepest_level, level)
            continue

        # TP check
        if price >= avg_entry * (1 + tp_pct):
            pnl = (price - avg_entry) * pos_qty
            equity += pnl
            trades.append({'time_close': tm, 'pnl': pnl, 'levels': level+1, 'avg_entry': avg_entry, 'exit_price': price})
            pos_qty = 0.0; avg_entry = None; level = -1
            next_buy_price = None; last_fill_price = None; size_usdt = base_order_usdt
            continue

        # Add safety order
        if next_buy_price is not None and price <= next_buy_price and level + 1 < safety_orders:
            next_usdt = size_usdt * volume_scale
            current_notional = pos_qty * price
            if current_notional + next_usdt <= max_position_usdt + 1e-9:
                qty = qty_from_usdt(next_usdt, price)
                pos_qty += qty
                avg_entry = ((avg_entry * (pos_qty - qty)) + price * qty) / pos_qty
                level += 1; size_usdt = next_usdt
                last_fill_price = price
                next_buy_price = last_fill_price * (1 - step_pct)
                deepest_level = max(deepest_level, level)
                largest_pos_usdt = max(largest_pos_usdt, pos_qty * price)
            else:
                next_buy_price = None

    eq = pd.DataFrame(equity_curve, columns=['time','equity'])
    summary = {
        'bars': len(df),
        'trades_closed': len(trades),
        'total_pnl': float(np.sum([t['pnl'] for t in trades])),
        'win_rate': float(np.mean([1.0 if t['pnl']>0 else 0.0 for t in trades])) if trades else float('nan'),
        'max_drawdown_abs': float(max_dd),
        'max_drawdown_pct': float(max_dd / start_equity * 100.0),
        'deepest_level_hit': int(deepest_level),
        'largest_position_usdt': float(largest_pos_usdt),
        'ending_equity': float(eq['equity'].iloc[-1]),
    }

    if plot_path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.plot(eq['time'], eq['equity'])
        plt.title("Equity Curve - DOGEUSDT 1H - DCA Grid (No SL)")
        plt.xlabel("Time"); plt.ylabel("Equity (USDT)")
        plt.tight_layout(); plt.savefig(plot_path)

    if trades_csv:
        pd.DataFrame(trades).to_csv(trades_csv, index=False)

    if summary_json:
        import json
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump({'params': {
                'base_order_usdt': base_order_usdt,
                'safety_orders': safety_orders,
                'step_pct': step_pct,
                'volume_scale': volume_scale,
                'tp_pct': tp_pct,
                'max_position_usdt': max_position_usdt,
                'start_equity': start_equity,
            }, 'summary': summary}, ensure_ascii=False, indent=2)

    return summary, eq, trades

if __name__ == "__main__":
    # Example run
    summary, eq, trades = backtest_dca_grid(
        csv_path="DOGEUSDT_1h.csv",
        base_order_usdt=10.0,
        safety_orders=12,
        step_pct=0.015,
        volume_scale=1.4,
        tp_pct=0.008,
        max_position_usdt=500.0,
        start_equity=1000.0,
        plot_path="doge_grid_equity.png",
        trades_csv="doge_grid_trades.csv",
        summary_json="doge_grid_summary.json"
    )
    print(summary)
