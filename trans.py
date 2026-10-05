# -*- coding: utf-8 -*-
"""缠论逐日无未来函数模拟交易。

策略核心复用 chan_strategy_core.py（来自长文本-1790912360.txt）。
每个交易日只把截至当日的数据交给算法；仅交易当日新确认的正式买卖点。

默认规则：
- 初始资金 100000；每次买点使用可用现金 30%；每次卖点卖出持仓 50%。
- 1B/2B/3B 均可加仓，1S/2S/3S 均可减仓；潜在点（带 ?）不交易。
- 每次买入后，把该买点价格设为保护位且只上移；后续日线 low 跌破时全仓止损。
- 默认信号在当日收盘确认，并按当日收盘成交；可用 --execution next_open 改成次日开盘成交。
- 默认每次计算只使用当前日及之前 180 个自然日的数据（可配置）。
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

import chan as core


@dataclass
class PositionLot:
    shares: int
    stop_price: float
    source_type: str
    buy_date: str


@dataclass
class Portfolio:
    cash: float
    lots: list = None

    def __post_init__(self):
        if self.lots is None:
            self.lots = []

    @property
    def shares(self):
        return sum(x.shares for x in self.lots)

    @property
    def stop_price(self):
        """仅用于权益输出；多批次时返回最高保护位。"""
        return max((x.stop_price for x in self.lots), default=None)


@dataclass
class Trade:
    signal_date: str
    trade_date: str
    action: str
    signal: str
    price: float
    shares: int
    cash_after: float
    shares_after: int
    equity_after: float
    note: str


def ensure_macd(df: pd.DataFrame) -> pd.DataFrame:
    """在当前可见窗口内计算 MACD，不读取未来数据。"""
    out = df.copy().reset_index(drop=True)
    ema12 = out['close'].ewm(span=12, adjust=False).mean()
    ema26 = out['close'].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    out['MACDh_12_26_9'] = (dif - dea) * 2
    return out


def calculate_visible_signals(visible_df: pd.DataFrame):
    """仅用 visible_df 重算结构并返回信号；visible_df 最后一行就是“今天”。"""
    data = ensure_macd(visible_df)
    mdf = core.process_inclusion(data)
    fractals = core.find_fractals(mdf)
    strokes = core.build_strokes(fractals, mdf, min_k_between=3)
    pivots = core.build_pivots(strokes)
    buys, sells = core.find_buy_sell_points(strokes, pivots, mdf, data)
    # 失败检测也只能看到当日及以前的数据。
    core.apply_signal_failure_pipeline(buys, sells, pivots, strokes, mdf, data)
    return data, mdf, buys, sells


def signal_date(sig, data: pd.DataFrame, mdf: pd.DataFrame) -> pd.Timestamp:
    oi = int(mdf['orig_idx'].values[int(sig['idx'])])
    return pd.Timestamp(data.iloc[oi]['time_key']).normalize()


def signal_key(sig, data, mdf):
    """稳定信号键；信号通常会在分型确认日才首次出现，但日期可能落在前几根 K。"""
    dt = signal_date(sig, data, mdf)
    return sig['type'], str(dt.date()), round(float(sig['price']), 6)


def collect_formal_signals(signals, data, mdf):
    out = []
    for s in signals:
        if (s['type'].endswith('?') or s.get('invalidated')
                or s.get('failed')):
            continue
        out.append((signal_key(s, data, mdf), s))
    return out


def choose_new_signal(signals, side):
    """同一确认日新增多个信号时只执行一个：3类 > 2类 > 1类。"""
    if not signals:
        return None
    order = ({'3B': 3, '2B': 2, '1B': 1} if side == 'buy'
             else {'3S': 3, '2S': 2, '1S': 1})
    return max(signals, key=lambda s: order.get(s['type'], 0))


def fill_price(raw_price: float, side: str, slippage_bps: float) -> float:
    rate = slippage_bps / 10000.0
    return raw_price * (1 + rate if side == 'buy' else 1 - rate)


def buy(portfolio, budget_fraction, raw_price, lot_size, commission_rate,
        min_commission, slippage_bps, stop_price, source_type, buy_date):
    price = fill_price(raw_price, 'buy', slippage_bps)
    budget = portfolio.cash * budget_fraction
    unit_cost = price * lot_size
    lots = int(budget // unit_cost)
    while lots > 0:
        qty = lots * lot_size
        value = qty * price
        fee = max(value * commission_rate, min_commission)
        if value + fee <= portfolio.cash + 1e-9:
            portfolio.cash -= value + fee
            portfolio.lots.append(PositionLot(
                shares=qty, stop_price=float(stop_price),
                source_type=source_type, buy_date=str(pd.Timestamp(buy_date).date())))
            return qty, price, fee
        lots -= 1
    return 0, price, 0.0


def _remove_shares_fifo(portfolio, qty):
    """普通减仓按 FIFO 从各批次扣减，不改变剩余批次的独立保护位。"""
    remain = qty
    for lot in list(portfolio.lots):
        take = min(lot.shares, remain)
        lot.shares -= take
        remain -= take
        if lot.shares == 0:
            portfolio.lots.remove(lot)
        if remain == 0:
            break


def sell(portfolio, holding_fraction, raw_price, lot_size, commission_rate,
         min_commission, slippage_bps, all_position=False):
    if portfolio.shares <= 0:
        return 0, raw_price, 0.0
    if all_position:
        qty = portfolio.shares
    else:
        target = math.ceil(portfolio.shares * holding_fraction)
        qty = min(portfolio.shares, int(math.ceil(target / lot_size) * lot_size))
    price = fill_price(raw_price, 'sell', slippage_bps)
    value = qty * price
    fee = max(value * commission_rate, min_commission)
    portfolio.cash += value - fee
    _remove_shares_fifo(portfolio, qty)
    return qty, price, fee


def append_trade(trades, signal_dt, trade_dt, action, sig_type, price, qty,
                 portfolio, mark_price, note):
    trades.append(Trade(
        signal_date=str(pd.Timestamp(signal_dt).date()),
        trade_date=str(pd.Timestamp(trade_dt).date()),
        action=action,
        signal=sig_type,
        price=float(price),
        shares=int(qty),
        cash_after=float(portfolio.cash),
        shares_after=int(portfolio.shares),
        equity_after=float(portfolio.cash + portfolio.shares * mark_price),
        note=note,
    ))


def weekly_trend_state(visible_df, today):
    """只用已完成周K判断高一级别趋势，返回 bull/neutral/bear。"""
    x = visible_df.set_index('time_key').sort_index()
    weekly = x.resample('W-FRI').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    # 非周五时，本周周K仍未完成，禁止拿它参与过滤。
    weekly = weekly[weekly.index <= pd.Timestamp(today)]
    if len(weekly) < 12:
        return 'neutral'
    close = weekly['close']
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    if close.iloc[-1] > ema10.iloc[-1] and ema10.iloc[-1] > ema10.iloc[-2] and hist.iloc[-1] >= 0:
        return 'bull'
    if close.iloc[-1] < ema20.iloc[-1] and ema10.iloc[-1] < ema10.iloc[-2] and hist.iloc[-1] < 0:
        return 'bear'
    return 'neutral'


def run_backtest(df, start, end, initial_cash=100000.0, history_days=180,
                 buy_fractions=None, sell_fractions=None, lot_size=1,
                 commission_rate=0.0003, min_commission=0.0,
                 slippage_bps=0.0, execution='close', stop_mode='intraday',
                 signal_cache=None, weekly_buy_matrix=None, cooldown_bars=10):
    """逐日回放；分级仓位、独立保护位、周线×买点二维过滤和1B冷却。"""
    buy_fractions = buy_fractions or {'1B': 0.40, '2B': 0.35, '3B': 0.25}
    sell_fractions = sell_fractions or {'1S': 0.40, '2S': 0.35, '3S': 1.00}
    weekly_buy_matrix = weekly_buy_matrix or {
        'bull': {'1B': 1.00, '2B': 1.00, '3B': 1.00},
        'neutral': {'1B': 0.40, '2B': 0.70, '3B': 1.00},
        'bear': {'1B': 0.10, '2B': 0.40, '3B': 0.70},
    }
    signal_cache = signal_cache if signal_cache is not None else {}
    df = df.sort_values('time_key').drop_duplicates('time_key').reset_index(drop=True)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    trade_rows = df[(df.time_key >= start) & (df.time_key <= end)]
    if trade_rows.empty:
        raise ValueError(f'回测区间 {start.date()}~{end.date()} 没有 K 线')

    p = Portfolio(float(initial_cash))
    trades = []
    equity_rows = []
    pending = None
    seen_signal_keys = set()
    first_replay_day = True
    cooldown_remaining = 0
    history_start = start - pd.Timedelta(days=history_days)

    for global_i in trade_rows.index:
        row = df.loc[global_i]
        today = pd.Timestamp(row.time_key).normalize()
        visible_now = df[(df.time_key >= history_start) & (df.time_key <= today)].copy()
        week_key = f'week:{today.date()}'
        if week_key in signal_cache:
            week_state = signal_cache[week_key]
        else:
            week_state = weekly_trend_state(visible_now, today)
            signal_cache[week_key] = week_state
        stopped_today = False

        # next_open：执行上一交易日收盘确认的信号。
        if pending is not None:
            sig, side, signal_dt = pending
            raw = float(row.open)
            if side == 'buy' and (sig['type'] != '1B' or cooldown_remaining <= 0):
                # 冷却只限制重复1B；2B/3B属于确认信号，可直接解除冷却。
                if sig['type'] in ('2B', '3B'):
                    cooldown_remaining = 0
                frac = buy_fractions[sig['type']] * weekly_buy_matrix[week_state][sig['type']]
                qty, px, fee = buy(p, frac, raw, lot_size,
                                   commission_rate, min_commission, slippage_bps,
                                   sig['price'], sig['type'], today)
                if qty:
                    append_trade(trades, signal_dt, today, 'BUY', sig['type'], px,
                                 qty, p, float(row.close), f'次日开盘成交；手续费 {fee:.2f}')
            elif side == 'sell':
                frac = sell_fractions[sig['type']]
                qty, px, fee = sell(p, frac, raw, lot_size,
                                    commission_rate, min_commission, slippage_bps)
                if qty:
                    append_trade(trades, signal_dt, today, 'SELL', sig['type'], px,
                                 qty, p, float(row.close), f'次日开盘成交；手续费 {fee:.2f}')
            pending = None

        # 每个买入批次独立止损；新买点不会抬高旧仓保护位。
        triggered = []
        for lot in list(p.lots):
            broken = (float(row.low) < lot.stop_price if stop_mode == 'intraday'
                      else float(row.close) < lot.stop_price)
            if broken:
                triggered.append(lot)
        if triggered:
            qty = sum(x.shares for x in triggered)
            highest_stop = max(x.stop_price for x in triggered)
            raw_stop = (min(float(row.open), highest_stop)
                        if stop_mode == 'intraday' else float(row.close))
            px = fill_price(raw_stop, 'sell', slippage_bps)
            value = qty * px
            fee = max(value * commission_rate, min_commission)
            p.cash += value - fee
            for lot in triggered:
                p.lots.remove(lot)
            append_trade(trades, today, today, 'STOP', 'STOP', px, qty, p,
                         float(row.close),
                         f'{stop_mode} 触发 {len(triggered)} 个独立批次保护位；手续费 {fee:.2f}')
            if any(x.source_type == '1B' for x in triggered):
                cooldown_remaining = cooldown_bars
            stopped_today = True

        # 固定左边界：开始日前 history_days 作为预热，随后历史只累积、不滚动删除。
        cache_key = str(today.date())
        if cache_key in signal_cache:
            data, mdf, buys, sells = signal_cache[cache_key]
        else:
            data, mdf, buys, sells = calculate_visible_signals(visible_now)
            signal_cache[cache_key] = (data, mdf, buys, sells)
        formal_buys = collect_formal_signals(buys, data, mdf)
        formal_sells = collect_formal_signals(sells, data, mdf)
        current_keys = {k for k, _ in formal_buys + formal_sells}

        if first_replay_day:
            # 预热窗口中早已存在的历史信号只作为状态，不在回测首日补交易；
            # 若信号本身日期正好是首日，则允许执行。
            new_buys = [s for k, s in formal_buys
                        if signal_date(s, data, mdf) == today]
            new_sells = [s for k, s in formal_sells
                         if signal_date(s, data, mdf) == today]
            first_replay_day = False
        else:
            # 分型/笔通常要未来几根已完成 K 才确认，因此交易“今天首次出现”的信号，
            # 而不是错误地要求信号标注日期必须等于今天。
            new_buys = [s for k, s in formal_buys if k not in seen_signal_keys]
            new_sells = [s for k, s in formal_sells if k not in seen_signal_keys]
        seen_signal_keys.update(current_keys)

        buy_sig = choose_new_signal(new_buys, 'buy')
        sell_sig = choose_new_signal(new_sells, 'sell')

        # 同日买卖冲突：正式 3 类优先；同类时卖出优先（风险优先）。
        sig = side = None
        if buy_sig and sell_sig:
            bp = int(buy_sig['type'][0]); sp = int(sell_sig['type'][0])
            sig, side = ((sell_sig, 'sell') if sp >= bp else (buy_sig, 'buy'))
        elif buy_sig:
            sig, side = buy_sig, 'buy'
        elif sell_sig:
            sig, side = sell_sig, 'sell'

        if sig is not None:
            annotated_dt = signal_date(sig, data, mdf)
            confirm_note = f'信号于 {today.date()} 首次确认'
            if execution == 'next_open':
                pending = (sig, side, annotated_dt)
            elif side == 'buy' and (sig['type'] != '1B' or cooldown_remaining <= 0):
                if sig['type'] in ('2B', '3B'):
                    cooldown_remaining = 0
                frac = buy_fractions[sig['type']] * weekly_buy_matrix[week_state][sig['type']]
                qty, px, fee = buy(p, frac, float(row.close), lot_size,
                                   commission_rate, min_commission, slippage_bps,
                                   sig['price'], sig['type'], today)
                if qty:
                    append_trade(trades, annotated_dt, today, 'BUY', sig['type'], px, qty,
                                 p, float(row.close),
                                 f'{confirm_note}；收盘成交；周线={week_state}；手续费 {fee:.2f}')
            elif side == 'sell':
                frac = sell_fractions[sig['type']]
                qty, px, fee = sell(p, frac, float(row.close), lot_size,
                                    commission_rate, min_commission, slippage_bps)
                if qty:
                    append_trade(trades, annotated_dt, today, 'SELL', sig['type'], px, qty,
                                 p, float(row.close),
                                 f'{confirm_note}；收盘成交；手续费 {fee:.2f}')

        equity_rows.append({
            'date': str(today.date()), 'close': float(row.close),
            'cash': p.cash, 'shares': p.shares,
            'position_value': p.shares * float(row.close),
            'equity': p.cash + p.shares * float(row.close),
            'stop_price': p.stop_price,
            'weekly_trend': week_state,
            'cooldown_remaining': cooldown_remaining,
        })
        if cooldown_remaining > 0 and not stopped_today:
            cooldown_remaining -= 1

    last_close = float(trade_rows.iloc[-1].close)
    final_equity = p.cash + p.shares * last_close
    return p, final_equity, pd.DataFrame([asdict(t) for t in trades]), pd.DataFrame(equity_rows)


def optimize_strategy(df, start, end, initial_cash, history_days, lot_size,
                      commission_rate, min_commission, slippage_bps, execution):
    """网格测试分级仓位与止损方式；共享信号缓存，避免重复计算结构。"""
    buy_profiles = [
        {'1B': 0.35, '2B': 0.25, '3B': 0.15},
        {'1B': 0.40, '2B': 0.30, '3B': 0.20},
        {'1B': 0.50, '2B': 0.40, '3B': 0.30},
        {'1B': 0.60, '2B': 0.40, '3B': 0.25},
    ]
    sell_profiles = [
        {'1S': 0.50, '2S': 0.50, '3S': 1.00},
        {'1S': 0.50, '2S': 0.75, '3S': 1.00},
    ]
    weekly_profiles = [
        {
            'bull': {'1B': 1.0, '2B': 1.0, '3B': 1.0},
            'neutral': {'1B': 0.4, '2B': 0.7, '3B': 1.0},
            'bear': {'1B': 0.1, '2B': 0.4, '3B': 0.7},
        },
        {
            'bull': {'1B': 1.0, '2B': 1.0, '3B': 1.0},
            'neutral': {'1B': 0.6, '2B': 0.8, '3B': 1.0},
            'bear': {'1B': 0.0, '2B': 0.5, '3B': 0.8},
        },
        {
            'bull': {'1B': 1.0, '2B': 1.0, '3B': 1.0},
            'neutral': {'1B': 0.8, '2B': 1.0, '3B': 1.0},
            'bear': {'1B': 0.15, '2B': 0.7, '3B': 1.0},
        },
        {
            'bull': {'1B': 1.0, '2B': 1.0, '3B': 1.0},
            'neutral': {'1B': 1.0, '2B': 1.0, '3B': 1.0},
            'bear': {'1B': 0.25, '2B': 1.0, '3B': 1.0},
        },
    ]
    cooldown_profiles = (0, 5, 10, 20)
    cache, rows, best = {}, [], None
    for bp in buy_profiles:
        for sp in sell_profiles:
            for stop_mode in ('intraday', 'close'):
                for wp in weekly_profiles:
                    for cooldown in cooldown_profiles:
                        _, final, trades, equity = run_backtest(
                            df, start, end, initial_cash, history_days,
                            bp, sp, lot_size, commission_rate, min_commission,
                            slippage_bps, execution, stop_mode, cache,
                            wp, cooldown)
                        ret = final / initial_cash - 1
                        dd = (float((equity.equity / equity.equity.cummax() - 1).min())
                              if len(equity) else 0.0)
                        # 跨行情评分：完整周期 + 前后半程，惩罚最差半程和最大回撤。
                        if len(equity) >= 4:
                            mid = len(equity) // 2
                            eq0 = float(equity.equity.iloc[0])
                            eqm = float(equity.equity.iloc[mid])
                            eqn = float(equity.equity.iloc[-1])
                            first_ret = eqm / eq0 - 1
                            second_ret = eqn / eqm - 1
                        else:
                            first_ret = second_ret = ret
                        worst_ret = min(first_ret, second_ret)
                        score = (0.50 * ret + 0.20 * first_ret +
                                 0.20 * second_ret + 0.10 * worst_ret -
                                 0.35 * abs(dd))
                        item = {'buy': bp, 'sell': sp, 'stop_mode': stop_mode,
                                'weekly': wp, 'cooldown': cooldown,
                                'final': final, 'return': ret,
                                'first_return': first_ret,
                                'second_return': second_ret,
                                'max_drawdown': dd,
                                'score': score, 'trades': len(trades)}
                        rows.append(item)
                        if best is None or item['score'] > best['score']:
                            best = item
    ranking = pd.DataFrame([{
        'buy_1B': x['buy']['1B'], 'buy_2B': x['buy']['2B'],
        'buy_3B': x['buy']['3B'], 'sell_1S': x['sell']['1S'],
        'sell_2S': x['sell']['2S'], 'sell_3S': x['sell']['3S'],
        'weekly_bull_1B': x['weekly']['bull']['1B'],
        'weekly_bull_2B': x['weekly']['bull']['2B'],
        'weekly_bull_3B': x['weekly']['bull']['3B'],
        'weekly_neutral_1B': x['weekly']['neutral']['1B'],
        'weekly_neutral_2B': x['weekly']['neutral']['2B'],
        'weekly_neutral_3B': x['weekly']['neutral']['3B'],
        'weekly_bear_1B': x['weekly']['bear']['1B'],
        'weekly_bear_2B': x['weekly']['bear']['2B'],
        'weekly_bear_3B': x['weekly']['bear']['3B'],
        'cooldown_bars': x['cooldown'],
        'stop_mode': x['stop_mode'], 'final_equity': x['final'],
        'return_pct': x['return'] * 100,
        'first_half_return_pct': x['first_return'] * 100,
        'second_half_return_pct': x['second_return'] * 100,
        'max_drawdown_pct': x['max_drawdown'] * 100,
        'score': x['score'], 'trades': x['trades']
    } for x in rows]).sort_values('score', ascending=False)
    return best, ranking, cache


def load_data(args):
    if args.source == 'txt':
        if not os.path.exists(args.txt):
            raise FileNotFoundError(args.txt)
        return core.load_from_txt(args.txt)
    # 只拉一次完整区间；逐日回放时仍严格切片，不把未来数据传给算法。
    data, _ = core.load_from_futu(args.stock, args.start, args.end,
                                  args.ktype, args.warmup)
    return data


def main():
    p = argparse.ArgumentParser(description='缠论逐日无未来函数模拟交易')
    p.add_argument('--source', choices=['txt', 'futu'], default='futu')
    p.add_argument('--txt', default='', help='source=txt 时的数据文件')
    p.add_argument('--stock', default='HK.800700')
    p.add_argument('--start', default='2025-10-01')
    p.add_argument('--end', default='2026-10-02')
    p.add_argument('--ktype', default='K_DAY',
                   choices=['K_DAY', 'K_60M', 'K_120M', 'K_240M', 'K_WEEK', 'K_MON'])
    p.add_argument('--warmup', type=int, default=180,
                   help='futu 拉取 start 之前的预热自然日数')
    p.add_argument('--history-days', type=int, default=180,
                   help='start 前预热自然日数；左边界固定，随后历史逐日累积')
    p.add_argument('--initial-cash', type=float, default=1000000.0)
    p.add_argument('--buy-1b', type=float, default=0.35)
    p.add_argument('--buy-2b', type=float, default=0.25)
    p.add_argument('--buy-3b', type=float, default=0.15)
    p.add_argument('--sell-1s', type=float, default=0.50)
    p.add_argument('--sell-2s', type=float, default=0.75)
    p.add_argument('--sell-3s', type=float, default=1.00)
    p.add_argument('--weekly-neutral-1b', type=float, default=1.00)
    p.add_argument('--weekly-neutral-2b', type=float, default=1.00)
    p.add_argument('--weekly-neutral-3b', type=float, default=1.00)
    p.add_argument('--weekly-bear-1b', type=float, default=0.25)
    p.add_argument('--weekly-bear-2b', type=float, default=1.00)
    p.add_argument('--weekly-bear-3b', type=float, default=1.00)
    p.add_argument('--cooldown-bars', type=int, default=0,
                   help='任一批次止损后暂停新买入的交易日数')
    p.add_argument('--optimize', action='store_true',
                   help='自动测试分级仓位、周线过滤、冷却期及两种止损口径')
    p.add_argument('--optimize-out', default='chan_backtest_optimization.csv')
    p.add_argument('--lot-size', type=int, default=1,
                   help='最小交易股数；港股请按标的手数设置')
    p.add_argument('--commission-rate', type=float, default=0.0003)
    p.add_argument('--min-commission', type=float, default=0.0)
    p.add_argument('--slippage-bps', type=float, default=0.0)
    p.add_argument('--execution', choices=['close', 'next_open'], default='close')
    p.add_argument('--stop-mode', choices=['intraday', 'close'], default='intraday',
                   help='intraday=low 跌破即止损；close=收盘跌破才止损')
    p.add_argument('--trades-out', default='chan_backtest_trades.csv')
    p.add_argument('--equity-out', default='chan_backtest_equity.csv')
    args = p.parse_args()

    ratio_names = ('buy_1b', 'buy_2b', 'buy_3b',
                   'sell_1s', 'sell_2s', 'sell_3s')
    for name in ratio_names:
        v = getattr(args, name)
        if not 0 < v <= 1:
            p.error(f'--{name.replace("_", "-")} 必须在 (0, 1]')
    if args.initial_cash <= 0 or args.history_days <= 0 or args.lot_size <= 0:
        p.error('资金、历史天数、lot-size 必须大于 0')
    if args.cooldown_bars < 0:
        p.error('--cooldown-bars 不能小于 0')
    for name in ('weekly_neutral_1b', 'weekly_neutral_2b', 'weekly_neutral_3b',
                 'weekly_bear_1b', 'weekly_bear_2b', 'weekly_bear_3b'):
        if not 0 <= getattr(args, name) <= 1:
            p.error(f'--{name.replace("_", "-")} 必须在 [0, 1]')

    df = load_data(args)
    buy_fractions = {'1B': args.buy_1b, '2B': args.buy_2b, '3B': args.buy_3b}
    sell_fractions = {'1S': args.sell_1s, '2S': args.sell_2s, '3S': args.sell_3s}
    weekly_buy_matrix = {
        'bull': {'1B': 1.0, '2B': 1.0, '3B': 1.0},
        'neutral': {'1B': args.weekly_neutral_1b,
                    '2B': args.weekly_neutral_2b,
                    '3B': args.weekly_neutral_3b},
        'bear': {'1B': args.weekly_bear_1b,
                 '2B': args.weekly_bear_2b,
                 '3B': args.weekly_bear_3b},
    }
    cooldown_bars = args.cooldown_bars
    signal_cache = {}
    if args.optimize:
        best, ranking, signal_cache = optimize_strategy(
            df, args.start, args.end, args.initial_cash, args.history_days,
            args.lot_size, args.commission_rate, args.min_commission,
            args.slippage_bps, args.execution)
        ranking.to_csv(args.optimize_out, index=False, encoding='utf-8-sig')
        buy_fractions, sell_fractions = best['buy'], best['sell']
        weekly_buy_matrix, cooldown_bars = best['weekly'], best['cooldown']
        args.stop_mode = best['stop_mode']
        print('========== 自动调参结果 ==========')
        print(f'买入比例: {buy_fractions}')
        print(f'卖出比例: {sell_fractions}')
        print(f'周线×买点倍率: {weekly_buy_matrix}')
        print(f'止损冷却: {cooldown_bars} 个交易日')
        print(f'止损方式: {args.stop_mode}')
        print(f'风险调整分数: {best["score"]:.4f}，收益率 {best["return"]*100:.2f}%，'
              f'最大回撤 {best["max_drawdown"]*100:.2f}%')
        print(f'完整排名: {args.optimize_out}')

    portfolio, final_equity, trades, equity = run_backtest(
        df, args.start, args.end, args.initial_cash, args.history_days,
        buy_fractions, sell_fractions, args.lot_size,
        args.commission_rate, args.min_commission, args.slippage_bps,
        args.execution, args.stop_mode, signal_cache,
        weekly_buy_matrix, cooldown_bars)
    trades.to_csv(args.trades_out, index=False, encoding='utf-8-sig')
    equity.to_csv(args.equity_out, index=False, encoding='utf-8-sig')

    ret = (final_equity / args.initial_cash - 1) * 100
    print('\n========== 回测结果 ==========')
    print(f'区间: {args.start} ~ {args.end}  历史窗口: {args.history_days} 自然日')
    print(f'初始资金: {args.initial_cash:.2f}')
    print(f'期末现金: {portfolio.cash:.2f}')
    print(f'期末持仓: {portfolio.shares} 股')
    print(f'期末总资产: {final_equity:.2f}  收益率: {ret:.2f}%')
    print(f'交易次数: {len(trades)}')

    print('\n========== 交易记录 ==========')
    if trades.empty:
        print('回测区间内没有发生交易。')
    else:
        action_zh = {'BUY': '买入', 'SELL': '卖出', 'STOP': '止损卖出'}
        for row in trades.itertuples(index=False):
            position_value = max(0.0, row.equity_after - row.cash_after)
            position_ratio = (position_value / row.equity_after * 100
                              if row.equity_after > 0 else 0.0)
            signal_text = f'（{row.signal}）' if row.signal != 'STOP' else ''
            print(
                f'{row.trade_date} {action_zh.get(row.action, row.action)}{signal_text} '
                f'{row.shares} 股，成交价 {row.price:.2f} 元，'
                f'成交金额 {row.shares * row.price:.2f} 元；'
                f'交易后现金 {row.cash_after:.2f} 元，持仓 {row.shares_after} 股，'
                f'仓位约 {position_ratio:.2f}%，总资产 {row.equity_after:.2f} 元'
            )

    print(f'\n交易明细: {args.trades_out}')
    print(f'每日权益: {args.equity_out}')


if __name__ == '__main__':
    main()
