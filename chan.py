# -*- coding: utf-8 -*-
"""
缠中说禅（缠论）技术分析完整实现
=====================================
基于原文核心章节：
  - 教你炒股票17：走势终完美
  - 教你炒股票20：走势中枢及第三类买卖点
  - 教你炒股票24-25：MACD 对背驰的辅助判断
  - 教你炒股票65-68：线段划分（特征序列 + 分型）

流程：
  1. 从 futu 拉日K（或从本地文本读取）
  2. K线包含关系处理
  3. 分型识别（顶分型 / 底分型）
  4. 笔（相邻顶底分型 + 中间至少 3 根独立K线）
  5. 线段（连续同向 3 笔的组合）
  6. 中枢（连续 3 段有重叠 → 区间 ZG=min(highs), ZD=max(lows)）
  7. 三类买卖点（一/二/三 买卖点 + 潜在观察位）
  8. matplotlib 可视化：K线 + 分型 + 笔 + 线段 + 中枢 + 买卖点 + MACD 副图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib as mpl
import re
import os

# 中文字体
for font in ['PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei', 'Microsoft YaHei',
             'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']:
    mpl.rcParams['font.sans-serif'] = [font] + mpl.rcParams['font.sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


# ================================================================
# 0. 数据读取
# ================================================================
def load_from_txt(path: str) -> pd.DataFrame:
    """从附件文本读取（空格分隔，首列是索引）"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header = re.split(r'\s+', lines[0].strip())
    rows = []
    for ln in lines[1:]:
        toks = re.split(r'\s+', ln.strip())
        if len(toks) < 3:
            continue
        time_key = toks[1] + ' ' + toks[2]
        rest = toks[3:]
        rows.append([time_key] + rest)
    cols = ['time_key'] + header[1:]
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        if c == 'time_key':
            df[c] = pd.to_datetime(df[c])
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.reset_index(drop=True)
    return df


def load_from_futu(stock_code='HK.800700', start='2025-10-02', end='2026-11-25'):
    """从 futu 拉数据（可选，需要 OpenD 已启动）"""
    from futu import OpenQuoteContext, KLType, RET_OK
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data, _ = quote_ctx.request_history_kline(stock_code, start=start, end=end,
                                                    ktype=KLType.K_DAY)
    quote_ctx.close()
    if ret != RET_OK:
        raise RuntimeError(f"futu 拉取失败: {data}")
    data = data.sort_values('time_key').reset_index(drop=True)
    data['time_key'] = pd.to_datetime(data['time_key'])
    return data


# ================================================================
# 1. K 线包含关系处理
# ================================================================
def process_inclusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    合并包含K线（原文第 63-64 讲）
      向上（前两根走势上升）：取 max(H), max(L)
      向下：取 min(H), min(L)
    """
    if len(df) == 0:
        return df.copy()
    highs = df['high'].values
    lows = df['low'].values
    times = df['time_key'].values
    opens = df['open'].values
    closes = df['close'].values

    merged_h = [highs[0]]
    merged_l = [lows[0]]
    merged_t = [times[0]]
    merged_o = [opens[0]]
    merged_c = [closes[0]]
    merged_orig_idx = [0]
    direction = 0

    for i in range(1, len(df)):
        h, l = highs[i], lows[i]
        ph, pl = merged_h[-1], merged_l[-1]
        contained = (h <= ph and l >= pl) or (h >= ph and l <= pl)
        if contained:
            if len(merged_h) >= 2:
                if merged_h[-1] > merged_h[-2] and merged_l[-1] > merged_l[-2]:
                    direction = 1
                elif merged_h[-1] < merged_h[-2] and merged_l[-1] < merged_l[-2]:
                    direction = -1
            if direction == 1:
                merged_h[-1] = max(ph, h)
                merged_l[-1] = max(pl, l)
            elif direction == -1:
                merged_h[-1] = min(ph, h)
                merged_l[-1] = min(pl, l)
            else:
                merged_h[-1] = max(ph, h)
                merged_l[-1] = max(pl, l)
            merged_t[-1] = times[i]
            merged_c[-1] = closes[i]
            merged_orig_idx[-1] = i
        else:
            merged_h.append(h)
            merged_l.append(l)
            merged_t.append(times[i])
            merged_o.append(opens[i])
            merged_c.append(closes[i])
            merged_orig_idx.append(i)
            if h > ph and l > pl:
                direction = 1
            elif h < ph and l < pl:
                direction = -1

    return pd.DataFrame({
        'time_key': merged_t, 'open': merged_o,
        'high': merged_h, 'low': merged_l, 'close': merged_c,
        'orig_idx': merged_orig_idx,
    })


# ================================================================
# 2. 分型识别
# ================================================================
def find_fractals(mdf: pd.DataFrame):
    """顶分型：中间K的 high 和 low 均高于两侧；底分型反之"""
    fractals = []
    hs = mdf['high'].values
    ls = mdf['low'].values
    for i in range(1, len(mdf) - 1):
        if hs[i] > hs[i-1] and hs[i] > hs[i+1] and ls[i] > ls[i-1] and ls[i] > ls[i+1]:
            fractals.append((i, 'top', hs[i]))
        elif ls[i] < ls[i-1] and ls[i] < ls[i+1] and hs[i] < hs[i-1] and hs[i] < hs[i+1]:
            fractals.append((i, 'bot', ls[i]))
    return fractals


# ================================================================
# 3. 笔的划分
# ================================================================
def build_strokes(fractals, mdf: pd.DataFrame, min_k_between=3):
    """
    笔：严格顶底交替；同向留极端；异向需 idx 距离 >= min_k_between 且价位合理
    """
    if not fractals:
        return []
    endpoints = []
    for f in fractals:
        if not endpoints:
            endpoints.append(f)
            continue
        last = endpoints[-1]
        if f[1] == last[1]:
            if (f[1] == 'top' and f[2] > last[2]) or \
               (f[1] == 'bot' and f[2] < last[2]):
                endpoints[-1] = f
            continue
        price_ok = (last[1] == 'bot' and f[1] == 'top' and f[2] > last[2]) or \
                   (last[1] == 'top' and f[1] == 'bot' and f[2] < last[2])
        if not price_ok:
            continue
        if f[0] - last[0] < min_k_between:
            continue
        endpoints.append(f)

    strokes = []
    for i in range(len(endpoints) - 1):
        a = endpoints[i]; b = endpoints[i+1]
        direction = 'up' if (a[1] == 'bot' and b[1] == 'top') else 'down'
        strokes.append({
            'start_idx': a[0], 'end_idx': b[0],
            'start_price': a[2], 'end_price': b[2],
            'direction': direction,
        })
    return strokes


# ================================================================
# 4. 线段划分
# ================================================================
def build_segments(strokes):
    """线段：同向笔连续延伸；反向笔终点越过前一同向笔起点则线段终结"""
    if len(strokes) < 3:
        return []
    segments = []
    n = len(strokes)
    seg_dir = strokes[0]['direction']
    seg_start_stroke = 0
    seg_start_price = strokes[0]['start_price']
    seg_end_stroke = 0
    seg_extreme = strokes[0]['end_price']

    def extend(stroke, seg_dir, seg_extreme):
        if stroke['direction'] != seg_dir:
            return False, seg_extreme
        if seg_dir == 'up' and stroke['end_price'] > seg_extreme:
            return True, stroke['end_price']
        if seg_dir == 'down' and stroke['end_price'] < seg_extreme:
            return True, stroke['end_price']
        return False, seg_extreme

    k = 1
    while k < n:
        s = strokes[k]
        if s['direction'] == seg_dir:
            ok, seg_extreme = extend(s, seg_dir, seg_extreme)
            if ok:
                seg_end_stroke = k
            k += 1
        else:
            reversed_end = s['end_price']
            prev_same_start = None
            for j in range(k-1, -1, -1):
                if strokes[j]['direction'] == seg_dir:
                    prev_same_start = strokes[j]['start_price']
                    break
            broken = False
            if prev_same_start is not None:
                if seg_dir == 'up' and reversed_end < prev_same_start:
                    broken = True
                if seg_dir == 'down' and reversed_end > prev_same_start:
                    broken = True
            if broken and (seg_end_stroke - seg_start_stroke + 1) >= 3:
                segments.append({
                    'start_stroke': seg_start_stroke,
                    'end_stroke': seg_end_stroke,
                    'start_price': seg_start_price,
                    'end_price': seg_extreme,
                    'direction': seg_dir,
                })
                seg_dir = s['direction']
                seg_start_stroke = k
                seg_start_price = s['start_price']
                seg_end_stroke = k
                seg_extreme = s['end_price']
                k += 1
            else:
                k += 1

    if seg_end_stroke > seg_start_stroke or (n - seg_start_stroke) >= 1:
        segments.append({
            'start_stroke': seg_start_stroke,
            'end_stroke': seg_end_stroke,
            'start_price': seg_start_price,
            'end_price': seg_extreme,
            'direction': seg_dir,
        })
    return segments


# ================================================================
# 5. 中枢识别
# ================================================================
def build_pivots(strokes):
    """
    中枢：A-B-C 三段方向交替，重叠区间 [ZD, ZG]。
    延伸：后续笔与 [ZD,ZG] 有交集就纳入；某笔终点跌破/涨破边界则中枢结束。
    """
    pivots = []
    n = len(strokes)

    def rng(s):
        return (min(s['start_price'], s['end_price']),
                max(s['start_price'], s['end_price']))

    i = 1
    while i + 2 < n:
        A, B, C = strokes[i], strokes[i+1], strokes[i+2]
        if not (A['direction'] != B['direction'] and B['direction'] != C['direction']):
            i += 1
            continue
        lA, hA = rng(A); lB, hB = rng(B); lC, hC = rng(C)
        ZG = min(hA, hB, hC)
        ZD = max(lA, lB, lC)
        if ZG <= ZD:
            i += 1
            continue

        start_stroke = i
        end_stroke = i + 2
        highs = [hA, hB, hC]; lows = [lA, lB, lC]
        j = i + 3
        while j < n:
            sj = strokes[j]
            ep = sj['end_price']
            if ep < ZD or ep > ZG:
                break
            lj, hj = rng(sj)
            if hj >= ZD and lj <= ZG:
                end_stroke = j
                highs.append(hj); lows.append(lj)
                j += 1
            else:
                break
        pivots.append({
            'start_stroke': start_stroke, 'end_stroke': end_stroke,
            'ZG': ZG, 'ZD': ZD,
            'GG': max(highs), 'DD': min(lows),
            'enter_stroke': i - 1,
            'enter_direction': strokes[i-1]['direction'],
        })
        i = end_stroke + 1
    return pivots


# ================================================================
# 6. 买卖点识别
# ================================================================
def macd_area(macdh, i0, i1):
    seg = macdh[i0:i1+1]
    seg = seg[~np.isnan(seg)]
    if len(seg) == 0:
        return 0.0, 0.0
    pos = seg[seg > 0].sum()
    neg = -seg[seg < 0].sum()
    return float(pos), float(neg)


def find_buy_sell_points(strokes, pivots, mdf: pd.DataFrame, orig_df: pd.DataFrame):
    """
    一买 1B：下跌同向两笔中，后一笔创新低但 MACD面积/斜率/幅度衰减（背驰）
    一卖 1S：上涨同向两笔中，后一笔创新高但衰减
    二买 2B：一买后回抽（下跌笔）终点不破一买价位
    二卖 2S：一卖后反抽（上涨笔）终点不破一卖价位
    三买 3B：中枢向上突破，回抽笔终点仍在 ZG 之上
    三卖 3S：中枢向下跌破，反抽笔终点仍在 ZD 之下
    3B?/3S?：已破位但回抽笔尚未走出的观察位
    """
    buys, sells = [], []
    macdh = orig_df['MACDh_12_26_9'].values
    orig_idx_map = mdf['orig_idx'].values

    # 一买 / 一卖：三选二背驰
    for i in range(2, len(strokes)):
        a = strokes[i-2]; c = strokes[i]
        if a['direction'] != c['direction']:
            continue
        a0, a1 = orig_idx_map[a['start_idx']], orig_idx_map[a['end_idx']]
        c0, c1 = orig_idx_map[c['start_idx']], orig_idx_map[c['end_idx']]
        pa_pos, pa_neg = macd_area(macdh, a0, a1)
        pc_pos, pc_neg = macd_area(macdh, c0, c1)
        a_range = abs(a['end_price'] - a['start_price'])
        c_range = abs(c['end_price'] - c['start_price'])
        macd_valid = (pa_pos + pa_neg) > 5 and (pc_pos + pc_neg) > 5
        slope_a = a_range / max(1, a1 - a0)
        slope_c = c_range / max(1, c1 - c0)

        if c['direction'] == 'down':
            new_low = c['end_price'] < a['end_price']
            hits = sum([macd_valid and pc_neg < pa_neg * 0.8,
                        slope_c < slope_a * 0.8,
                        c_range < a_range * 0.7])
            if new_low and hits >= 2:
                buys.append({'type': '1B', 'idx': c['end_idx'],
                             'price': c['end_price'],
                             'note': f'底背驰(命中{hits}/3)', 'stroke_i': i})
        else:
            new_high = c['end_price'] > a['end_price']
            hits = sum([macd_valid and pc_pos < pa_pos * 0.8,
                        slope_c < slope_a * 0.8,
                        c_range < a_range * 0.7])
            if new_high and hits >= 2:
                sells.append({'type': '1S', 'idx': c['end_idx'],
                              'price': c['end_price'],
                              'note': f'顶背驰(命中{hits}/3)', 'stroke_i': i})

    # 盘整背驰补充（中枢内同向笔比较）
    for pv in pivots:
        same_dir_strokes = [(k, strokes[k]) for k in
                            range(pv['start_stroke'], pv['end_stroke']+1)]
        for direction in ['down', 'up']:
            arr = [(k, s) for k, s in same_dir_strokes if s['direction'] == direction]
            for j in range(1, len(arr)):
                _, a = arr[j-1]; k_c, c = arr[j]
                a0, a1 = orig_idx_map[a['start_idx']], orig_idx_map[a['end_idx']]
                c0, c1 = orig_idx_map[c['start_idx']], orig_idx_map[c['end_idx']]
                pa_pos, pa_neg = macd_area(macdh, a0, a1)
                pc_pos, pc_neg = macd_area(macdh, c0, c1)
                if direction == 'down':
                    if c['end_price'] <= a['end_price'] and pa_neg > 0 and pc_neg < pa_neg * 0.9:
                        if not any(b['idx'] == c['end_idx'] for b in buys):
                            buys.append({'type': '1B', 'idx': c['end_idx'],
                                         'price': c['end_price'],
                                         'note': '中枢内盘整底背驰', 'stroke_i': k_c})
                else:
                    if c['end_price'] >= a['end_price'] and pa_pos > 0 and pc_pos < pa_pos * 0.9:
                        if not any(s['idx'] == c['end_idx'] for s in sells):
                            sells.append({'type': '1S', 'idx': c['end_idx'],
                                          'price': c['end_price'],
                                          'note': '中枢内盘整顶背驰', 'stroke_i': k_c})

    # 二买 / 二卖
    for b in list(buys):
        si = b['stroke_i']
        if si + 2 < len(strokes):
            n1 = strokes[si + 1]; n2 = strokes[si + 2]
            if n1['direction'] == 'up' and n2['direction'] == 'down' and n2['end_price'] > b['price']:
                buys.append({'type': '2B', 'idx': n2['end_idx'],
                             'price': n2['end_price'],
                             'note': f'一买后回抽不破 {b["price"]:.0f}',
                             'stroke_i': si + 2})
    for s in list(sells):
        si = s['stroke_i']
        if si + 2 < len(strokes):
            n1 = strokes[si + 1]; n2 = strokes[si + 2]
            if n1['direction'] == 'down' and n2['direction'] == 'up' and n2['end_price'] < s['price']:
                sells.append({'type': '2S', 'idx': n2['end_idx'],
                              'price': n2['end_price'],
                              'note': f'一卖后反抽不破 {s["price"]:.0f}',
                              'stroke_i': si + 2})

    # 三买 / 三卖
    for pv in pivots:
        ZG, ZD = pv['ZG'], pv['ZD']
        end_i = pv['end_stroke']
        last_in_pivot = strokes[end_i]
        broke_down = last_in_pivot['direction'] == 'down' and last_in_pivot['end_price'] < ZD
        broke_up = last_in_pivot['direction'] == 'up' and last_in_pivot['end_price'] > ZG

        if broke_up:
            if end_i + 1 < len(strokes):
                pull = strokes[end_i + 1]
                if pull['direction'] == 'down' and pull['end_price'] > ZG:
                    buys.append({'type': '3B', 'idx': pull['end_idx'],
                                 'price': pull['end_price'],
                                 'note': f'突破中枢[{ZD:.0f},{ZG:.0f}]回抽不回',
                                 'stroke_i': end_i + 1})
            else:
                buys.append({'type': '3B?', 'idx': last_in_pivot['end_idx'],
                             'price': last_in_pivot['end_price'],
                             'note': f'突破中枢，等回抽{ZG:.0f}上方',
                             'stroke_i': end_i})
        elif broke_down:
            if end_i + 1 < len(strokes):
                pull = strokes[end_i + 1]
                if pull['direction'] == 'up' and pull['end_price'] < ZD:
                    sells.append({'type': '3S', 'idx': pull['end_idx'],
                                  'price': pull['end_price'],
                                  'note': f'跌破中枢[{ZD:.0f},{ZG:.0f}]反抽不回',
                                  'stroke_i': end_i + 1})
            else:
                sells.append({'type': '3S?', 'idx': last_in_pivot['end_idx'],
                              'price': last_in_pivot['end_price'],
                              'note': f'跌破中枢，等反抽{ZD:.0f}下方',
                              'stroke_i': end_i})
        else:
            if end_i + 1 >= len(strokes):
                continue
            leave = strokes[end_i + 1]
            if end_i + 2 < len(strokes):
                pull = strokes[end_i + 2]
                if leave['direction'] == 'up' and leave['end_price'] > ZG:
                    if pull['direction'] == 'down' and pull['end_price'] > ZG:
                        buys.append({'type': '3B', 'idx': pull['end_idx'],
                                     'price': pull['end_price'],
                                     'note': f'突破中枢[{ZD:.0f},{ZG:.0f}]回抽不回',
                                     'stroke_i': end_i + 2})
                elif leave['direction'] == 'down' and leave['end_price'] < ZD:
                    if pull['direction'] == 'up' and pull['end_price'] < ZD:
                        sells.append({'type': '3S', 'idx': pull['end_idx'],
                                      'price': pull['end_price'],
                                      'note': f'跌破中枢[{ZD:.0f},{ZG:.0f}]反抽不回',
                                      'stroke_i': end_i + 2})
            else:
                if leave['direction'] == 'up' and leave['end_price'] > ZG:
                    buys.append({'type': '3B?', 'idx': leave['end_idx'],
                                 'price': leave['end_price'],
                                 'note': f'突破中枢，等回抽{ZG:.0f}上方',
                                 'stroke_i': end_i + 1})
                elif leave['direction'] == 'down' and leave['end_price'] < ZD:
                    sells.append({'type': '3S?', 'idx': leave['end_idx'],
                                  'price': leave['end_price'],
                                  'note': f'跌破中枢，等反抽{ZD:.0f}下方',
                                  'stroke_i': end_i + 1})
    return buys, sells


# ================================================================
# 7. 可视化
# ================================================================
def plot_chan(orig_df, mdf, fractals, strokes, segments, pivots, buys, sells,
              title='恒生科技指数 HK.800700  缠论分析'):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.05], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1], sharex=ax)

    n = len(orig_df)
    x = np.arange(n)
    dates = orig_df['time_key'].dt.strftime('%m-%d').values

    # K线
    for i in range(n):
        o, c, h, l = orig_df.loc[i, ['open', 'close', 'high', 'low']]
        color = '#e74c3c' if c >= o else '#2ecc71'
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=1)
        rect = Rectangle((i - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-3),
                         facecolor=color, edgecolor=color, zorder=2)
        ax.add_patch(rect)

    # 分型
    orig_idx_map = mdf['orig_idx'].values
    for fi, ftype, fp in fractals:
        xi = orig_idx_map[fi]
        if ftype == 'top':
            ax.annotate('', xy=(xi, fp * 1.005), xytext=(xi, fp * 1.02),
                        arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1))
        else:
            ax.annotate('', xy=(xi, fp * 0.995), xytext=(xi, fp * 0.98),
                        arrowprops=dict(arrowstyle='->', color='#3498db', lw=1))

    # 笔
    for s in strokes:
        x0 = orig_idx_map[s['start_idx']]
        x1 = orig_idx_map[s['end_idx']]
        ax.plot([x0, x1], [s['start_price'], s['end_price']],
                color='#34495e', lw=1.4, zorder=3)

    # 线段
    for seg in segments:
        s0 = strokes[seg['start_stroke']]
        s1 = strokes[seg['end_stroke']]
        x0 = orig_idx_map[s0['start_idx']]
        x1 = orig_idx_map[s1['end_idx']]
        color = '#c0392b' if seg['direction'] == 'up' else '#27ae60'
        ax.plot([x0, x1], [seg['start_price'], seg['end_price']],
                color=color, lw=2.5, alpha=0.55, zorder=4)

    # 中枢
    for pv in pivots:
        s_start = strokes[pv['start_stroke']]
        s_end = strokes[pv['end_stroke']]
        x0 = orig_idx_map[s_start['start_idx']]
        x1 = orig_idx_map[s_end['end_idx']]
        rect = Rectangle((x0, pv['ZD']), x1 - x0, pv['ZG'] - pv['ZD'],
                         facecolor='#f39c12', edgecolor='#d35400',
                         alpha=0.18, lw=1.2, zorder=1.5)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, pv['ZG'], f"中枢[{pv['ZD']:.0f},{pv['ZG']:.0f}]",
                ha='center', va='bottom', fontsize=8, color='#7f4a00')

    # 买卖点
    marker_buy = {'1B': ('^', '#c0392b', 90), '2B': ('^', '#e74c3c', 70),
                  '3B': ('^', '#f39c12', 70), '3B?': ('*', '#f39c12', 130)}
    marker_sell = {'1S': ('v', '#16a085', 90), '2S': ('v', '#27ae60', 70),
                   '3S': ('v', '#2ecc71', 70), '3S?': ('*', '#2ecc71', 130)}
    for b in buys:
        xi = orig_idx_map[b['idx']]
        m, cc, sz = marker_buy[b['type']]
        ax.scatter([xi], [b['price']], marker=m, s=sz, color=cc,
                   edgecolors='black', linewidths=0.6, zorder=6)
        ax.annotate(b['type'], (xi, b['price']), xytext=(0, -14),
                    textcoords='offset points', ha='center',
                    fontsize=8, color=cc, fontweight='bold')
    for s in sells:
        xi = orig_idx_map[s['idx']]
        m, cc, sz = marker_sell[s['type']]
        ax.scatter([xi], [s['price']], marker=m, s=sz, color=cc,
                   edgecolors='black', linewidths=0.6, zorder=6)
        ax.annotate(s['type'], (xi, s['price']), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=8, color=cc, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('价格')
    ax.grid(True, alpha=0.25)

    # 当下走势解读
    if strokes:
        last_stroke = strokes[-1]
        last_pivot = pivots[-1] if pivots else None
        text_lines = ['【当下走势解读】',
                      f'最新一笔: {last_stroke["direction"]} '
                      f'{last_stroke["start_price"]:.0f} → {last_stroke["end_price"]:.0f}']
        if last_pivot:
            zd, zg = last_pivot['ZD'], last_pivot['ZG']
            last_p = last_stroke['end_price']
            text_lines.append(f'最新中枢: [{zd:.0f}, {zg:.0f}]')
            if last_p < zd:
                text_lines.append(f'当前价 {last_p:.0f} 已跌破中枢下沿 → 关注反抽止涨（潜在三卖）')
            elif last_p > zg:
                text_lines.append(f'当前价 {last_p:.0f} 已突破中枢上沿 → 关注回抽不入（潜在三买）')
            else:
                text_lines.append(f'当前价 {last_p:.0f} 仍在中枢内 → 震荡观察')
        ax.text(0.99, 0.02, '\n'.join(text_lines), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffbe6',
                          edgecolor='#d4a017', alpha=0.9))

    # 图例
    legend_elems = [
        Line2D([0], [0], color='#34495e', lw=1.4, label='笔'),
        Line2D([0], [0], color='#c0392b', lw=2.5, alpha=0.55, label='线段(上)'),
        Line2D([0], [0], color='#27ae60', lw=2.5, alpha=0.55, label='线段(下)'),
        Rectangle((0, 0), 1, 1, facecolor='#f39c12', alpha=0.3, label='中枢'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#c0392b', markersize=10, label='1B 一买'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#e74c3c', markersize=9, label='2B 二买'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#f39c12', markersize=9, label='3B 三买'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#16a085', markersize=10, label='1S 一卖'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#27ae60', markersize=9, label='2S 二卖'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#2ecc71', markersize=9, label='3S 三卖'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', ncol=2, fontsize=9)

    # MACD 副图
    macdh = orig_df['MACDh_12_26_9'].values
    colors = ['#e74c3c' if v >= 0 else '#2ecc71' for v in np.nan_to_num(macdh)]
    axm.bar(x, macdh, color=colors, width=0.8)
    axm.axhline(0, color='black', lw=0.5)
    axm.set_ylabel('MACDh (12,26,9)')
    axm.grid(True, alpha=0.25)

    step = max(1, n // 15)
    axm.set_xticks(x[::step])
    axm.set_xticklabels(dates[::step], rotation=30, ha='right', fontsize=9)
    ax.set_xlim(-1, n)

    plt.tight_layout()
    return fig


# ================================================================
# 主流程
# ================================================================
def main():
    txt_path = '长文本-1790181607.txt'
    if os.path.exists(txt_path):
        orig_df = load_from_txt(txt_path)
        print(f"[加载] 从本地文本读取 {len(orig_df)} 根K线")
    else:
        orig_df = load_from_futu()
        print(f"[加载] 从 futu 拉取 {len(orig_df)} 根K线")

    # 若无 MACDh 列，自行计算
    if 'MACDh_12_26_9' not in orig_df.columns or orig_df['MACDh_12_26_9'].isna().all():
        ema12 = orig_df['close'].ewm(span=12, adjust=False).mean()
        ema26 = orig_df['close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        orig_df['MACDh_12_26_9'] = (dif - dea) * 2

    mdf = process_inclusion(orig_df)
    print(f"[包含处理] {len(orig_df)} -> {len(mdf)} 根合并K")

    fractals = find_fractals(mdf)
    print(f"[分型] 共 {len(fractals)} 个")

    strokes = build_strokes(fractals, mdf, min_k_between=3)
    print(f"[笔] 共 {len(strokes)} 笔")
    for i, s in enumerate(strokes):
        st = orig_df.loc[mdf['orig_idx'].values[s['start_idx']], 'time_key'].strftime('%m-%d')
        et = orig_df.loc[mdf['orig_idx'].values[s['end_idx']], 'time_key'].strftime('%m-%d')
        print(f"  笔{i:2d} {s['direction']:4s} {st}({s['start_price']:.2f}) -> {et}({s['end_price']:.2f})")

    segments = build_segments(strokes)
    print(f"[线段] 共 {len(segments)} 段")
    for i, seg in enumerate(segments):
        print(f"  段{i} {seg['direction']:4s} 笔{seg['start_stroke']}~{seg['end_stroke']}  "
              f"{seg['start_price']:.2f} -> {seg['end_price']:.2f}")

    pivots = build_pivots(strokes)
    print(f"[中枢] 共 {len(pivots)} 个")
    for i, pv in enumerate(pivots):
        print(f"  中枢{i} 笔{pv['start_stroke']}~{pv['end_stroke']}  "
              f"ZD={pv['ZD']:.2f} ZG={pv['ZG']:.2f}")

    buys, sells = find_buy_sell_points(strokes, pivots, mdf, orig_df)
    print(f"[买卖点] 买 {len(buys)}  卖 {len(sells)}")
    for b in buys:
        d = orig_df.loc[mdf['orig_idx'].values[b['idx']], 'time_key'].strftime('%Y-%m-%d')
        print(f"  {b['type']} {d} @ {b['price']:.2f}  {b['note']}")
    for s in sells:
        d = orig_df.loc[mdf['orig_idx'].values[s['idx']], 'time_key'].strftime('%Y-%m-%d')
        print(f"  {s['type']} {d} @ {s['price']:.2f}  {s['note']}")

    fig = plot_chan(orig_df, mdf, fractals, strokes, segments, pivots, buys, sells)
    fig.savefig('chan_result.png', dpi=140, bbox_inches='tight')
    print("[保存] chan_result.png")
    plt.show()


if __name__ == '__main__':
    main()