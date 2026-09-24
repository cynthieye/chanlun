# -*- coding: utf-8 -*-
"""
缠中说禅（缠论）技术分析完整实现
=====================================
基于原文核心章节：
  - 教你炒股票17：走势终完美（走势必然完成 盘整/上涨/下跌 三种走势类型之一）
  - 教你炒股票20：走势中枢及第三类买卖点
  - 教你炒股票24-25：MACD 对背驰的辅助判断
  - 教你炒股票65-68：线段划分（特征序列 + 分型）

流程：
  1. 从 futu 拉日K（或从本地文本读取）
  2. K线包含关系处理（去除包含K线）
  3. 分型识别（顶分型 / 底分型）
  4. 笔（相邻顶底分型 + 中间至少 4 根独立K线）
  5. 线段（特征序列 + 特征序列分型）
  6. 中枢（连续 3 段有重叠 → 中枢区间 ZG=min(gaos), ZD=max(dis)）
  7. 三类买卖点：
     - 一买：下跌走势中枢向下延伸后出现的底背驰（MACD 面积 + 力度）
     - 二买：一买之后的次级别回抽不破一买低点
     - 三买：中枢突破后回抽不回中枢的买点（离开中枢的中枢结束确认）
     卖点对称
  8. matplotlib 可视化：K线 + 分型 + 笔 + 中枢矩形 + 买卖点标注 + MACD 副图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        # 首列是行号，第二三列组合成 time_key
        # 格式: 0 2026-06-02 00:00:00 4995.83 ...
        idx = toks[0]
        time_key = toks[1] + ' ' + toks[2]
        rest = toks[3:]
        rows.append([time_key] + rest)
    cols = ['time_key'] + header[1:]  # 去掉 time_key 那一列后的头
    # header[1:] 从 'open' 开始
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        if c == 'time_key':
            df[c] = pd.to_datetime(df[c])
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.reset_index(drop=True)
    return df

def load_from_futu(stock_code='HK.00700', start='2026-03-11', end='2026-11-25'):
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
    包含定义：K2 完全被 K1 包含（H2<=H1 且 L2>=L1）或相反
    合并规则：
      向上（前两根走势上升）：取 max(H), max(L)
      向下：取 min(H), min(L)
    合并后保留 time_key（用第二根的时间）
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
    # 原始 idx 映射（每根合并K对应的原始最后一根 idx）
    merged_orig_idx = [0]

    direction = 0  # 1 up, -1 down, 0 unknown

    for i in range(1, len(df)):
        h, l = highs[i], lows[i]
        ph, pl = merged_h[-1], merged_l[-1]

        # 检查包含
        contained = (h <= ph and l >= pl) or (h >= ph and l <= pl)
        if contained:
            # 判方向：以合并K前一根为准
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
                # 首次未知方向：向上处理（保留大值）
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

    out = pd.DataFrame({
        'time_key': merged_t,
        'open': merged_o,
        'high': merged_h,
        'low': merged_l,
        'close': merged_c,
        'orig_idx': merged_orig_idx,
    })
    return out


# ================================================================
# 2. 分型识别
# ================================================================
def find_fractals(mdf: pd.DataFrame):
    """
    顶分型：三根相邻K，中间K的 high 最高、low 也最高
    底分型：三根相邻K，中间K的 low 最低、high 也最低
    返回列表 [(idx_in_mdf, 'top'/'bot', price)]
    """
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
    笔（原文教你炒股票 63-64）：
      两相邻分型（一顶一底）间的合并K线数（含端点）应 >= 5，即 idx 差 >= 4；
      为适配日线小样本，此处默认 min_k_between=3。

    严格贪心算法（一遍扫描 + 局部回退修正）：
      维护 endpoints 数组，严格顶底交替。对每个新分型 f：
        - 若 endpoints 为空 → 入。
        - 若 f 与 endpoints[-1] 同向：留更极端者（滑动更新）。
        - 若 f 与 endpoints[-1] 异向：
            * 检查价位合理（顶>前一底 或 底<前一顶）。
            * 若 idx 距离 >= min_k_between → 追加。
            * 否则丢弃 f（距离不足）。
      追加后可能出现"新锚点被后续更极端同向覆盖"，try_append 的同向分支已处理。
      不做多余的三点回退。
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
        # 异向：价位合理性
        price_ok = (last[1] == 'bot' and f[1] == 'top' and f[2] > last[2]) or \
                   (last[1] == 'top' and f[1] == 'bot' and f[2] < last[2])
        if not price_ok:
            continue
        # 距离
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
# 4. 线段划分（简化版：特征序列分型法）
# ================================================================
def build_segments(strokes):
    """
    线段 = 至少 3 笔且方向一致的组合。
    简化实现：以"特征序列"的顶底分型作为线段端点。
      - 上涨线段的特征序列 = 所有向下的笔
      - 下跌线段的特征序列 = 所有向上的笔
    这里给出一个稳健的近似实现：
      从第一笔开始尝试起线段，若后续同向笔创新高/低则延续，
      当反向笔破坏了前一同向笔的极值且形成对应分型，视为线段终结。
    """
    if len(strokes) < 3:
        return []

    segments = []
    i = 0
    n = len(strokes)
    # 起始方向 = 第一笔方向
    seg_dir = strokes[0]['direction']
    seg_start_stroke = 0
    seg_start_price = strokes[0]['start_price']
    seg_end_stroke = 0
    seg_extreme = strokes[0]['end_price']  # 目前线段末端极值

    def extend(stroke, seg_dir, seg_extreme):
        """同向笔是否延续线段极值"""
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
            # 反向笔：判断是否终结线段
            # 简化规则：连续两笔反向且第二反向笔的终点破坏了 seg_extreme 前的关键位
            # 更稳的判定：反向笔的 end_price 是否越过前一同向笔的起点
            # 即：上涨线段中，反向下笔的终点 < 上一上涨笔的起点 → 终结
            reversed_end = s['end_price']
            # 找上一同向笔
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
                # 结束当前线段（终点 = 上一同向笔的终点，即 seg_extreme）
                segments.append({
                    'start_stroke': seg_start_stroke,
                    'end_stroke': seg_end_stroke,
                    'start_price': seg_start_price,
                    'end_price': seg_extreme,
                    'direction': seg_dir,
                })
                # 新线段从反向笔开始
                seg_dir = s['direction']
                seg_start_stroke = k
                seg_start_price = s['start_price']
                seg_end_stroke = k
                seg_extreme = s['end_price']
                k += 1
            else:
                # 反向笔未终结线段（仅是同级别回撤，或不足 3 笔）
                k += 1

    # 收尾：把最后一段收入（用于当下分析）
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
    中枢（原文教你炒股票 20）：
      中枢由（次级别）走势类型 A-B-C 三段构成，且 A 与 C 同向、B 反向，
      三段的价格区间必须有共同的重叠区间：
        ZG = min(A的高, B的高, C的高)
        ZD = max(A的低, B的低, C的低)
      A 前面还有一段（第 0 段）称为"进入段"，即 strokes[i-1]，方向与 B 同、与 A/C 反。
    延伸：ABC 之后仍在 [ZD,ZG] 内震荡的每一笔纳入延伸；
          一旦某笔与 [ZD,ZG] 无交集 → 中枢结束，该笔即"离开笔"。
    """
    pivots = []
    n = len(strokes)

    def rng(s):
        return (min(s['start_price'], s['end_price']),
                max(s['start_price'], s['end_price']))

    # 从 i=1 开始（i-1 是进入段），A=strokes[i], B=strokes[i+1], C=strokes[i+2]
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
        # 延伸：只要后续笔与 [ZD,ZG] 有交集 就算中枢内
        # 但若某笔的终点已跌破 ZD 或涨破 ZG，视为"离开笔"，不再纳入中枢
        j = i + 3
        while j < n:
            sj = strokes[j]
            ep = sj['end_price']
            # 关键：若该笔终点已经离开中枢上下沿，则该笔是"离开笔"，中枢已结束
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
            'start_stroke': start_stroke,
            'end_stroke': end_stroke,
            'ZG': ZG, 'ZD': ZD,
            'GG': max(highs),
            'DD': min(lows),
            'enter_stroke': i - 1,
            'enter_direction': strokes[i-1]['direction'],
        })
        i = end_stroke + 1
    return pivots


# ================================================================
# 6. 买卖点识别
# ================================================================
def macd_area(macdh, i0, i1):
    """区间 [i0, i1] 上 MACDh 的正/负面积（绝对值和）"""
    seg = macdh[i0:i1+1]
    seg = seg[~np.isnan(seg)]
    if len(seg) == 0:
        return 0.0, 0.0
    pos = seg[seg > 0].sum()
    neg = -seg[seg < 0].sum()
    return float(pos), float(neg)


def build_sub_strokes(strokes, mdf: pd.DataFrame, orig_df: pd.DataFrame,
                       min_stroke_len=15, min_gap=2,
                       amp_ratio=0.10, amp_pct=0.015):
    """
    为"超长笔"提取子笔（次级别拆分），供绘图画细线用。
    - 只对原始 K 线跨度 >= min_stroke_len 的笔生效
    - 下跌笔内找 high 局部极大（次级反弹高点）
    - 上涨笔内找 low 局部极小（次级回撤低点）
    - 端点显著性过滤：反弹/回撤幅度 >= 该笔幅度 * amp_ratio 且 >= 起点价 * amp_pct
    - 相邻极值点至少间隔 min_gap 根 K
    返回：[[(orig_idx, price), ...], ...] 每个长笔一条子笔序列（含起点、内部极值、终点），
          与 strokes 顺序对齐；非长笔位置返回 None。
    """
    orig_idx_map = mdf['orig_idx'].values
    highs = orig_df['high'].values
    lows = orig_df['low'].values
    closes = orig_df['close'].values

    result = []
    for stroke in strokes:
        s0 = orig_idx_map[stroke['start_idx']]
        s1 = orig_idx_map[stroke['end_idx']]
        length = s1 - s0
        if length < min_stroke_len:
            result.append(None)
            continue

        start_price = float(closes[s0])
        end_price = float(closes[s1])
        stroke_range = abs(end_price - start_price)
        if stroke_range <= 0:
            result.append(None)
            continue

        # 起点用主笔起点价（与 strokes 保持一致）
        pts = [(int(s0), float(stroke['start_price']))]

        last_hit = -10
        # 交替方向扫描：下跌笔里 high↔low 交替，上涨笔里 low↔high 交替
        # 先按位置扫，收集所有显著端点
        candidates = []  # (orig_idx, price, kind)  kind: 'H' or 'L'
        for k in range(s0 + 3, s1 - 3):
            if k - last_hit < min_gap:
                continue
            win_l = max(s0, k - 3)
            win_r = min(s1, k + 4)
            if stroke['direction'] == 'down':
                # 反弹高点
                if highs[k] == highs[win_l:win_r].max() and highs[k] > highs[k-1] and highs[k] > highs[k+1]:
                    rally = highs[k] - min(lows[s0:k+1])
                    if rally >= stroke_range * amp_ratio and rally / start_price >= amp_pct:
                        candidates.append((int(k), float(highs[k]), 'H'))
                        last_hit = k
                        continue
                # 低点（子笔另一端）
                if lows[k] == lows[win_l:win_r].min() and lows[k] < lows[k-1] and lows[k] < lows[k+1]:
                    drop_from_prev_high = max(highs[s0:k+1]) - lows[k]
                    if drop_from_prev_high >= stroke_range * amp_ratio and drop_from_prev_high / start_price >= amp_pct:
                        candidates.append((int(k), float(lows[k]), 'L'))
                        last_hit = k
            else:
                # 上涨笔：回撤低点
                if lows[k] == lows[win_l:win_r].min() and lows[k] < lows[k-1] and lows[k] < lows[k+1]:
                    pullback = max(highs[s0:k+1]) - lows[k]
                    if pullback >= stroke_range * amp_ratio and pullback / start_price >= amp_pct:
                        candidates.append((int(k), float(lows[k]), 'L'))
                        last_hit = k
                        continue
                # 高点
                if highs[k] == highs[win_l:win_r].max() and highs[k] > highs[k-1] and highs[k] > highs[k+1]:
                    up_from_prev_low = highs[k] - min(lows[s0:k+1])
                    if up_from_prev_low >= stroke_range * amp_ratio and up_from_prev_low / start_price >= amp_pct:
                        candidates.append((int(k), float(highs[k]), 'H'))
                        last_hit = k

        # 保证子笔交替方向：主笔起点方向由主笔决定
        # 下跌笔：起点 H → 内部第一段 L → H → L …→ 终点 L
        # 上涨笔：起点 L → 内部第一段 H → L → H …→ 终点 H
        first_kind_wanted = 'L' if stroke['direction'] == 'down' else 'H'
        filtered = []
        expected = first_kind_wanted
        for c in candidates:
            if c[2] == expected:
                filtered.append(c)
                expected = 'H' if expected == 'L' else 'L'

        for c in filtered:
            pts.append((c[0], c[1]))

        # 终点用主笔终点价
        pts.append((int(s1), float(stroke['end_price'])))

        # 至少 2 个内部端点才算有意义的子笔序列，否则视为不拆
        if len(pts) <= 2:
            result.append(None)
        else:
            result.append(pts)

    return result


def find_buy_sell_points(strokes, pivots, mdf: pd.DataFrame, orig_df: pd.DataFrame):
    """
    三类买卖点（原文教你炒股票 20/21/24-25）：

    ===== 一买 1B / 一卖 1S =====
      核心：MACD 面积背驰（原文第24-25讲：面积缩小即为背驰）
      同向两笔（后一笔相对前一同向笔）：
        - 后笔创新高/新低 且 MACD 面积缩小 → 顶/底背驰 → 一卖/一买

    ===== 二买 2B / 二卖 2S =====
      原文教你炒股票20：**"第二类买点定义：一个次级别上涨结束后再次下跌形成的低点，
                          不跌破前面第一类买点的低点，就是第二类买点"**
      算法：一买 1B 之后的第一个下跌笔终点若 > 1B 价 → 2B（不要求"上涨-下跌"两段）
            一卖 1S 之后的第一个上涨笔终点若 < 1S 价 → 2S

      补充："中枢跌破/突破" 后的第一次反抽/回抽：
        - 中枢跌破后的反弹笔终点 若在 ZD 以下 → 2S 候选
        - 中枢突破后的回抽笔终点 若在 ZG 以上 → 2B 候选（同时也是 3B）

    ===== 三买 3B / 三卖 3S =====
      中枢向上突破，回抽笔终点 > ZG → 3B
      中枢向下跌破，反抽笔终点 < ZD → 3S
    """
    buys, sells = [], []
    macdh = orig_df['MACDh_12_26_9'].values
    orig_idx_map = mdf['orig_idx'].values

    def dedup_append(lst, item):
        if not any(x['idx'] == item['idx'] and x['type'] == item['type'] for x in lst):
            lst.append(item)

    # ================================================================
    # 一买 / 一卖：MACD 面积背驰（放宽版）
    # ================================================================
    for i in range(2, len(strokes)):
        a = strokes[i-2]; c = strokes[i]
        if a['direction'] != c['direction']:
            continue
        a0, a1 = orig_idx_map[a['start_idx']], orig_idx_map[a['end_idx']]
        c0, c1 = orig_idx_map[c['start_idx']], orig_idx_map[c['end_idx']]
        pa_pos, pa_neg = macd_area(macdh, a0, a1)
        pc_pos, pc_neg = macd_area(macdh, c0, c1)
        macd_valid = (pa_pos + pa_neg) > 5 and (pc_pos + pc_neg) > 5

        if c['direction'] == 'down':
            new_low = c['end_price'] < a['end_price']
            # 缠论核心：MACD 负面积缩小 = 背驰
            if new_low and macd_valid and pc_neg < pa_neg * 0.95:
                dedup_append(buys, {
                    'type': '1B', 'idx': c['end_idx'],
                    'price': c['end_price'],
                    'note': f'底背驰(MACD面积 {pc_neg:.0f}<{pa_neg:.0f})',
                    'stroke_i': i,
                })
        else:
            new_high = c['end_price'] > a['end_price']
            if new_high and macd_valid and pc_pos < pa_pos * 0.95:
                dedup_append(sells, {
                    'type': '1S', 'idx': c['end_idx'],
                    'price': c['end_price'],
                    'note': f'顶背驰(MACD面积 {pc_pos:.0f}<{pa_pos:.0f})',
                    'stroke_i': i,
                })

    # ================================================================
    # 一买 / 一卖：盘整背驰（中枢内 A 与 C 段比较）
    # ================================================================
    for pv in pivots:
        s0 = pv['start_stroke']    # A
        s2 = pv['end_stroke']      # C（或延伸后的最后一段，方向可能与 A 同向或反向，需过滤）
        A = strokes[s0]
        end_s = strokes[s2]
        # 只在 A 与 end_s 方向相同（同向"离开笔"）时比较
        if A['direction'] == end_s['direction']:
            a0, a1 = orig_idx_map[A['start_idx']], orig_idx_map[A['end_idx']]
            c0, c1 = orig_idx_map[end_s['start_idx']], orig_idx_map[end_s['end_idx']]
            pa_pos, pa_neg = macd_area(macdh, a0, a1)
            pc_pos, pc_neg = macd_area(macdh, c0, c1)
            if end_s['direction'] == 'down':
                if end_s['end_price'] < A['end_price'] and pa_neg > 5 and pc_neg < pa_neg * 0.95:
                    dedup_append(buys, {
                        'type': '1B', 'idx': end_s['end_idx'],
                        'price': end_s['end_price'],
                        'note': f'中枢盘整底背驰(面积 {pc_neg:.0f}<{pa_neg:.0f})',
                        'stroke_i': s2,
                    })
            else:
                if end_s['end_price'] > A['end_price'] and pa_pos > 5 and pc_pos < pa_pos * 0.95:
                    dedup_append(sells, {
                        'type': '1S', 'idx': end_s['end_idx'],
                        'price': end_s['end_price'],
                        'note': f'中枢盘整顶背驰(面积 {pc_pos:.0f}<{pa_pos:.0f})',
                        'stroke_i': s2,
                    })

    # ================================================================
    # 二买 / 二卖：一买后第一个反向笔终点不破一买价
    # 缠论20讲原文定义：只需 1 段反向，不要求 "上涨-下跌" 两段
    # 
    # 加强版：
    #   规则A (标准)：1B 后的 up-down 两段，第二段 down 终点 > 1B 价 → 2B
    #   规则B (对称买回)：1S 后紧接的第一段 down 终点 → 2B 候选（顶背驰后的回撤买回）
    #                     1B 后紧接的第一段 up 终点   → 2S 候选（底背驰后的反弹止盈）
    # ================================================================
    def add_second_points():
        # 规则 A：标准二买
        base_1B = [b for b in buys if b['type'] == '1B']
        for b in base_1B:
            si = b['stroke_i']
            if si + 1 >= len(strokes):
                continue
            up_stroke = strokes[si + 1]
            if up_stroke['direction'] != 'up':
                continue
            if si + 2 < len(strokes):
                down_stroke = strokes[si + 2]
                if down_stroke['direction'] == 'down' and down_stroke['end_price'] > b['price']:
                    dedup_append(buys, {
                        'type': '2B', 'idx': down_stroke['end_idx'],
                        'price': down_stroke['end_price'],
                        'note': f'一买后回抽不破 1B({b["price"]:.0f})',
                        'stroke_i': si + 2,
                    })
        base_1S = [s for s in sells if s['type'] == '1S']
        for s in base_1S:
            si = s['stroke_i']
            if si + 1 >= len(strokes):
                continue
            down_stroke = strokes[si + 1]
            if down_stroke['direction'] != 'down':
                continue
            if si + 2 < len(strokes):
                up_stroke = strokes[si + 2]
                if up_stroke['direction'] == 'up' and up_stroke['end_price'] < s['price']:
                    dedup_append(sells, {
                        'type': '2S', 'idx': up_stroke['end_idx'],
                        'price': up_stroke['end_price'],
                        'note': f'一卖后反抽不破 1S({s["price"]:.0f})',
                        'stroke_i': si + 2,
                    })
        # 规则 B：顶背驰后紧接的第一段下跌终点 = 二买（对称买回）
        for s in base_1S:
            si = s['stroke_i']
            if si + 1 >= len(strokes):
                continue
            nxt = strokes[si + 1]
            if nxt['direction'] == 'down':
                dedup_append(buys, {
                    'type': '2B', 'idx': nxt['end_idx'],
                    'price': nxt['end_price'],
                    'note': f'一卖后回撤到位（对称买回）',
                    'stroke_i': si + 1,
                })
        # 底背驰后紧接的第一段上涨终点 = 二卖（对称止盈）
        for b in base_1B:
            si = b['stroke_i']
            if si + 1 >= len(strokes):
                continue
            nxt = strokes[si + 1]
            if nxt['direction'] == 'up':
                dedup_append(sells, {
                    'type': '2S', 'idx': nxt['end_idx'],
                    'price': nxt['end_price'],
                    'note': f'一买后反弹到位（对称止盈）',
                    'stroke_i': si + 1,
                })
    add_second_points()

    # ================================================================
    # 趋势卖点/买点（缠论20讲："下跌趋势中每一次反弹都是卖点"）
    # 判定：如果连续 2 个以上中枢逐级下移（ZG 依次降低），当前处于下跌趋势中；
    #       则每一次向上反弹笔终点若 <= 上一个中枢的 ZG → 趋势卖点 (2S)。
    #       对称：连续 2 个以上中枢逐级上移，每次向下回踩笔终点 >= 上一个中枢 ZD → 趋势买点 (2B)。
    # ================================================================
    if len(pivots) >= 2:
        for k in range(1, len(pivots)):
            prev = pivots[k-1]
            curr = pivots[k]
            # 下移中枢
            if curr['ZG'] < prev['ZG'] and curr['ZD'] < prev['ZD']:
                # 当前中枢内部/结束后的每一次向上笔终点若 <= prev.ZG → 趋势卖点
                for si in range(curr['start_stroke'], min(curr['end_stroke']+3, len(strokes))):
                    s = strokes[si]
                    if s['direction'] == 'up' and s['end_price'] <= prev['ZG']:
                        # 且该 up 笔的终点相对前一 up 笔（若存在）力度衰减，才更可靠
                        # 简化：直接标为趋势 2S 候选
                        dedup_append(sells, {
                            'type': '2S', 'idx': s['end_idx'],
                            'price': s['end_price'],
                            'note': f'下跌趋势中反弹(<= 前中枢ZG {prev["ZG"]:.0f})',
                            'stroke_i': si,
                        })
            # 上移中枢
            if curr['ZG'] > prev['ZG'] and curr['ZD'] > prev['ZD']:
                for si in range(curr['start_stroke'], min(curr['end_stroke']+3, len(strokes))):
                    s = strokes[si]
                    if s['direction'] == 'down' and s['end_price'] >= prev['ZD']:
                        dedup_append(buys, {
                            'type': '2B', 'idx': s['end_idx'],
                            'price': s['end_price'],
                            'note': f'上涨趋势中回撤(>= 前中枢ZD {prev["ZD"]:.0f})',
                            'stroke_i': si,
                        })

    # ================================================================
    # 三买 / 三卖：中枢突破/跌破后的回抽/反抽
    # ================================================================
    for pv in pivots:
        ZG, ZD = pv['ZG'], pv['ZD']
        end_i = pv['end_stroke']
        last_in_pivot = strokes[end_i]
        broke_down = last_in_pivot['direction'] == 'down' and last_in_pivot['end_price'] < ZD
        broke_up = last_in_pivot['direction'] == 'up' and last_in_pivot['end_price'] > ZG

        # 计算离开笔与回抽/反抽笔
        def emit_3rd(leave_idx, pull_idx):
            if leave_idx >= len(strokes):
                return
            leave = strokes[leave_idx]
            up_out = leave['direction'] == 'up' and leave['end_price'] > ZG
            down_out = leave['direction'] == 'down' and leave['end_price'] < ZD
            if not (up_out or down_out):
                return
            if pull_idx < len(strokes):
                pull = strokes[pull_idx]
                if up_out and pull['direction'] == 'down' and pull['end_price'] > ZG:
                    dedup_append(buys, {
                        'type': '3B', 'idx': pull['end_idx'],
                        'price': pull['end_price'],
                        'note': f'突破中枢[{ZD:.0f},{ZG:.0f}]回抽不入',
                        'stroke_i': pull_idx,
                    })
                elif down_out and pull['direction'] == 'up' and pull['end_price'] < ZD:
                    dedup_append(sells, {
                        'type': '3S', 'idx': pull['end_idx'],
                        'price': pull['end_price'],
                        'note': f'跌破中枢[{ZD:.0f},{ZG:.0f}]反抽不入',
                        'stroke_i': pull_idx,
                    })
            else:
                if up_out:
                    dedup_append(buys, {
                        'type': '3B?', 'idx': leave['end_idx'],
                        'price': leave['end_price'],
                        'note': f'突破中枢，等回抽{ZG:.0f}上方',
                        'stroke_i': leave_idx,
                    })
                elif down_out:
                    dedup_append(sells, {
                        'type': '3S?', 'idx': leave['end_idx'],
                        'price': leave['end_price'],
                        'note': f'跌破中枢，等反抽{ZD:.0f}下方',
                        'stroke_i': leave_idx,
                    })

        if broke_up or broke_down:
            emit_3rd(end_i, end_i + 1)
        else:
            emit_3rd(end_i + 1, end_i + 2)

    # 二买/二卖再补一遍（因为可能新加入 1B/1S 来自盘整背驰）
    add_second_points()

    # ================================================================
    # 补丁：中枢跌破/突破的离开笔终点，若已完成反抽 → 潜在 1B?/1S?
    # 覆盖"下跌加速末端"这类未形成严格背驰的关键位（如 6/26 类型）
    # ================================================================
    for pv in pivots:
        ZG, ZD = pv['ZG'], pv['ZD']
        end_i = pv['end_stroke']
        # 情况 A: 中枢内最后一笔就是跌破/突破笔
        last = strokes[end_i]
        if last['direction'] == 'down' and last['end_price'] < ZD and end_i + 1 < len(strokes):
            # 有后续笔反弹 → 前一段下跌是"离开"，其终点是 1B? 观察位
            dedup_append(buys, {
                'type': '1B?', 'idx': last['end_idx'],
                'price': last['end_price'],
                'note': f'跌破中枢{ZD:.0f}后止跌观察(潜在一买)',
                'stroke_i': end_i,
            })
        elif last['direction'] == 'up' and last['end_price'] > ZG and end_i + 1 < len(strokes):
            dedup_append(sells, {
                'type': '1S?', 'idx': last['end_idx'],
                'price': last['end_price'],
                'note': f'突破中枢{ZG:.0f}后止涨观察(潜在一卖)',
                'stroke_i': end_i,
            })
        # 情况 B: 中枢结束后第一笔是离开笔（未纳入中枢延伸）
        if end_i + 1 < len(strokes):
            leave = strokes[end_i + 1]
            if leave['direction'] == 'down' and leave['end_price'] < ZD and end_i + 2 < len(strokes):
                dedup_append(buys, {
                    'type': '1B?', 'idx': leave['end_idx'],
                    'price': leave['end_price'],
                    'note': f'跌破中枢{ZD:.0f}后止跌观察(潜在一买)',
                    'stroke_i': end_i + 1,
                })
            elif leave['direction'] == 'up' and leave['end_price'] > ZG and end_i + 2 < len(strokes):
                dedup_append(sells, {
                    'type': '1S?', 'idx': leave['end_idx'],
                    'price': leave['end_price'],
                    'note': f'突破中枢{ZG:.0f}后止涨观察(潜在一卖)',
                    'stroke_i': end_i + 1,
                })

    # ================================================================
    # 补丁：长下跌笔/长上涨笔内部的次级反弹/回撤观察位
    # 覆盖 2/10、2/23 类型：主笔太长（超过 min_k_between 距离），内部有明显反弹
    # 用原始 K 线 high/low 找显著局部极值
    # ================================================================
    highs = orig_df['high'].values
    lows = orig_df['low'].values
    for stroke in strokes:
        s0 = orig_idx_map[stroke['start_idx']]
        s1 = orig_idx_map[stroke['end_idx']]
        length = s1 - s0
        if length < 15:   # 只处理超长笔（>15 根原始K线）
            continue
        start_price = float(orig_df.iloc[s0]['close'])
        end_price = float(orig_df.iloc[s1]['close'])
        stroke_range = abs(end_price - start_price)
        if stroke_range <= 0:
            continue
        # 在原始 K 线区间 [s0+3, s1-3] 内找显著局部极值（跳过端点附近）
        last_hit = -10
        for k in range(s0 + 3, s1 - 3):
            if k - last_hit < 4:
                continue  # 相邻观察位至少间隔 4 根K
            win_l = max(s0, k - 3)
            win_r = min(s1, k + 4)
            if stroke['direction'] == 'down':
                # 下跌笔内的次级反弹高点：high 局部最大
                if highs[k] == highs[win_l:win_r].max() and highs[k] > highs[k-1] and highs[k] > highs[k+1]:
                    # 反弹幅度需至少占该笔总跌幅的 15%，避免噪声
                    rally = highs[k] - min(lows[s0:k+1])
                    if rally >= stroke_range * 0.15 and rally / start_price >= 0.02:
                        mdf_idx = int(np.argmin(np.abs(orig_idx_map - k)))
                        dedup_append(sells, {
                            'type': '2S?', 'idx': mdf_idx,
                            'price': float(highs[k]),
                            'note': f'下跌笔内次级反弹观察位',
                            'stroke_i': -1,
                        })
                        last_hit = k
            else:
                if lows[k] == lows[win_l:win_r].min() and lows[k] < lows[k-1] and lows[k] < lows[k+1]:
                    pullback = max(highs[s0:k+1]) - lows[k]
                    if pullback >= stroke_range * 0.15 and pullback / start_price >= 0.02:
                        mdf_idx = int(np.argmin(np.abs(orig_idx_map - k)))
                        dedup_append(buys, {
                            'type': '2B?', 'idx': mdf_idx,
                            'price': float(lows[k]),
                            'note': f'上涨笔内次级回撤观察位',
                            'stroke_i': -1,
                        })
                        last_hit = k

    # 排序：按 idx 升序
    buys.sort(key=lambda x: x['idx'])
    sells.sort(key=lambda x: x['idx'])

    # 同一位置多标签合并：正式点 > 潜在观察点；同为正式时 1 > 3 > 2
    def _priority(t):
        return {'1B': 30, '1S': 30, '3B': 20, '3S': 20, '2B': 10, '2S': 10,
                '1B?': 3, '1S?': 3, '3B?': 2, '3S?': 2,
                '2B?': 1, '2S?': 1}.get(t, 0)
    def _reduce(pts):
        by_idx = {}
        for p in pts:
            if p['idx'] not in by_idx or _priority(p['type']) > _priority(by_idx[p['idx']]['type']):
                by_idx[p['idx']] = p
        return sorted(by_idx.values(), key=lambda x: x['idx'])
    buys = _reduce(buys)
    sells = _reduce(sells)
    return buys, sells


def interpret_market(orig_df, mdf, strokes, pivots, buys, sells):
    """
    基于《缠中说禅》原文（教你炒股票 17-25、63-84）生成当下走势解读文本。
    输出 list[str]，每项一行。

    结构：
      1. 基本面板：最新日期、收盘价、最新一笔
      2. 走势类型：盘整 / 上涨趋势 / 下跌趋势（依据中枢数量与同向性）
      3. 当前位置：相对最新中枢的位置态
      4. 最近信号：最后一个正式买/卖点
      5. 情景推演：IF-THEN 条件式操作建议（原文规则）
    """
    lines = ['【当下走势解读（基于缠论原文）】']
    if not strokes:
        lines.append('样本过短，未形成有效笔。')
        return lines

    last_date = orig_df['time_key'].iloc[-1]
    last_close = float(orig_df['close'].iloc[-1])
    last_date_str = last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date)[:10]
    last_stroke = strokes[-1]

    lines.append(f'· 最新日期: {last_date_str}   收盘: {last_close:.2f}')
    dir_zh = '上' if last_stroke['direction'] == 'up' else '下'
    lines.append(
        f'· 最新一笔({dir_zh}): {last_stroke["start_price"]:.0f} → '
        f'{last_stroke["end_price"]:.0f}  幅度 '
        f'{(last_stroke["end_price"]-last_stroke["start_price"])/last_stroke["start_price"]*100:+.1f}%'
    )

    # ---------- 走势类型判定（原文 63-64 页 走势分解定理） ----------
    # 依据最近两个中枢：同向且不重叠 → 趋势；单枢或反向重叠 → 盘整
    trend_type = '盘整'
    trend_note = ''
    if len(pivots) >= 2:
        p_last, p_prev = pivots[-1], pivots[-2]
        # 中枢方向：ZD/ZG 中值比较
        mid_last = (p_last['ZD'] + p_last['ZG']) / 2
        mid_prev = (p_prev['ZD'] + p_prev['ZG']) / 2
        overlap = not (p_last['ZD'] > p_prev['ZG'] or p_last['ZG'] < p_prev['ZD'])
        if not overlap:
            if mid_last < mid_prev:
                trend_type = '下跌趋势'
                trend_note = f'中枢逐级下移（{mid_prev:.0f} → {mid_last:.0f}）'
            elif mid_last > mid_prev:
                trend_type = '上涨趋势'
                trend_note = f'中枢逐级上移（{mid_prev:.0f} → {mid_last:.0f}）'
        else:
            trend_type = '盘整（中枢重叠/延伸）'
            trend_note = '最新中枢与前中枢有重叠，属同级别延伸'
    elif len(pivots) == 1:
        trend_type = '盘整（单一中枢）'
    lines.append(f'· 走势类型: {trend_type}' + (f'  · {trend_note}' if trend_note else ''))

    # ---------- 当前位置态（原文 75-84 页 三类买卖点） ----------
    if not pivots:
        lines.append('· 位置态: 尚未形成有效中枢')
    else:
        last_pivot = pivots[-1]
        zd, zg = last_pivot['ZD'], last_pivot['ZG']
        # 中枢时间范围（用最后一段收尾笔的时间）
        try:
            end_stroke = strokes[last_pivot['end_stroke']]
            start_stroke = strokes[last_pivot['start_stroke']]
            piv_t0 = orig_df.iloc[mdf['orig_idx'].values[start_stroke['start_idx']]]['time_key']
            piv_t1 = orig_df.iloc[mdf['orig_idx'].values[end_stroke['end_idx']]]['time_key']
            piv_range = f'（{piv_t0.strftime("%m-%d")} ~ {piv_t1.strftime("%m-%d")}）'
        except Exception:
            piv_range = ''

        lines.append(f'· 最新中枢: [{zd:.0f}, {zg:.0f}] {piv_range}')

        # 三态：破上 / 破下 / 中枢内
        if last_close > zg:
            up_pct = (last_close - zg) / zg * 100
            lines.append(f'· 位置态: 收盘 {last_close:.0f} 已突破中枢上沿 ZG={zg:.0f}（+{up_pct:.1f}%）')
            lines.append('  【离开中枢向上】按原文：')
            lines.append(f'  ▸ 若次级别回抽不跌破 ZG {zg:.0f} → 第三类买点，趋势延续，可介入')
            lines.append(f'  ▸ 若回抽跌回 {zg:.0f} 内部 → 中枢延伸/扩展，三买失败，退回震荡')
            lines.append('  ▸ 上破后继续新高，需警惕本级别趋势背驰形成一卖')
        elif last_close < zd:
            dn_pct = (zd - last_close) / zd * 100
            lines.append(f'· 位置态: 收盘 {last_close:.0f} 已跌破中枢下沿 ZD={zd:.0f}（-{dn_pct:.1f}%）')
            lines.append('  【离开中枢向下】按原文：')
            lines.append(f'  ▸ 若次级别反抽不站回 ZD {zd:.0f} → 第三类卖点，下跌延续，应减仓/观望')
            lines.append(f'  ▸ 若反抽站回 {zd:.0f} 内部 → 中枢延伸/扩展，三卖失败，回归震荡')
            lines.append('  ▸ 下破后继续新低，需关注本级别趋势背驰形成一买（次级别底分型 + MACD 面积缩小）')
        else:
            # 中枢内
            pos_in = (last_close - zd) / (zg - zd) * 100 if zg > zd else 50
            lines.append(f'· 位置态: 收盘 {last_close:.0f} 位于中枢内（相对下沿 {pos_in:.0f}%）')
            lines.append('  【中枢震荡】按原文：')
            lines.append(f'  ▸ 近 ZG {zg:.0f} → 逢高减，等破位方向；近 ZD {zd:.0f} → 逢低加')
            lines.append('  ▸ 中枢完成 9 段（每次进出算一段）后进入延伸判定；关注是否有力度不衰减的向上或向下离开笔')

    # ---------- 最近信号（正式点优先，? 潜在点次之） ----------
    all_pts = [(b, 'B') for b in buys] + [(s, 'S') for s in sells]
    all_pts.sort(key=lambda x: x[0]['idx'])
    if all_pts:
        # 找最近的正式点
        formal = [p for p in all_pts if not p[0]['type'].endswith('?')]
        latest = formal[-1] if formal else all_pts[-1]
        p, side = latest
        try:
            sig_date = orig_df.iloc[mdf['orig_idx'].values[p['idx']]]['time_key'].strftime('%Y-%m-%d')
        except Exception:
            sig_date = '?'
        lines.append(f'· 最近信号: {p["type"]} @ {sig_date}  {p["price"]:.0f}   {p["note"]}')

        t = p['type']
        # 依据最近信号给出跟进操作（原文 47-59, 75-84 页）
        if t == '1B':
            lines.append('  ▸ 一买已现 → 等次级别反弹后回抽不破本低点，成立二买；组合建仓')
        elif t == '2B':
            lines.append('  ▸ 二买已现 → 上涨延续；若后续突破前中枢 ZG 且回抽不入 → 三买加仓')
        elif t == '3B':
            lines.append('  ▸ 三买已现 → 上涨中枢移动确立；持有，直至出现顶背驰做一卖')
        elif t == '1S':
            lines.append('  ▸ 一卖已现 → 等次级别回落后反抽不破本高点，成立二卖；组合减仓')
        elif t == '2S':
            lines.append('  ▸ 二卖已现 → 下跌延续；若后续跌破前中枢 ZD 且反抽不入 → 三卖清仓')
        elif t == '3S':
            lines.append('  ▸ 三卖已现 → 下跌中枢移动确立；空仓等待底背驰一买')
    else:
        lines.append('· 最近信号: 无')

    # ---------- 补充：级别提示 ----------
    n_up_str = sum(1 for s in strokes if s['direction'] == 'up')
    n_dn_str = sum(1 for s in strokes if s['direction'] == 'down')
    lines.append(f'· 结构统计: {len(strokes)} 笔（{n_up_str} 上 / {n_dn_str} 下），'
                 f'{len(pivots)} 中枢，'
                 f'买 {len(buys)} 卖 {len(sells)}')

    return lines


# ================================================================
# 7. 可视化
# ================================================================
def plot_chan(orig_df, mdf, fractals, strokes, segments, pivots, buys, sells,
              sub_strokes=None,
              title='恒生科技指数 HK.800700  缠论分析'):
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.05], hspace=0.08,
                          left=0.05, right=0.72, top=0.94, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1], sharex=ax)

    n = len(orig_df)
    x = np.arange(n)
    dates = orig_df['time_key'].dt.strftime('%m-%d').values

    # --- K线 ---
    for i in range(n):
        o, c, h, l = orig_df.loc[i, ['open', 'close', 'high', 'low']]
        color = '#e74c3c' if c >= o else '#2ecc71'
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=1)
        rect = Rectangle((i - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-3),
                         facecolor=color, edgecolor=color, zorder=2)
        ax.add_patch(rect)

    # --- 分型（用合并后 mdf 的 orig_idx 映射回原始 x） ---
    orig_idx_map = mdf['orig_idx'].values
    for fi, ftype, fp in fractals:
        xi = orig_idx_map[fi]
        if ftype == 'top':
            ax.annotate('', xy=(xi, fp * 1.005), xytext=(xi, fp * 1.02),
                        arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1))
        else:
            ax.annotate('', xy=(xi, fp * 0.995), xytext=(xi, fp * 0.98),
                        arrowprops=dict(arrowstyle='->', color='#3498db', lw=1))

    # --- 笔 ---
    # 先画长笔的子笔（细线，次级别）
    if sub_strokes:
        for sub in sub_strokes:
            if not sub:
                continue
            xs = [p[0] for p in sub]
            ys = [p[1] for p in sub]
            ax.plot(xs, ys, color='#7f8c8d', lw=0.7, linestyle='-',
                    alpha=0.75, zorder=2.5)
            # 子笔的内部端点用小空心圆标一下（不含主笔起终点）
            for p in sub[1:-1]:
                ax.plot(p[0], p[1], marker='o', mfc='white',
                        mec='#7f8c8d', ms=4, zorder=2.6)
    # 再画主笔（长笔加粗以示区分）
    for s in strokes:
        x0 = orig_idx_map[s['start_idx']]
        x1 = orig_idx_map[s['end_idx']]
        y0 = s['start_price']; y1 = s['end_price']
        # 长笔（原始 K 线跨度 >=15）用粗线，其余保持默认
        span = x1 - x0
        lw_main = 2.2 if span >= 15 else 1.4
        ax.plot([x0, x1], [y0, y1], color='#34495e', lw=lw_main, zorder=3)

    # --- 线段（更粗） ---
    for seg in segments:
        s0 = strokes[seg['start_stroke']]
        s1 = strokes[seg['end_stroke']]
        x0 = orig_idx_map[s0['start_idx']]
        x1 = orig_idx_map[s1['end_idx']]
        y0 = seg['start_price']; y1 = seg['end_price']
        color = '#c0392b' if seg['direction'] == 'up' else '#27ae60'
        ax.plot([x0, x1], [y0, y1], color=color, lw=2.5, alpha=0.55, zorder=4)

    # --- 中枢 ---
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

    # --- 买卖点 ---
    marker_map_buy = {'1B': ('^', '#c0392b', 90),
                       '2B': ('^', '#e74c3c', 70),
                       '3B': ('^', '#f39c12', 70),
                       '1B?': ('D', '#c0392b', 55),
                       '2B?': ('D', '#e74c3c', 45),
                       '3B?': ('*', '#f39c12', 130)}
    marker_map_sell = {'1S': ('v', '#16a085', 90),
                        '2S': ('v', '#27ae60', 70),
                        '3S': ('v', '#2ecc71', 70),
                        '1S?': ('D', '#16a085', 55),
                        '2S?': ('D', '#27ae60', 45),
                        '3S?': ('*', '#2ecc71', 130)}
    # 潜在观察位透明度更低
    def _alpha(t):
        return 0.55 if t.endswith('?') else 1.0
    for b in buys:
        xi = orig_idx_map[b['idx']]
        m, cc, sz = marker_map_buy[b['type']]
        a = _alpha(b['type'])
        ax.scatter([xi], [b['price']], marker=m, s=sz, color=cc,
                   edgecolors='black', linewidths=0.6, zorder=6, alpha=a)
        ax.annotate(b['type'], (xi, b['price']),
                    xytext=(0, -14), textcoords='offset points',
                    ha='center', fontsize=8, color=cc, fontweight='bold',
                    alpha=a)
    for s in sells:
        xi = orig_idx_map[s['idx']]
        m, cc, sz = marker_map_sell[s['type']]
        a = _alpha(s['type'])
        ax.scatter([xi], [s['price']], marker=m, s=sz, color=cc,
                   edgecolors='black', linewidths=0.6, zorder=6, alpha=a)
        ax.annotate(s['type'], (xi, s['price']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, color=cc, fontweight='bold',
                    alpha=a)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('价格')
    ax.grid(True, alpha=0.25)

    # 当下走势解读（放到主图右侧图外，避免遮挡 K 线）
    interp_lines = interpret_market(orig_df, mdf, strokes, pivots, buys, sells)
    text = '\n'.join(interp_lines)
    fig.text(0.735, 0.94, text,
             ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffbe6',
                       edgecolor='#d4a017', alpha=0.95))

    # 图例
    legend_elems = [
        Line2D([0], [0], color='#34495e', lw=2.2, label='笔（长笔加粗）'),
        Line2D([0], [0], color='#7f8c8d', lw=0.7, alpha=0.75, label='子笔（次级）'),
        Line2D([0], [0], color='#c0392b', lw=2.5, alpha=0.55, label='线段(上)'),
        Line2D([0], [0], color='#27ae60', lw=2.5, alpha=0.55, label='线段(下)'),
        Rectangle((0, 0), 1, 1, facecolor='#f39c12', alpha=0.3, label='中枢'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#c0392b',
               markersize=10, label='1B 一买'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#e74c3c',
               markersize=9, label='2B 二买'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#f39c12',
               markersize=9, label='3B 三买'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#16a085',
               markersize=10, label='1S 一卖'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#27ae60',
               markersize=9, label='2S 二卖'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#2ecc71',
               markersize=9, label='3S 三卖'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#c0392b',
               markersize=7, alpha=0.6, label='B? 潜在买点'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#27ae60',
               markersize=7, alpha=0.6, label='S? 潜在卖点'),
    ]
    # 图例放到整个 figure 底部横向排列（图外），避免遮挡 K 线
    fig.legend(handles=legend_elems, loc='lower center',
               bbox_to_anchor=(0.385, 0.005), ncol=13, fontsize=8,
               frameon=True)

    # --- MACD 副图 ---
    macdh = orig_df['MACDh_12_26_9'].values
    colors = ['#e74c3c' if v >= 0 else '#2ecc71' for v in np.nan_to_num(macdh)]
    axm.bar(x, macdh, color=colors, width=0.8)
    axm.axhline(0, color='black', lw=0.5)
    axm.set_ylabel('MACDh (12,26,9)')
    axm.grid(True, alpha=0.25)

    # X 轴标签
    step = max(1, n // 15)
    axm.set_xticks(x[::step])
    axm.set_xticklabels(dates[::step], rotation=30, ha='right', fontsize=9)
    ax.set_xlim(-1, n)

    # 已通过 add_gridspec(left/right/top/bottom) 固定布局，无需 tight_layout
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

    # 若无 MACDh 列，自行计算（防御性）
    if 'MACDh_12_26_9' not in orig_df.columns or orig_df['MACDh_12_26_9'].isna().all():
        ema12 = orig_df['close'].ewm(span=12, adjust=False).mean()
        ema26 = orig_df['close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        orig_df['MACDh_12_26_9'] = (dif - dea) * 2

    # 1. 包含处理
    mdf = process_inclusion(orig_df)
    print(f"[包含处理] {len(orig_df)} -> {len(mdf)} 根合并K")

    # 2. 分型
    fractals = find_fractals(mdf)
    print(f"[分型] 共 {len(fractals)} 个"
          f" (顶 {sum(1 for f in fractals if f[1]=='top')},"
          f" 底 {sum(1 for f in fractals if f[1]=='bot')})")

    # 3. 笔
    strokes = build_strokes(fractals, mdf, min_k_between=3)
    print(f"[笔] 共 {len(strokes)} 笔")
    for i, s in enumerate(strokes):
        st = orig_df.loc[mdf['orig_idx'].values[s['start_idx']], 'time_key'].strftime('%m-%d')
        et = orig_df.loc[mdf['orig_idx'].values[s['end_idx']], 'time_key'].strftime('%m-%d')
        print(f"  笔{i:2d} {s['direction']:4s} {st}({s['start_price']:.2f}) -> {et}({s['end_price']:.2f})")

    # 4. 线段
    segments = build_segments(strokes)
    print(f"[线段] 共 {len(segments)} 段")
    for i, seg in enumerate(segments):
        print(f"  段{i} {seg['direction']:4s} 笔{seg['start_stroke']}~{seg['end_stroke']}  "
              f"{seg['start_price']:.2f} -> {seg['end_price']:.2f}")

    # 5. 中枢
    pivots = build_pivots(strokes)
    print(f"[中枢] 共 {len(pivots)} 个")
    for i, pv in enumerate(pivots):
        print(f"  中枢{i} 笔{pv['start_stroke']}~{pv['end_stroke']}  "
              f"ZD={pv['ZD']:.2f} ZG={pv['ZG']:.2f}")

    # 6. 买卖点
    buys, sells = find_buy_sell_points(strokes, pivots, mdf, orig_df)
    print(f"[买卖点] 买 {len(buys)}  卖 {len(sells)}")
    for b in buys:
        d = orig_df.loc[mdf['orig_idx'].values[b['idx']], 'time_key'].strftime('%Y-%m-%d')
        print(f"  {b['type']} {d} @ {b['price']:.2f}  {b['note']}")
    for s in sells:
        d = orig_df.loc[mdf['orig_idx'].values[s['idx']], 'time_key'].strftime('%Y-%m-%d')
        print(f"  {s['type']} {d} @ {s['price']:.2f}  {s['note']}")

    # 7. 绘图
    sub_strokes = build_sub_strokes(strokes, mdf, orig_df)
    n_sub = sum(1 for x in sub_strokes if x)
    if n_sub:
        print(f"[子笔] 共 {n_sub} 条长笔被拆出子笔序列")
        for i, sub in enumerate(sub_strokes):
            if not sub:
                continue
            pts_str = ' → '.join(
                f"{orig_df.iloc[p[0]]['time_key'].strftime('%m-%d')}({p[1]:.0f})"
                for p in sub
            )
            print(f"  笔{i}: {pts_str}")
    fig = plot_chan(orig_df, mdf, fractals, strokes, segments, pivots,
                    buys, sells, sub_strokes=sub_strokes)
    out_png = 'chan_result.png'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f"[保存] {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
