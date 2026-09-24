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
from datetime import timedelta

# 中文字体
for font in ['PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei', 'Microsoft YaHei',
             'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']:
    mpl.rcParams['font.sans-serif'] = [font] + mpl.rcParams['font.sans-serif']
mpl.rcParams['axes.unicode_minus'] = False


# ================================================================
# 0. 数据读取
# ================================================================
def _fmt_ts(ts, style='short'):
    """把 pandas Timestamp / datetime 按数据是否为日内自适应格式化。
    style='short' → 日线:%m-%d, 日内:%m-%d %H:%M
    style='full'  → 日线:%Y-%m-%d, 日内:%Y-%m-%d %H:%M
    """
    if not hasattr(ts, 'strftime'):
        return str(ts)[:16]
    intraday = ts.hour != 0 or ts.minute != 0
    if style == 'full':
        return ts.strftime('%Y-%m-%d %H:%M') if intraday else ts.strftime('%Y-%m-%d')
    return ts.strftime('%m-%d %H:%M') if intraday else ts.strftime('%m-%d')


def load_from_txt(path: str) -> pd.DataFrame:
    """从附件文本读取（空格分隔，首列是索引）。

    容错：跳过前面所有非数据行（例如 futu 连接日志），直到遇到 header 行
    （以 'time_key' 开头的那一行）。
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    # 定位 header 行
    header_idx = 0
    for i, ln in enumerate(raw_lines):
        toks = re.split(r'\s+', ln.strip())
        if 'time_key' in toks:
            header_idx = i
            break
    lines = raw_lines[header_idx:]
    header = re.split(r'\s+', lines[0].strip())
    rows = []
    for ln in lines[1:]:
        toks = re.split(r'\s+', ln.strip())
        if len(toks) < 3:
            continue
        # 数据行必须以数字（行号）开头，跳过非法行
        if not toks[0].lstrip('-').isdigit():
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


def load_from_futu(stock_code='HK.800700', start='2026-06-02', end='2026-11-25',
                   ktype='K_DAY', warmup_days=180):
    """
    从 futu 拉数据（可选，需要 OpenD 已启动）。

    ktype 直接透传 futu 官方枚举，常用：
      - 'K_DAY'   日线
      - 'K_60M'   1 小时线
      - 'K_120M'  2 小时线
      - 'K_240M'  4 小时线
      - 'K_WEEK'  周线 / 'K_MON' 月线（如有需要）

    warmup_days:
      为了让 MACD/均线/中枢等指标在用户指定的 start 处已经"预热"，实际拉取时
      向前多取 warmup_days 天数据；返回值 display_start 用于绘图裁剪。
      默认 180 天（≈ MA120 + 缓冲，保证 MA250 也能大部分算出）。

    Returns
    -------
    (data, display_start) : (pd.DataFrame, pd.Timestamp)
      data           = 全量 K 线（含预热段）
      display_start  = 用户实际指定的 start（转成 Timestamp）
    """
    from futu import OpenQuoteContext, KLType, RET_OK

    user_start = pd.to_datetime(start)
    fetch_start = (user_start - timedelta(days=warmup_days)).strftime('%Y-%m-%d')

    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    try:
        real_ktype = getattr(KLType, ktype)
        ret, data, _ = quote_ctx.request_history_kline(
            stock_code, start=fetch_start, end=end, ktype=real_ktype)
        if ret != RET_OK:
            raise RuntimeError(f"futu 拉取 {ktype} 失败: {data}")
        data = data.sort_values('time_key').reset_index(drop=True)
        data['time_key'] = pd.to_datetime(data['time_key'])
    finally:
        quote_ctx.close()
    return data, user_start


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
# 4. 线段划分（特征序列分型法 —— 原文 P344-346 第 67 讲）
# ================================================================
def _merge_feature_inclusion(feats, direction):
    """处理特征序列的包含关系。
    feats: list of (lo, hi, stroke_idx)  —— 每个反向笔的价格区间
    direction: 'up' 或 'down'（当前线段方向；决定包含合并的取舍）
       - 上升线段：反向笔=下跌笔，特征序列上求"顶分型"（后续），包含时取高高、低高
       - 下降线段：反向笔=上涨笔，特征序列上求"底分型"，包含时取低低、高低
    返回：合并后的特征序列 list of (lo, hi, stroke_idx)
    """
    if not feats:
        return []
    merged = [feats[0]]
    for lo, hi, si in feats[1:]:
        p_lo, p_hi, p_si = merged[-1]
        # 判断包含
        contains_new = (p_lo <= lo and p_hi >= hi)   # 前包含新
        contains_prev = (lo <= p_lo and hi >= p_hi)  # 新包含前
        if contains_new or contains_prev:
            if direction == 'up':
                # 向上线段的特征序列：包含时取"最高高、最高低"（向上合并）
                new_hi = max(p_hi, hi)
                new_lo = max(p_lo, lo)
            else:
                new_hi = min(p_hi, hi)
                new_lo = min(p_lo, lo)
            merged[-1] = (new_lo, new_hi, si)  # 用较新那个的 stroke_idx
        else:
            merged.append((lo, hi, si))
    return merged


def build_segments(strokes):
    """
    线段划分 —— 特征序列分型法（原文 P344-346，第 67 讲）：
      - 上升线段的特征序列 = 所有向下笔的区间 [end, start]（下笔）
      - 下降线段的特征序列 = 所有向上笔的区间 [start, end]（上笔）
      - 特征序列相邻元素若有包含关系 → 按线段方向合并
      - 在特征序列上找顶/底分型（三元素）→ 线段结束
        · 无缺口（元素 1-2 区间重叠）→ 直接在分型高/低点处结束
        · 有缺口（元素 1-2 区间不重叠）→ 需要反向特征序列再出现相应分型才确认
      - 至少 3 笔构成一段（原文 P340："被有重叠部分的连续三笔破坏"）
    """
    if len(strokes) < 3:
        return []

    n = len(strokes)
    segments = []

    # 用有限状态机在笔序列上滑动
    seg_dir = strokes[0]['direction']
    seg_start = 0                            # 线段起始笔索引
    seg_extreme_stroke = 0                   # 目前触及极值的同向笔索引
    seg_extreme_price = strokes[0]['end_price']

    # 待处理笔的游标
    k = 1
    while k < n:
        # 特征序列 = 所有 [seg_start+1, k] 中方向与 seg_dir 相反 的笔
        feats = []
        for j in range(seg_start + 1, k + 1):
            sj = strokes[j]
            if sj['direction'] == seg_dir:
                # 同向笔：更新极值
                if seg_dir == 'up' and sj['end_price'] > seg_extreme_price:
                    seg_extreme_price = sj['end_price']
                    seg_extreme_stroke = j
                elif seg_dir == 'down' and sj['end_price'] < seg_extreme_price:
                    seg_extreme_price = sj['end_price']
                    seg_extreme_stroke = j
                continue
            lo = min(sj['start_price'], sj['end_price'])
            hi = max(sj['start_price'], sj['end_price'])
            feats.append((lo, hi, j))

        # 合并包含
        merged = _merge_feature_inclusion(feats, seg_dir)

        # 找特征序列分型（至少 3 个元素）
        broken = False
        confirm_at = None  # 分型确认的中间元素笔索引
        seg_end_stroke = seg_extreme_stroke

        if len(merged) >= 3:
            # 只看最后 3 个元素，判断是否形成分型
            e1, e2, e3 = merged[-3], merged[-2], merged[-1]
            lo1, hi1, si1 = e1
            lo2, hi2, si2 = e2
            lo3, hi3, si3 = e3
            if seg_dir == 'up':
                # 顶分型：e2 的 hi 是最高，且 hi2 > hi1 且 hi2 > hi3
                is_top = (hi2 > hi1 and hi2 > hi3 and lo2 > lo1 and lo2 > lo3)
                if is_top:
                    # 判断缺口：e1 与 e2 是否有重叠
                    gap12 = (hi1 < lo2)  # 无重叠：hi1 < lo2 即为缺口
                    if not gap12:
                        # 无缺口：线段在 e2 之前（即 e2 之前的最后同向笔终点）结束
                        # 端点 = 顶分型对应的极值笔
                        broken = True
                        confirm_at = si2  # 分型第 2 元素笔
                    else:
                        # 有缺口：需要反向序列再出现底分型才确认
                        # 简化实现：从 e2 之后的笔开始，反向序列即"同向 seg_dir 的笔"
                        # 找 si2 之后的 3 个 seg_dir 方向笔的底分型（下线段中的顶）
                        rev_feats = []
                        for j2 in range(si2 + 1, n):
                            sj2 = strokes[j2]
                            if sj2['direction'] != seg_dir:
                                continue
                            lo_ = min(sj2['start_price'], sj2['end_price'])
                            hi_ = max(sj2['start_price'], sj2['end_price'])
                            rev_feats.append((lo_, hi_, j2))
                        rev_merged = _merge_feature_inclusion(rev_feats, 'down')
                        if len(rev_merged) >= 3:
                            r1, r2, r3 = rev_merged[-3], rev_merged[-2], rev_merged[-1]
                            is_bot = (r2[0] < r1[0] and r2[0] < r3[0]
                                      and r2[1] < r1[1] and r2[1] < r3[1])
                            if is_bot:
                                broken = True
                                confirm_at = r2[2]
            else:
                # 下降线段：底分型
                is_bot = (lo2 < lo1 and lo2 < lo3 and hi2 < hi1 and hi2 < hi3)
                if is_bot:
                    gap12 = (lo1 > hi2)
                    if not gap12:
                        broken = True
                        confirm_at = si2
                    else:
                        rev_feats = []
                        for j2 in range(si2 + 1, n):
                            sj2 = strokes[j2]
                            if sj2['direction'] != seg_dir:
                                continue
                            lo_ = min(sj2['start_price'], sj2['end_price'])
                            hi_ = max(sj2['start_price'], sj2['end_price'])
                            rev_feats.append((lo_, hi_, j2))
                        rev_merged = _merge_feature_inclusion(rev_feats, 'up')
                        if len(rev_merged) >= 3:
                            r1, r2, r3 = rev_merged[-3], rev_merged[-2], rev_merged[-1]
                            is_top = (r2[0] > r1[0] and r2[0] > r3[0]
                                      and r2[1] > r1[1] and r2[1] > r3[1])
                            if is_top:
                                broken = True
                                confirm_at = r2[2]

        if broken and (seg_extreme_stroke - seg_start + 1) >= 3:
            # 结束当前线段
            segments.append({
                'start_stroke': seg_start,
                'end_stroke': seg_extreme_stroke,
                'start_price': strokes[seg_start]['start_price'],
                'end_price': seg_extreme_price,
                'direction': seg_dir,
            })
            # 新线段从极值笔的下一笔开始
            new_start = seg_extreme_stroke + 1
            if new_start >= n:
                break
            seg_dir = strokes[new_start]['direction']
            seg_start = new_start
            seg_extreme_stroke = new_start
            seg_extreme_price = strokes[new_start]['end_price']
            k = new_start + 1
        else:
            k += 1

    # 收尾：最后一段（未确认破坏）也纳入，用于当下分析
    if seg_start < n:
        segments.append({
            'start_stroke': seg_start,
            'end_stroke': seg_extreme_stroke,
            'start_price': strokes[seg_start]['start_price'],
            'end_price': seg_extreme_price,
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
        # 中枢延伸判据（缠论 P77 定理一 + P82 第三类买卖点定义）：
        # 原文 Zn 是"与中枢方向一致的次级别走势段"，其区间 [dn,gn] 与 [ZD,ZG]
        # 有重叠即延伸。实操上等价于：中枢震荡里"回试段"（反向笔）的终点若不
        # 破 [ZD,ZG]，则震荡继续；一旦某笔终点破位（ep<ZD 或 ep>ZG）→ 该笔
        # 即"离开笔"，中枢结束（否则等到形成三买/三卖也不合直觉：破位那一
        # 刻就应该终结当前中枢，把破位后的走势交给"新生/扩展"判定）。
        # 这样避免把明显破位后再反抽的行情（如恒科 6/22 深度下探）也被吞
        # 进一个大中枢，从而丢失趋势背驰一买。
        j = i + 3
        while j < n:
            sj = strokes[j]
            ep = sj['end_price']
            # 关键：笔终点已离开中枢上下沿 → 视为离开笔，中枢结束
            if ep < ZD or ep > ZG:
                break
            lj, hj = rng(sj)
            # 二次保险：即便终点在区间内，笔的整体区间也必须与 [ZD,ZG] 有重叠
            if hj < ZD or lj > ZG:
                break
            end_stroke = j
            highs.append(hj); lows.append(lj)
            j += 1
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
    classify_pivot_relations(pivots)
    return pivots


def classify_pivot_relations(pivots):
    """根据缠论 P77-78 & 定理二判定相邻中枢关系（原地写入 pv['relation']）：
      - 'new'       新生：两中枢 [ZD,ZG] 无重叠，且波动区间 [DD,GG] 也无重叠
      - 'expansion' 扩展：两中枢 [ZD,ZG] 无重叠，但波动区间 [DD,GG] 有重叠
                    （对应原文定理二第三条：后 ZG<前 ZD 且后 GG>=前 DD → 高级别中枢）
      - 'extension' 延伸：两中枢 [ZD,ZG] 有重叠（同一中枢的自然延伸）
    第一个中枢标为 'first'。
    """
    for pi, pv in enumerate(pivots):
        if pi == 0:
            pv['relation'] = 'first'
            continue
        prev = pivots[pi-1]
        # 中枢区间重叠？
        overlap_z = not (pv['ZD'] > prev['ZG'] or pv['ZG'] < prev['ZD'])
        # 波动区间重叠？
        overlap_gg = not (pv['DD'] > prev['GG'] or pv['GG'] < prev['DD'])
        if overlap_z:
            pv['relation'] = 'extension'
        elif overlap_gg:
            pv['relation'] = 'expansion'
        else:
            pv['relation'] = 'new'
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
    # 缠论 C.9：区分盘整背驰(中枢内 A/C) vs 趋势背驰(相邻两同向中枢的 A/C)
    #   → 建立 stroke → pivot 映射，用于判断当前背驰跨不跨中枢
    # ================================================================
    # 构建 stroke_idx → pivot_idx 映射
    stroke2pivot = {}
    for pi, pv in enumerate(pivots):
        for si in range(pv['start_stroke'], pv['end_stroke'] + 1):
            stroke2pivot.setdefault(si, pi)  # 若一笔跨多个中枢，取第一个（早）

    def _classify_backchi(a_stroke_i, c_stroke_i):
        """返回 '盘整' 或 '趋势'。
        - 同一中枢内 → 盘整
        - 不同中枢 or 未在中枢内 → 趋势（比较保守：既然 a/c 已跨中枢，视为趋势）
        """
        pa = stroke2pivot.get(a_stroke_i)
        pc = stroke2pivot.get(c_stroke_i)
        if pa is not None and pc is not None and pa == pc:
            return '盘整'
        return '趋势'

    for i in range(2, len(strokes)):
        a = strokes[i-2]; c = strokes[i]
        if a['direction'] != c['direction']:
            continue
        a0, a1 = orig_idx_map[a['start_idx']], orig_idx_map[a['end_idx']]
        c0, c1 = orig_idx_map[c['start_idx']], orig_idx_map[c['end_idx']]
        pa_pos, pa_neg = macd_area(macdh, a0, a1)
        pc_pos, pc_neg = macd_area(macdh, c0, c1)
        macd_valid = (pa_pos + pa_neg) > 5 and (pc_pos + pc_neg) > 5
        kind = _classify_backchi(i-2, i)

        if c['direction'] == 'down':
            # 原文 P25/P30：第二次下跌"不必创新低"，只要 MACD 力度衰减即为背驰
            #   严格背驰：c.end < a.end（创新低 + 面积缩小）
            #   宽松背驰(W 底)：c.end 未大幅低于 a.end 但也未大幅高于（在 a.end ± 3% 内）
            #                   + MACD 面积明显缩小（<= a * 0.6，更严格避免误报）
            new_low = c['end_price'] < a['end_price']
            near_low = (c['end_price'] <= a['end_price'] * 1.03  # 距离 a 低点不超过 3%
                        and c['end_price'] >= a['end_price'] * 0.90)  # 也不能低太多（那样是新低场景）
            if new_low and macd_valid and pc_neg < pa_neg * 0.95:
                dedup_append(buys, {
                    'type': '1B', 'idx': c['end_idx'],
                    'price': c['end_price'],
                    'note': f'{kind}背驰底(MACD面积 {pc_neg:.0f}<{pa_neg:.0f})',
                    'stroke_i': i,
                })
            elif near_low and macd_valid and pc_neg < pa_neg * 0.6 and pa_neg > 20:
                dedup_append(buys, {
                    'type': '1B', 'idx': c['end_idx'],
                    'price': c['end_price'],
                    'note': f'{kind}背驰底(W底 面积 {pc_neg:.0f}<<{pa_neg:.0f})',
                    'stroke_i': i,
                })
        else:
            new_high = c['end_price'] > a['end_price']
            near_high = (c['end_price'] >= a['end_price'] * 0.97
                         and c['end_price'] <= a['end_price'] * 1.10)
            if new_high and macd_valid and pc_pos < pa_pos * 0.95:
                dedup_append(sells, {
                    'type': '1S', 'idx': c['end_idx'],
                    'price': c['end_price'],
                    'note': f'{kind}背驰顶(MACD面积 {pc_pos:.0f}<{pa_pos:.0f})',
                    'stroke_i': i,
                })
            elif near_high and macd_valid and pc_pos < pa_pos * 0.6 and pa_pos > 20:
                dedup_append(sells, {
                    'type': '1S', 'idx': c['end_idx'],
                    'price': c['end_price'],
                    'note': f'{kind}背驰顶(M头 面积 {pc_pos:.0f}<<{pa_pos:.0f})',
                    'stroke_i': i,
                })

    # ================================================================
    # 一买 / 一卖：盘整背驰（中枢内 A 段 与 中枢的同向"离开笔" C 段 比较）
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
                        'note': f'盘整背驰底(中枢内 面积 {pc_neg:.0f}<{pa_neg:.0f})',
                        'stroke_i': s2,
                    })
            else:
                if end_s['end_price'] > A['end_price'] and pa_pos > 5 and pc_pos < pa_pos * 0.95:
                    dedup_append(sells, {
                        'type': '1S', 'idx': end_s['end_idx'],
                        'price': end_s['end_price'],
                        'note': f'盘整背驰顶(中枢内 面积 {pc_pos:.0f}<{pa_pos:.0f})',
                        'stroke_i': s2,
                    })

    # ================================================================
    # 一买 / 一卖：趋势背驰（相邻两同向中枢：前中枢"进入 A" 与 后中枢"离开 C" 比较）
    # 原文 P105-106：趋势背驰 → 保证跌回前中枢，力度确定，一买/一卖成立
    # ================================================================
    for pi in range(1, len(pivots)):
        pv_prev, pv_curr = pivots[pi-1], pivots[pi]
        # 前中枢的方向可以看 A（start_stroke）的方向；两中枢同向定义为：
        #   前后两个中枢的"离开笔"方向相同（都下 → 下跌趋势；都上 → 上涨趋势）
        end_prev = strokes[pv_prev['end_stroke']]
        end_curr = strokes[pv_curr['end_stroke']]
        A_prev = strokes[pv_prev['start_stroke']]
        A_curr = strokes[pv_curr['start_stroke']]
        # 只有前后中枢的"起始笔 A"方向都相同（都是往同一方向进入的），才是同向趋势
        if A_prev['direction'] != A_curr['direction']:
            continue
        dirn = A_prev['direction']
        # 中枢关系需满足"下移 or 上移"（后中枢整体在前中枢的对应方向侧）
        if dirn == 'down':
            if not (pv_curr['ZG'] < pv_prev['ZG'] and pv_curr['ZD'] < pv_prev['ZD']):
                continue
        else:
            if not (pv_curr['ZG'] > pv_prev['ZG'] and pv_curr['ZD'] > pv_prev['ZD']):
                continue
        # A = 前中枢的 start 段，C = 后中枢的 end 段（若方向不同则跳过）
        if end_curr['direction'] != dirn:
            continue
        a0, a1 = orig_idx_map[A_prev['start_idx']], orig_idx_map[A_prev['end_idx']]
        c0, c1 = orig_idx_map[end_curr['start_idx']], orig_idx_map[end_curr['end_idx']]
        pa_pos, pa_neg = macd_area(macdh, a0, a1)
        pc_pos, pc_neg = macd_area(macdh, c0, c1)
        if dirn == 'down':
            if end_curr['end_price'] < A_prev['end_price'] and pa_neg > 5 and pc_neg < pa_neg * 0.95:
                dedup_append(buys, {
                    'type': '1B', 'idx': end_curr['end_idx'],
                    'price': end_curr['end_price'],
                    'note': f'趋势背驰底(跨中枢 面积 {pc_neg:.0f}<{pa_neg:.0f})',
                    'stroke_i': pv_curr['end_stroke'],
                })
        else:
            if end_curr['end_price'] > A_prev['end_price'] and pa_pos > 5 and pc_pos < pa_pos * 0.95:
                dedup_append(sells, {
                    'type': '1S', 'idx': end_curr['end_idx'],
                    'price': end_curr['end_price'],
                    'note': f'趋势背驰顶(跨中枢 面积 {pc_pos:.0f}<{pa_pos:.0f})',
                    'stroke_i': pv_curr['end_stroke'],
                })

    # ================================================================
    # 一买 / 一卖：跨中枢新低/新高背驰（补漏）
    # 场景：只形成了 1 个中枢，随后一笔离开中枢向下（or 向上）创出全局新低（or 新高）；
    #      与中枢**进入段**（enter_stroke，方向和离开段同向）的 MACD 面积比较缩小 → 1B/1S
    # 原文依据：P105-106 趋势背驰的一般化——只要"离开中枢的这一笔"相对"进入中枢的
    #          那一笔（同为主趋势方向）"力度衰减且创新极值即成立；不必等到形成两个中枢。
    # 这补的是上面 811 行"两中枢比较"要求太严的情况：HK.800700 08-12→09-11 类
    # 破位下跌只有 1 个可参照中枢，走不到 811 分支，但方向与力度都是标准趋势背驰。
    # ================================================================
    for pv in pivots:
        enter_idx = pv.get('enter_stroke', pv['start_stroke'] - 1)
        if enter_idx < 0 or enter_idx >= len(strokes):
            continue
        enter_s = strokes[enter_idx]
        dirn = enter_s['direction']
        # 中枢结束后，扫描后续同向笔序列，找**创新极值**且**跨越 ZD/ZG**的最远那一笔
        # 这样即使离开中枢后被拆成 上→下→上→下 多段，也能正确定位到"最深"那笔作为 C
        best_leave_i = None
        best_price = None
        j = pv['end_stroke'] + 1
        # 只在离开后未形成新中枢前扫描（走到 pivots 里下一个 pivot 开始为止）
        j_stop = len(strokes)
        # 找下一个 pivot 的起点作为扫描边界，避免跟"两中枢比较"分支重复
        for pv2 in pivots:
            if pv2['start_stroke'] > pv['end_stroke']:
                j_stop = min(j_stop, pv2['start_stroke'])
                break
        while j < j_stop:
            sj = strokes[j]
            if sj['direction'] == dirn:
                if dirn == 'down':
                    if sj['end_price'] < pv['ZD']:
                        if best_price is None or sj['end_price'] < best_price:
                            best_price = sj['end_price']
                            best_leave_i = j
                else:
                    if sj['end_price'] > pv['ZG']:
                        if best_price is None or sj['end_price'] > best_price:
                            best_price = sj['end_price']
                            best_leave_i = j
            j += 1
        if best_leave_i is None:
            continue
        leave_s = strokes[best_leave_i]
        a0, a1 = orig_idx_map[enter_s['start_idx']], orig_idx_map[enter_s['end_idx']]
        c0, c1 = orig_idx_map[leave_s['start_idx']], orig_idx_map[leave_s['end_idx']]
        pa_pos, pa_neg = macd_area(macdh, a0, a1)
        pc_pos, pc_neg = macd_area(macdh, c0, c1)
        if dirn == 'down':
            new_low = leave_s['end_price'] < enter_s['end_price']
            if new_low and pa_neg > 5 and pc_neg < pa_neg * 0.95:
                dedup_append(buys, {
                    'type': '1B', 'idx': leave_s['end_idx'],
                    'price': leave_s['end_price'],
                    'note': f'趋势背驰底(离开中枢[{pv["ZD"]:.0f},{pv["ZG"]:.0f}] 面积 {pc_neg:.0f}<{pa_neg:.0f})',
                    'stroke_i': best_leave_i,
                })
        else:
            new_high = leave_s['end_price'] > enter_s['end_price']
            if new_high and pa_pos > 5 and pc_pos < pa_pos * 0.95:
                dedup_append(sells, {
                    'type': '1S', 'idx': leave_s['end_idx'],
                    'price': leave_s['end_price'],
                    'note': f'趋势背驰顶(离开中枢[{pv["ZD"]:.0f},{pv["ZG"]:.0f}] 面积 {pc_pos:.0f}<{pa_pos:.0f})',
                    'stroke_i': best_leave_i,
                })

    # ================================================================
    # 二买 / 二卖：严格按缠论 P91 原文
    #   一买 → 次级上涨笔 → 次级下跌笔，下笔终点 > 1B 价 → 正式 2B
    #                                     下笔终点 < 1B 价 → 降级 2B?
    #   卖点对称。（原"对称买回/止盈"分支属波浪理论做法，已删除。）
    # ================================================================
    def add_second_points():
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
                if down_stroke['direction'] != 'down':
                    continue
                if down_stroke['end_price'] > b['price']:
                    dedup_append(buys, {
                        'type': '2B', 'idx': down_stroke['end_idx'],
                        'price': down_stroke['end_price'],
                        'note': f'一买后回抽不破 1B({b["price"]:.0f})',
                        'stroke_i': si + 2,
                    })
                else:
                    dedup_append(buys, {
                        'type': '2B?', 'idx': down_stroke['end_idx'],
                        'price': down_stroke['end_price'],
                        'note': f'一买后回抽已破 1B({b["price"]:.0f})，降级观察',
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
                if up_stroke['direction'] != 'up':
                    continue
                if up_stroke['end_price'] < s['price']:
                    dedup_append(sells, {
                        'type': '2S', 'idx': up_stroke['end_idx'],
                        'price': up_stroke['end_price'],
                        'note': f'一卖后反抽不破 1S({s["price"]:.0f})',
                        'stroke_i': si + 2,
                    })
                else:
                    dedup_append(sells, {
                        'type': '2S?', 'idx': up_stroke['end_idx'],
                        'price': up_stroke['end_price'],
                        'note': f'一卖后反抽已破 1S({s["price"]:.0f})，降级观察',
                        'stroke_i': si + 2,
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
        if last['direction'] == 'down' and last['end_price'] < ZD:
            # 跌破中枢下沿 → 潜在一买观察位（不再强要求后续有反弹笔）
            dedup_append(buys, {
                'type': '1B?', 'idx': last['end_idx'],
                'price': last['end_price'],
                'note': f'跌破中枢{ZD:.0f}后止跌观察(潜在一买)',
                'stroke_i': end_i,
            })
        elif last['direction'] == 'up' and last['end_price'] > ZG:
            dedup_append(sells, {
                'type': '1S?', 'idx': last['end_idx'],
                'price': last['end_price'],
                'note': f'突破中枢{ZG:.0f}后止涨观察(潜在一卖)',
                'stroke_i': end_i,
            })
        # 情况 B: 中枢结束后第一笔是离开笔（未纳入中枢延伸）
        if end_i + 1 < len(strokes):
            leave = strokes[end_i + 1]
            if leave['direction'] == 'down' and leave['end_price'] < ZD:
                dedup_append(buys, {
                    'type': '1B?', 'idx': leave['end_idx'],
                    'price': leave['end_price'],
                    'note': f'跌破中枢{ZD:.0f}后止跌观察(潜在一买)',
                    'stroke_i': end_i + 1,
                })
            elif leave['direction'] == 'up' and leave['end_price'] > ZG:
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


# ================================================================
# 7. 单个买卖点的详细解释（配合"最近 N 个信号"图上说明）
# ================================================================
_TYPE_DEFS = {
    '1B': ('一类买点 1B',
           '缠论原文 教你炒股票 20/24-25：下跌走势的最后一段出现"力度衰减"（MACD 面积缩小 / 或 W 底型面积大幅收敛），构成底背驰 → 一买。'),
    '2B': ('二类买点 2B',
           '缠论原文 教你炒股票 20-21：一买之后第一次向上、随后回抽的低点未跌破 1B 低点 → 二买。'),
    '3B': ('三类买点 3B',
           '缠论原文 教你炒股票 20（第三类买卖点定理）：一段次级别向上离开中枢后，回抽笔的低点不再跌回中枢 ZG 之下 → 三买。'),
    '1S': ('一类卖点 1S',
           '缠论原文 教你炒股票 20/24-25：上涨走势的最后一段出现"力度衰减"，构成顶背驰 → 一卖。'),
    '2S': ('二类卖点 2S',
           '缠论原文 教你炒股票 20-21：一卖之后第一次向下、随后反抽的高点未升破 1S 高点 → 二卖。'),
    '3S': ('三类卖点 3S',
           '缠论原文 教你炒股票 20（第三类买卖点定理）：一段次级别向下离开中枢后，反抽笔的高点不再升回中枢 ZD 之上 → 三卖。'),
}


def _find_related_pivot(sig, strokes, pivots):
    """找信号最相关的中枢：优先取信号发生笔所在或紧邻的中枢。"""
    si = sig.get('stroke_i', -1)
    if si < 0 or not pivots:
        return None
    # 1) 信号笔在某个中枢范围内
    for pv in pivots:
        if pv['start_stroke'] <= si <= pv['end_stroke']:
            return pv
    # 2) 信号笔是某中枢的离开笔（end_stroke+1）
    for pv in pivots:
        if pv['end_stroke'] + 1 == si:
            return pv
    # 3) 信号笔之前最近的中枢
    prev = None
    for pv in pivots:
        if pv['end_stroke'] < si:
            prev = pv
    return prev


def _collect_same_bar_signals(sig, buys, sells):
    """
    找到与 sig 位于同一根 K 线（相同 idx）的其他信号。
    用于识别"3S 与 1B? 同位置"这种缠论"卖买同源、级别不同"场景。
    返回 list[dict]（不含 sig 自己）。
    """
    same = []
    for other in list(buys) + list(sells):
        if other is sig:
            continue
        if other.get('idx') == sig.get('idx'):
            same.append(other)
    return same


def _level_note_for_conflict(sig, same_bar):
    """
    缠论《教你炒股票 20/24/72》：同一位置同时出现 3S 与 1B（或 3B 与 1S）
    并不冲突——3 类信号看的是本级别中枢结构，1 类信号看的是次级别力度衰减。
    操作上：本级别正式信号优先，潜在的 1 类做"次级别观察"处理。

    根据 sig 与 same_bar 的组合返回一句级别提示。
    覆盖四种"卖买同源"组合（含正式/潜在的排列）：
      本身是 1B（含 1B?），同位置有 3S/3S?
      本身是 1S（含 1S?），同位置有 3B/3B?
      本身是 3S（含 3S?），同位置有 1B/1B?
      本身是 3B（含 3B?），同位置有 1S/1S?
    """
    if not same_bar:
        return ''
    base = sig['type'].rstrip('?')
    other_bases = {o['type'].rstrip('?') for o in same_bar}

    # 场景 A：本身是 1B（含 1B?），同位置有 3S 家族 → 次级别观察位
    if base == '1B' and '3S' in other_bases:
        return ('[级别] 本级别看这是 3S（中枢破位、新一段下跌开启）；此处 1B 只可能是'
                '次级别（如 60min）观察位，日线 1B 需待本段下跌走完并出现日线背驰后成立。')
    # 场景 B：本身是 1S（含 1S?），同位置有 3B 家族 → 对称
    if base == '1S' and '3B' in other_bases:
        return ('[级别] 本级别看这是 3B（中枢突破、新一段上涨开启）；此处 1S 只可能是'
                '次级别（如 60min）观察位，日线 1S 需待本段上涨走完并出现日线背驰后成立。')
    # 场景 C：本身是 3S（含 3S?），同位置有 1B 家族 → 主动说明"卖买同源"
    if base == '3S' and '1B' in other_bases:
        return ('[级别] 原文"卖买同源"：本级别 3S 成立的同时，次级别 1B 候选可能已出现。'
                '操作次序：先按 3S 离场，空仓等次级别 1B 落地再进。')
    if base == '3B' and '1S' in other_bases:
        return ('[级别] 原文"卖买同源"：本级别 3B 成立的同时，次级别 1S 候选可能已出现。'
                '操作次序：先按 3B 进场，遇次级别 1S 减仓，日线 1S 才是主离场。')
    return ''


def explain_signal(sig, mdf, strokes, pivots, all_buys=None, all_sells=None):
    """
    为单个买卖点生成 2-4 行详细解释。
    每行不超过 42 字符，便于渲染在图上说明框内。
    返回 list[str]，第一行是标题（含日期/类型/价格），之后是解释。

    all_buys / all_sells: 用于检测"同一根 K 同时出现多个信号"的级别冲突场景；
                          缺省则不做级别标注。
    """
    base_type = sig['type'].rstrip('?')  # '1B?' → '1B'
    is_pending = sig['type'].endswith('?')
    typ_name, typ_def = _TYPE_DEFS.get(base_type, (sig['type'], ''))
    dt = _fmt_ts(mdf['time_key'].iloc[sig['idx']], style='full')

    lines = []
    # 首行：日期 类型 价格
    tag = '（潜在观察）' if is_pending else ''
    lines.append(f'· {dt}  {typ_name}{tag}  @ {sig["price"]:.2f}')

    # 第二行：原文定义（精简版）
    lines.append(f'  [定义] {typ_def}')

    # 第三行起：本次触发的具体理由（用 note + 关联中枢/笔的量化数据）
    reason_bits = [sig.get('note', '')]

    pv = _find_related_pivot(sig, strokes, pivots)
    si = sig.get('stroke_i', -1)
    if 0 <= si < len(strokes):
        s_now = strokes[si]
        t0 = _fmt_ts(mdf['time_key'].iloc[s_now['start_idx']], style='short')
        t1 = _fmt_ts(mdf['time_key'].iloc[s_now['end_idx']], style='short')
        dir_zh = '涨' if s_now['direction'] == 'up' else '跌'
        reason_bits.append(
            f'触发笔: {t0}({s_now["start_price"]:.0f})→{t1}({s_now["end_price"]:.0f}) {dir_zh}'
        )

    if pv is not None:
        reason_bits.append(
            f'关联中枢: [{pv["ZD"]:.0f}, {pv["ZG"]:.0f}]'
        )

    # 三类买卖点补一句"位置关系"
    if base_type == '3B' and pv is not None:
        reason_bits.append(f'回抽低点 {sig["price"]:.0f} > 中枢上沿 ZG={pv["ZG"]:.0f} → 中枢已被有效突破')
    elif base_type == '3S' and pv is not None:
        reason_bits.append(f'反抽高点 {sig["price"]:.0f} < 中枢下沿 ZD={pv["ZD"]:.0f} → 中枢已被有效跌破')

    for bit in reason_bits:
        if bit:
            lines.append(f'  [依据] {bit}')

    # 级别标注：同一位置同时出现 3S/3B 与 1B?/1S? 时按原文"卖买同源、级别不同"提示
    if all_buys is not None and all_sells is not None:
        same_bar = _collect_same_bar_signals(sig, all_buys, all_sells)
        level_note = _level_note_for_conflict(sig, same_bar)
        if level_note:
            lines.append(f'  {level_note}')

    return lines


def format_recent_signals_block(buys, sells, mdf, strokes, pivots, n=5,
                                include_pending=True):
    """
    从 buys+sells 里按时间倒序取最近 n 个信号，输出每个的详细解释块。
    include_pending: 是否包含 '1B?'/'2B?'/'3B?' 这类潜在信号。
    """
    all_sigs = list(buys) + list(sells)
    if not include_pending:
        all_sigs = [s for s in all_sigs if not s['type'].endswith('?')]
    if not all_sigs:
        return [f'【最近 {n} 个买卖点分析】', '  暂无信号']
    all_sigs.sort(key=lambda s: s['idx'], reverse=True)
    recent = all_sigs[:n]
    recent = list(reversed(recent))  # 时间正序输出（早→晚）

    out = [f'【最近 {len(recent)} 个买卖点分析（按时间正序）】']
    for i, sig in enumerate(recent, 1):
        out.append(f'{i}.')
        # 传入全量 buys/sells 以便检测同位置级别冲突
        out.extend(explain_signal(sig, mdf, strokes, pivots,
                                  all_buys=buys, all_sells=sells))
    return out


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
    last_date_str = _fmt_ts(last_date, style='full')
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
            piv_range = f'（{_fmt_ts(piv_t0)} ~ {_fmt_ts(piv_t1)}）'
        except Exception:
            piv_range = ''

        lines.append(f'· 最新中枢: [{zd:.0f}, {zg:.0f}] {piv_range}')
        # 中枢关系（新生/扩展/延伸）—— 缠论 P77-78 B.4
        rel = last_pivot.get('relation', 'first')
        if rel != 'first' and len(pivots) >= 2:
            prev_pv = pivots[-2]
            rel_zh = {'new': '新生（与前中枢完全不重叠）',
                      'expansion': '扩展（波动区间触及前中枢，形成高级别中枢）',
                      'extension': '延伸（与前中枢区间重叠，同级别震荡）'}.get(rel, rel)
            lines.append(
                f'· 中枢关系: {rel_zh}   前中枢[{prev_pv["ZD"]:.0f},{prev_pv["ZG"]:.0f}]'
                f' GG={prev_pv["GG"]:.0f} DD={prev_pv["DD"]:.0f}'
            )

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
            sig_date = _fmt_ts(orig_df.iloc[mdf['orig_idx'].values[p['idx']]]['time_key'], style='full')
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
              sub_strokes=None, display_start=None,
              title='恒生科技指数 HK.800700  缠论分析',
              n_recent_signals=5, recent_only_formal=False,
              show_fractals=True):
    """
    display_start: pd.Timestamp 或 None
      若非 None，则只在 x 轴上显示 time_key >= display_start 的部分（预热段留给
      指标计算，不进入可视区域）。所有分型/笔/中枢/买卖点仍是基于全量数据计算，
      只有 x 轴 xlim 被裁剪。

    n_recent_signals: int，默认 5
      在图右下方显示最近 N 个买卖点的详细解释（含原文定义与本次触发依据）。
      设为 0 时不显示。

    recent_only_formal: bool
      True 时只展示正式买卖点（1B/2B/3B/1S/2S/3S），不含 1B?/2B?/1S?/2S? 等潜在点。
    """
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.05], hspace=0.08,
                          left=0.05, right=0.72, top=0.94, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1], sharex=ax)

    n = len(orig_df)
    x = np.arange(n)
    # 自适应日期标签：若是日内数据（存在非零小时），显示 月-日 时:分；否则显示 月-日
    has_intraday = (orig_df['time_key'].dt.hour != 0).any() or \
                   (orig_df['time_key'].dt.minute != 0).any()
    if has_intraday:
        dates = orig_df['time_key'].dt.strftime('%m-%d %H:%M').values
    else:
        dates = orig_df['time_key'].dt.strftime('%m-%d').values

    # 计算展示区间起点（预热段之外），供中枢文字/买卖点判定使用
    if display_start is not None:
        mask = orig_df['time_key'] >= pd.to_datetime(display_start)
        x_start = int(np.argmax(mask.values)) if mask.any() else 0
    else:
        x_start = 0

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
    if show_fractals:
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
    x_end_all = n - 1
    for pv in pivots:
        s_start = strokes[pv['start_stroke']]
        s_end = strokes[pv['end_stroke']]
        x0 = orig_idx_map[s_start['start_idx']]
        x1 = orig_idx_map[s_end['end_idx']]
        # 完全在预热段（可见区间外）的中枢跳过绘制
        if x1 < x_start:
            continue
        # 部分与预热段相交：裁剪矩形起点到可见区间左边界，避免文本/矩形跑到图外
        vx0 = max(x0, x_start)
        vx1 = min(x1, x_end_all)
        if vx1 <= vx0:
            continue
        rect = Rectangle((vx0, pv['ZD']), vx1 - vx0, pv['ZG'] - pv['ZD'],
                         facecolor='#f39c12', edgecolor='#d35400',
                         alpha=0.18, lw=1.2, zorder=1.5)
        ax.add_patch(rect)
        # 文本 x 位置也约束在可见区间内
        tx = (vx0 + vx1) / 2
        ax.text(tx, pv['ZG'], f"中枢[{pv['ZD']:.0f},{pv['ZG']:.0f}]",
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

    # ---- 同位置信号去重（原文"卖买同源、级别不同"） ----
    # 当同一根 K（相同 idx）上出现多个信号，图上只画"最主要"的那一个，
    # 避免箭头堆叠遮挡；其余同位置信号仍在右侧说明框中出现并附带[级别]提示。
    # 优先级：正式 > 潜在；3 类 > 1 类 > 2 类（3 类是本级别中枢结构判断，最强）。
    _type_prio = {'3B': 0, '3S': 0, '1B': 1, '1S': 1, '2B': 2, '2S': 2}

    def _sig_prio(sig):
        base = sig['type'].rstrip('?')
        pending_penalty = 10 if sig['type'].endswith('?') else 0
        return _type_prio.get(base, 5) + pending_penalty

    # 每个 idx 只保留优先级最高的那一个（数值越小越优先）
    keep_by_idx = {}
    for sig in list(buys) + list(sells):
        idx = sig['idx']
        p = _sig_prio(sig)
        if idx not in keep_by_idx or p < keep_by_idx[idx][0]:
            keep_by_idx[idx] = (p, id(sig))

    def _plot_visible(sig):
        return keep_by_idx.get(sig['idx'], (99, 0))[1] == id(sig)

    for b in buys:
        if not _plot_visible(b):
            continue
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
        if not _plot_visible(s):
            continue
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
    interp_text = '\n'.join(interp_lines)
    # 上半：当下走势解读
    fig.text(0.735, 0.94, interp_text,
             ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffbe6',
                       edgecolor='#d4a017', alpha=0.95))

    # 下半：最近 N 个买卖点分析
    if n_recent_signals > 0:
        # 估算走势解读占用的纵向空间：按每行约 0.017 fig 高度
        interp_h = 0.017 * max(len(interp_lines), 5) + 0.02
        y_top = 0.94 - interp_h - 0.015  # 与上半留一点空隙
        recent_lines = format_recent_signals_block(
            buys, sells, mdf, strokes, pivots,
            n=n_recent_signals, include_pending=not recent_only_formal)
        recent_text = '\n'.join(recent_lines)
        # 若纵向不够，字号缩到 7.5
        n_line = len(recent_lines)
        fs = 8 if n_line <= 26 else 7 if n_line <= 34 else 6.2
        fig.text(0.735, y_top, recent_text,
                 ha='left', va='top', fontsize=fs,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#eefbf3',
                           edgecolor='#27ae60', alpha=0.95))

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
    # x 轴范围：若指定 display_start，只显示预热段之后的区间
    if display_start is not None:
        mask = orig_df['time_key'] >= pd.to_datetime(display_start)
        if mask.any():
            x_start = int(np.argmax(mask.values))  # 第一个 True 的位置
        else:
            x_start = 0
    else:
        x_start = 0
    x_end = n - 1
    visible_len = x_end - x_start + 1
    step = max(1, visible_len // 15)
    tick_pos = np.arange(x_start, x_end + 1, step)
    axm.set_xticks(tick_pos)
    axm.set_xticklabels(dates[tick_pos], rotation=30, ha='right', fontsize=9)
    ax.set_xlim(x_start - 0.5, x_end + 0.5)

    # y 轴自动按可见段缩放（避免预热段的历史价拉扁主图）
    vis_slice = orig_df.iloc[x_start:x_end + 1]
    y_lo = float(vis_slice['low'].min())
    y_hi = float(vis_slice['high'].max())
    y_pad = (y_hi - y_lo) * 0.05 or 1.0
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    # MACD 副图 y 轴同理按可见段缩放
    vis_macd = macdh[x_start:x_end + 1]
    vis_macd = vis_macd[~np.isnan(vis_macd)]
    if len(vis_macd):
        m_max = float(np.abs(vis_macd).max())
        axm.set_ylim(-m_max * 1.1, m_max * 1.1)

    # 已通过 add_gridspec(left/right/top/bottom) 固定布局，无需 tight_layout
    return fig


# ================================================================
# 主流程
# ================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='缠论日/时线分析')
    parser.add_argument('--source', choices=['txt', 'futu'], default='futu',
                        help='数据源：txt 从本地文件读，futu 走 OpenD 拉取')
    parser.add_argument('--txt', default='长文本-1790229713.txt',
                        help='本地 txt 数据文件路径（source=txt 时使用）')
    parser.add_argument('--stock', default='HK.00700',
                        help='股票/指数代码（source=futu 时使用）')
    parser.add_argument('--start', default='2026-02-01')
    parser.add_argument('--end',   default='2026-09-25')
    parser.add_argument('--ktype',
                        choices=['K_DAY', 'K_60M', 'K_120M', 'K_240M',
                                 'K_WEEK', 'K_MON'],
                        default='K_DAY',
                        help='K 线周期，直接透传给 futu：日/1H/2H/4H/周/月')
    parser.add_argument('--out', default='chan_result.png',
                        help='输出图片路径')
    parser.add_argument('--warmup', type=int, default=180,
                        help='futu 数据源预热天数：向前多拉 N 天用于计算 MACD/均线/中枢，'
                             '展示时裁剪回 --start 起，仅算法计算用全量。默认 180（半年）')
    parser.add_argument('--display-start', default=None,
                        help='txt 数据源可选：只展示 >= 此日期的 K 线，之前作为预热段。'
                             '格式 YYYY-MM-DD。')
    parser.add_argument('--recent-signals', type=int, default=10,
                        help='在图右侧显示最近 N 个买卖点的详细解释（含原文定义与'
                             '本次触发依据）。设 0 关闭。默认 5。')
    parser.add_argument('--recent-only-formal', action='store_true',
                        help='最近信号只显示正式买卖点（不含 1B?/2B?/1S?/2S? 等潜在点）。')
    parser.add_argument('--hide-fractals', action='store_true',
                        help='隐藏图上的顶/底分型箭头（橙色向下箭头=顶分型，蓝色向上箭头=底分型）。'
                             '默认显示；仅影响绘图，不影响笔/线段/中枢/买卖点的计算。')
    args = parser.parse_args()

    display_start = None
    if args.source == 'txt':
        if not os.path.exists(args.txt):
            raise FileNotFoundError(f"txt 数据文件不存在: {args.txt}")
        orig_df = load_from_txt(args.txt)
        print(f"[加载] 从本地文本读取 {len(orig_df)} 根K线  file={args.txt}")
        if args.display_start:
            display_start = pd.to_datetime(args.display_start)
    else:
        orig_df, display_start = load_from_futu(
            stock_code=args.stock, start=args.start, end=args.end,
            ktype=args.ktype, warmup_days=args.warmup)
        n_warm = int((orig_df['time_key'] < display_start).sum())
        n_show = len(orig_df) - n_warm
        print(f"[加载] 从 futu 拉取 {len(orig_df)} 根K线  "
              f"{args.stock} {args.ktype} 预热{n_warm}根 + 展示{n_show}根 "
              f"(展示区间 {args.start}~{args.end})")

    # 标题依据周期区分
    ktype_zh = {'K_DAY': '日线', 'K_60M': '1小时线', 'K_120M': '2小时线',
                'K_240M': '4小时线', 'K_WEEK': '周线', 'K_MON': '月线'}
    title_ktype = ktype_zh.get(args.ktype, args.ktype)
    title = f'{args.stock}  缠论分析（{title_ktype}）'

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
        st = _fmt_ts(orig_df.loc[mdf['orig_idx'].values[s['start_idx']], 'time_key'])
        et = _fmt_ts(orig_df.loc[mdf['orig_idx'].values[s['end_idx']], 'time_key'])
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
                    buys, sells, sub_strokes=sub_strokes,
                    display_start=display_start, title=title,
                    n_recent_signals=args.recent_signals,
                    recent_only_formal=args.recent_only_formal,
                    show_fractals=not args.hide_fractals)
    out_png = args.out
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f"[保存] {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
