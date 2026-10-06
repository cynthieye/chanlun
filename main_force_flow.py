"""高抛低吸 + 抄底逃顶：单文件联合版。

两套信号识别逻辑仍由两个独立函数实现，并在主图叠加；下方独立显示主力吸筹/出货火焰：
- calculate_buy_low_sell_high：高抛低吸
- calculate_bottom_top_escape：抄底逃顶
- calculate_main_force_flow：主力吸筹/出货

依赖：pip install futu-api pandas numpy matplotlib
运行：python combined_signals_standalone.py
"""
from __future__ import annotations

import argparse
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font, colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def configure_chinese_font() -> str | None:
    preferred = [
        "Microsoft YaHei", "Microsoft JhengHei", "SimHei", "SimSun",
        "PingFang SC", "Heiti SC", "Noto Sans CJK SC", "Noto Sans CJK JP",
        "Source Han Sans SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    fonts = list(font_manager.fontManager.ttflist)
    fonts.sort(key=lambda f: (
        preferred.index(f.name) if f.name in preferred else len(preferred), f.name
    ))
    required = [ord("高"), ord("抛"), ord("低"), ord("吸"), ord("逃"), ord("顶")]
    selected = None
    for font in fonts:
        try:
            cmap = ft2font.FT2Font(font.fname).get_charmap()
            if all(code in cmap for code in required):
                selected = font
                break
        except (RuntimeError, OSError):
            continue
    if selected:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected.name, "DejaVu Sans"]
        print(f"Matplotlib中文字体：{selected.name} ({selected.fname})")
    else:
        print("警告：未找到中文字体；Windows请启用微软雅黑，Linux请安装fonts-noto-cjk。")
    plt.rcParams["axes.unicode_minus"] = False
    return selected.name if selected else None


def fetch_futu_daily_kline(
    stock_code: str, start: str, end: str,
    host: str = "127.0.0.1", port: int = 11111,
) -> pd.DataFrame:
    """从Futu OpenD拉取完整日K，自动处理分页。"""
    try:
        from futu import KLType, OpenQuoteContext, RET_OK
    except ImportError as exc:
        raise RuntimeError("缺少futu-api，请执行：pip install futu-api") from exc

    quote_ctx = OpenQuoteContext(host=host, port=port)
    pages: list[pd.DataFrame] = []
    page_req_key = None
    try:
        while True:
            ret, page, page_req_key = quote_ctx.request_history_kline(
                code=stock_code, start=start, end=end,
                ktype=KLType.K_DAY, page_req_key=page_req_key,
            )
            if ret != RET_OK:
                raise RuntimeError(f"富途日K拉取失败：{page}")
            if page is not None and not page.empty:
                pages.append(page)
            if page_req_key is None:
                break
    finally:
        quote_ctx.close()

    if not pages:
        raise RuntimeError(f"富途未返回日K：{stock_code}, {start}～{end}")
    data = pd.concat(pages, ignore_index=True)
    required = ["time_key", "open", "close", "high", "low"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"富途返回数据缺少字段：{missing}")
    data["time_key"] = pd.to_datetime(data["time_key"], errors="coerce")
    for column in ["open", "close", "high", "low", "volume", "turnover"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return (
        data.dropna(subset=required).sort_values("time_key")
        .drop_duplicates("time_key", keep="last").reset_index(drop=True)
    )


def tdx_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信SMA(X,N,M)，不是rolling mean。"""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan)
    previous = np.nan
    for index, value in enumerate(values):
        if np.isnan(value):
            continue
        previous = value if np.isnan(previous) else (m * value + (n - m) * previous) / n
        output[index] = previous
    return pd.Series(output, index=series.index)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def one_confirmation_per_candidate(
    candidate: pd.Series, confirmation: pd.Series, valid_days: int
) -> pd.Series:
    """每个候选在有效期内最多产生一次确认。"""
    output = np.zeros(len(candidate), dtype=bool)
    remaining = 0
    for index, (new_candidate, confirmed) in enumerate(zip(candidate, confirmation)):
        if bool(new_candidate):
            remaining = valid_days
        if remaining > 0 and bool(confirmed):
            output[index] = True
            remaining = 0
        elif remaining > 0:
            remaining -= 1
    return pd.Series(output, index=candidate.index)


# ============================================================================
# 高抛低吸独立识别逻辑
# ============================================================================
def calculate_buy_low_sell_high(data: pd.DataFrame) -> pd.DataFrame:
    """高抛低吸：K/D交叉 + RSI + 趋势过滤 + 6日右侧确认。"""
    result = data.sort_values("time_key").reset_index(drop=True).copy()
    for column in ["open", "close", "high", "low"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    close, high, low = result["close"], result["high"], result["low"]

    result["EMA10"] = close.ewm(span=10, adjust=False).mean()
    result["EMA20"] = close.ewm(span=20, adjust=False).mean()
    result["MA60"] = close.rolling(60, min_periods=60).mean()
    result["MA120"] = close.rolling(120, min_periods=120).mean()
    result["RSI14"] = calculate_rsi(close)

    lowest = low.rolling(21, min_periods=21).min()
    highest = high.rolling(21, min_periods=21).max()
    rsv = ((close - lowest) / (highest - lowest).replace(0, np.nan) * 100).clip(0, 100)
    result["K"] = tdx_sma(rsv, 3, 1)
    result["D"] = tdx_sma(result["K"], 3, 1)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    result["MACD柱"] = macd - signal

    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low, (high - previous_close).abs(), (low - previous_close).abs()
    ], axis=1).max(axis=1)
    result["ATR14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    cross_up = (result["K"] > result["D"]) & (result["K"].shift(1) <= result["D"].shift(1))
    cross_down = (result["K"] < result["D"]) & (result["K"].shift(1) >= result["D"].shift(1))
    ma60_slope = result["MA60"].pct_change(5)
    low_trend_filter = (
        (close > result["MA120"])
        | (ma60_slope > -0.012)
        | ((close > result["EMA20"]) & (result["MACD柱"].diff() > 0))
    ).fillna(False)
    high_trend_filter = (
        (result["MACD柱"] < result["MACD柱"].shift(1))
        | (close < result["EMA10"])
        | (ma60_slope <= 0)
    ).fillna(False)

    result["低吸候选"] = (
        cross_up & (result["K"] < 32) & (result["RSI14"] < 45) & low_trend_filter
    ).fillna(False)
    result["高抛候选"] = (
        cross_down & (result["K"] > 68) & (result["RSI14"] > 55) & high_trend_filter
    ).fillna(False)

    price_cross_up = (close > result["EMA10"]) & (close.shift(1) <= result["EMA10"].shift(1))
    price_cross_down = (close < result["EMA10"]) & (close.shift(1) >= result["EMA10"].shift(1))
    macd_improving = result["MACD柱"] > result["MACD柱"].shift(1)
    macd_weakening = result["MACD柱"] < result["MACD柱"].shift(1)
    low_confirmation = price_cross_up & macd_improving & low_trend_filter
    high_confirmation = price_cross_down | (macd_weakening & (result["K"] < 60))
    result["低吸确认"] = one_confirmation_per_candidate(
        result["低吸候选"], low_confirmation, 6
    )
    result["高抛确认"] = one_confirmation_per_candidate(
        result["高抛候选"], high_confirmation, 6
    )
    # 高抛低吸副图强度：K、D、RSI共同描述0～100的短期位置。
    result["高抛低吸强度"] = (
        0.35 * result["K"] + 0.25 * result["D"] + 0.40 * result["RSI14"]
    ).clip(0, 100)
    return result


# ============================================================================
# 抄底逃顶独立识别逻辑
# ============================================================================
def calculate_bottom_top_escape(data: pd.DataFrame) -> pd.DataFrame:
    """抄底逃顶：120日极端位置 + RSI背离/MACD拐点 + 10日右侧确认。"""
    result = data.sort_values("time_key").reset_index(drop=True).copy()
    for column in ["open", "close", "high", "low"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    close, high, low = result["close"], result["high"], result["low"]

    result["EMA10"] = close.ewm(span=10, adjust=False).mean()
    result["EMA20"] = close.ewm(span=20, adjust=False).mean()
    result["MA60"] = close.rolling(60, min_periods=60).mean()
    result["MA120"] = close.rolling(120, min_periods=120).mean()
    result["RSI14"] = calculate_rsi(close)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    result["MACD柱"] = macd - signal

    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low, (high - previous_close).abs(), (low - previous_close).abs()
    ], axis=1).max(axis=1)
    result["ATR14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    low120 = low.rolling(120, min_periods=120).min()
    high120 = high.rolling(120, min_periods=120).max()
    result["区间位置"] = (
        (close - low120) / (high120 - low120).replace(0, np.nan) * 100
    ).clip(0, 100)

    prior_price_low = low.shift(5).rolling(30, min_periods=20).min()
    prior_price_high = high.shift(5).rolling(30, min_periods=20).max()
    prior_rsi_low = result["RSI14"].shift(5).rolling(30, min_periods=20).min()
    prior_rsi_high = result["RSI14"].shift(5).rolling(30, min_periods=20).max()
    result["底背离"] = (
        (low <= prior_price_low) & (result["RSI14"] > prior_rsi_low + 3)
    ).fillna(False)
    result["顶背离"] = (
        (high >= prior_price_high) & (result["RSI14"] < prior_rsi_high - 3)
    ).fillna(False)

    macd_improving = (result["MACD柱"] > result["MACD柱"].shift(1)) & (
        result["MACD柱"].shift(1) <= result["MACD柱"].shift(2)
    )
    macd_weakening = (result["MACD柱"] < result["MACD柱"].shift(1)) & (
        result["MACD柱"].shift(1) >= result["MACD柱"].shift(2)
    )
    near_low = low <= low.rolling(20, min_periods=20).min() * 1.015
    near_high = high >= high.rolling(20, min_periods=20).max() * 0.985
    bottom_condition = (
        (result["区间位置"] <= 18) & (result["RSI14"] <= 38)
        & near_low & (result["底背离"] | macd_improving)
    ).fillna(False)
    top_condition = (
        (result["区间位置"] >= 82) & (result["RSI14"] >= 62)
        & near_high & (result["顶背离"] | macd_weakening)
    ).fillna(False)
    result["抄底候选"] = bottom_condition & ~bottom_condition.shift(1, fill_value=False)
    result["逃顶候选"] = top_condition & ~top_condition.shift(1, fill_value=False)

    cross_ema_up = (close > result["EMA10"]) & (close.shift(1) <= result["EMA10"].shift(1))
    cross_ema_down = (close < result["EMA10"]) & (close.shift(1) >= result["EMA10"].shift(1))
    bottom_confirmation = cross_ema_up & (result["RSI14"] > 40) & (
        result["MACD柱"] > result["MACD柱"].shift(1)
    )
    top_confirmation = cross_ema_down & (result["RSI14"] < 60) & (
        result["MACD柱"] < result["MACD柱"].shift(1)
    )
    result["抄底确认"] = one_confirmation_per_candidate(
        result["抄底候选"], bottom_confirmation, 10
    )
    result["逃顶确认"] = one_confirmation_per_candidate(
        result["逃顶候选"], top_confirmation, 10
    )
    # 极端强度：正值表示偏向底部极端，负值表示偏向顶部极端。
    position_component = (50 - result["区间位置"]) / 50
    rsi_component = (50 - result["RSI14"]) / 50
    macd_scale = result["MACD柱"].abs().rolling(
        60, min_periods=20
    ).median().replace(0, np.nan)
    macd_component = (-result["MACD柱"] / (3 * macd_scale)).clip(-1, 1)
    result["极端强度"] = (
        (0.50 * position_component + 0.35 * rsi_component + 0.15 * macd_component)
        * 100
    ).clip(-100, 100).fillna(0)
    return result


def calculate_main_force_flow(data: pd.DataFrame, period: int = 34) -> pd.DataFrame:
    """计算阶段新低吸筹与阶段新高出货的Main Force Flow代理指标。"""
    if period < 2:
        raise ValueError("main-force-period必须大于等于2")

    result = data.sort_values("time_key").reset_index(drop=True).copy()
    previous_average = (
        (result["low"] + result["open"] + result["close"] + result["high"]) / 4
    ).shift(1)

    low_difference = result["low"] - previous_average
    low_numerator = tdx_sma(low_difference.abs(), period, 1)
    low_denominator = tdx_sma(low_difference.clip(lower=0), 13, 1).replace(0, np.nan)
    low_base = (low_numerator / low_denominator).ewm(
        span=period, adjust=False, min_periods=1
    ).mean()
    period_low = result["low"].rolling(period, min_periods=period).min()
    accumulation_input = pd.Series(
        np.where(result["low"] <= period_low, low_base, 0.0), index=result.index
    )
    accumulation_core = accumulation_input.ewm(span=3, adjust=False, min_periods=1).mean()
    accumulation_floor = accumulation_core.expanding().max().fillna(0) * 0.01
    result["主力吸筹"] = accumulation_core.where(
        accumulation_core > accumulation_floor, 0.0
    )

    high_difference = result["high"] - previous_average
    high_numerator = tdx_sma(high_difference.abs(), period, 1)
    high_denominator = tdx_sma(high_difference.clip(upper=0), period, 1).replace(0, np.nan)
    high_base = (high_numerator / high_denominator).ewm(
        span=period, adjust=False, min_periods=1
    ).mean()
    period_high = result["high"].rolling(period, min_periods=period).max()
    distribution_input = pd.Series(
        np.where(result["high"] >= period_high, high_base, 0.0), index=result.index
    )
    distribution_core = distribution_input.ewm(span=3, adjust=False, min_periods=1).mean()
    distribution_floor = distribution_core.abs().expanding().max().fillna(0) * 0.01
    result["主力出货"] = distribution_core.where(
        distribution_core.abs() > distribution_floor, 0.0
    )

    result["主力吸筹"] = result["主力吸筹"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).clip(0, 100)
    result["主力出货"] = result["主力出货"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).clip(-100, 0)
    return result


def draw_candles(ax, data: pd.DataFrame) -> None:
    x = mdates.date2num(data["time_key"])
    for xi, open_price, close_price, high_price, low_price in zip(
        x, data["open"], data["close"], data["high"], data["low"]
    ):
        color = "#ef5350" if close_price >= open_price else "#26a69a"
        ax.vlines(xi, low_price, high_price, color=color, linewidth=0.65, zorder=2)
        bottom = min(open_price, close_price)
        height = max(abs(close_price - open_price), 0.01)
        ax.add_patch(Rectangle(
            (xi - 0.32, bottom), 0.64, height,
            facecolor=color, edgecolor=color, linewidth=0.4, zorder=2,
        ))


def draw_layered_flame(ax, x, values, colors, label: str) -> None:
    """以高密度插值、轻微平滑和多层渐变绘制柔和火焰。"""
    y = np.asarray(values, dtype=float)
    x_num = mdates.date2num(pd.to_datetime(x))
    valid = np.isfinite(x_num) & np.isfinite(y)
    x_num, y = x_num[valid], y[valid]
    if len(x_num) < 2:
        return

    dense_count = max(int((x_num[-1] - x_num[0]) * 8), len(x_num) * 4)
    dense_x = np.linspace(x_num[0], x_num[-1], dense_count)
    dense_y = np.interp(dense_x, x_num, y)
    grid = np.arange(-10, 11)
    kernel = np.exp(-0.5 * (grid / 3.2) ** 2)
    kernel /= kernel.sum()
    dense_y = np.convolve(dense_y, kernel, mode="same")

    cmap = mcolors.LinearSegmentedColormap.from_list(f"{label}_flame", colors)
    for index, scale in enumerate(np.linspace(1.0, 0.10, 24)):
        progress = index / 23
        ax.fill_between(
            dense_x, 0, dense_y * scale,
            color=cmap(progress), alpha=0.90 if index == 0 else 0.48,
            linewidth=0, antialiased=True,
            label=label if index == 0 else None,
            zorder=2 + index * 0.01,
        )
    ax.plot(dense_x, dense_y, color=colors[0], lw=0.55, alpha=0.55, zorder=3)


def merge_nearby_signal_points(
    points: pd.DataFrame,
    y_column: str,
    keep: str,
    max_gap: int = 5,
) -> pd.DataFrame:
    """合并同类且相距不超过max_gap个交易日的信号，只保留最极端的一处。"""
    if points.empty:
        return points

    ordered = points.sort_index()
    clusters: list[list[int]] = []
    current_cluster: list[int] = []
    previous_index: int | None = None
    for index in ordered.index:
        numeric_index = int(index)
        if previous_index is None or numeric_index - previous_index <= max_gap:
            current_cluster.append(index)
        else:
            clusters.append(current_cluster)
            current_cluster = [index]
        previous_index = numeric_index
    if current_cluster:
        clusters.append(current_cluster)

    selected_indices = []
    for cluster in clusters:
        cluster_points = ordered.loc[cluster]
        selected_index = (
            cluster_points[y_column].idxmin()
            if keep == "min"
            else cluster_points[y_column].idxmax()
        )
        selected_indices.append(selected_index)
    return ordered.loc[selected_indices].sort_values("time_key")


def scatter_signal(
    ax, points: pd.DataFrame, y_column: str, atr_scale: float,
    marker: str, size: float, facecolor: str, edgecolor: str,
    label: str, text_offset: int, annotate: bool,
) -> None:
    if points.empty:
        return
    y = points[y_column] + points["ATR14"] * atr_scale
    ax.scatter(
        points["time_key"], y, marker=marker, s=size,
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.15,
        label=label, zorder=20, clip_on=False,
    )
    if annotate:
        for (_, row), point_y in zip(points.iterrows(), y):
            ax.annotate(
                f"{row['time_key']:%m-%d} {label}",
                xy=(row["time_key"], point_y),
                xytext=(0, text_offset), textcoords="offset points",
                ha="center", va="bottom" if text_offset > 0 else "top",
                fontsize=7.2, color=facecolor, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": facecolor, "lw": 0.55},
                zorder=25, annotation_clip=False,
            )


def select_dates(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return data.loc[
        (data["time_key"] >= start_time) & (data["time_key"] <= end_time)
    ].copy()


def merge_results(
    daily: pd.DataFrame, swing: pd.DataFrame, extreme: pd.DataFrame,
    start: str, end: str,
) -> pd.DataFrame:
    base_columns = [column for column in [
        "time_key", "open", "close", "high", "low", "volume", "turnover"
    ] if column in daily.columns]
    swing_fields = swing[[
        "time_key", "K", "D", "RSI14", "MACD柱", "高抛低吸强度",
        "低吸候选", "低吸确认", "高抛候选", "高抛确认",
    ]].rename(columns={column: f"高抛低吸_{column}" for column in [
        "K", "D", "RSI14", "MACD柱", "高抛低吸强度",
        "低吸候选", "低吸确认", "高抛候选", "高抛确认",
    ]})
    extreme_fields = extreme[[
        "time_key", "区间位置", "底背离", "顶背离", "极端强度",
        "抄底候选", "抄底确认", "逃顶候选", "逃顶确认",
    ]].rename(columns={column: f"抄底逃顶_{column}" for column in [
        "区间位置", "底背离", "顶背离", "极端强度",
        "抄底候选", "抄底确认", "逃顶候选", "逃顶确认",
    ]})
    merged = daily[base_columns].merge(swing_fields, on="time_key", how="left")
    merged = merged.merge(extreme_fields, on="time_key", how="left")
    return select_dates(merged, start, end)


def add_interactive_controls(fig, price_ax, flow_ax, price: pd.DataFrame) -> None:
    """添加两面板同步十字线、坐标数值、OHLC提示和滚轮缩放。"""
    axes = [price_ax, flow_ax]
    dates = mdates.date2num(price["time_key"].to_numpy())
    count = len(price)
    if count < 2:
        return

    def set_all_x_limits(left: float, right: float) -> None:
        """缩放主图日期范围。"""
        for axis in axes:
            axis.set_xlim(left, right, emit=False)

    # 主图日期十字线和水平虚线。
    vertical_lines = [
        axis.axvline(dates[0], color="#dddddd", ls="--", lw=0.75, alpha=0.72,
                     visible=False, zorder=50)
        for axis in axes
    ]
    horizontal_lines = [
        axis.axhline(0, color="#dddddd", ls="--", lw=0.75, alpha=0.72,
                     visible=False, zorder=50)
        for axis in axes
    ]
    date_box = flow_ax.annotate(
        "", xy=(dates[0], 0), xycoords=flow_ax.get_xaxis_transform(),
        xytext=(0, -22), textcoords="offset points", ha="center", va="top",
        color="#ffffff", fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#263238", "ec": "#90a4ae", "alpha": 0.95},
        visible=False, annotation_clip=False, zorder=60,
    )
    value_boxes = []
    for axis in axes:
        value_boxes.append(axis.annotate(
            "", xy=(1, 0), xycoords=axis.get_yaxis_transform(),
            xytext=(5, 0), textcoords="offset points", ha="left", va="center",
            color="#ffffff", fontsize=8,
            bbox={"boxstyle": "round,pad=0.22", "fc": "#263238", "ec": "#90a4ae", "alpha": 0.95},
            visible=False, annotation_clip=False, zorder=60,
        ))
    ohlc_box = price_ax.annotate(
        "", xy=(0.01, 0.98), xycoords="axes fraction", ha="left", va="top",
        color="#ffffff", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.32", "fc": "#111111", "ec": "#777777", "alpha": 0.88},
        visible=False, zorder=60,
    )

    def hide_crosshair():
        for line in vertical_lines + horizontal_lines:
            line.set_visible(False)
        for box in value_boxes:
            box.set_visible(False)
        date_box.set_visible(False)
        ohlc_box.set_visible(False)

    def on_mouse_move(event):
        if event.inaxes not in axes or event.xdata is None or event.ydata is None:
            hide_crosshair()
            fig.canvas.draw_idle()
            return
        index = int(np.abs(dates - event.xdata).argmin())
        snapped_x = dates[index]
        current_axis_index = axes.index(event.inaxes)
        for line in vertical_lines:
            line.set_xdata([snapped_x, snapped_x])
            line.set_visible(True)
        for axis_index, line in enumerate(horizontal_lines):
            line.set_visible(axis_index == current_axis_index)
            if axis_index == current_axis_index:
                line.set_ydata([event.ydata, event.ydata])
        for axis_index, box in enumerate(value_boxes):
            box.set_visible(axis_index == current_axis_index)
            if axis_index == current_axis_index:
                box.xy = (1, event.ydata)
                box.set_text(f"{event.ydata:.2f}")
        date_box.xy = (snapped_x, 0)
        date_box.set_text(pd.Timestamp(price.iloc[index]["time_key"]).strftime("%Y-%m-%d"))
        date_box.set_visible(True)
        if event.inaxes is price_ax:
            row = price.iloc[index]
            change = row["close"] - row["open"]
            ohlc_box.set_text(
                f"{row['time_key']:%Y-%m-%d}  "
                f"开 {row['open']:.2f}  高 {row['high']:.2f}  "
                f"低 {row['low']:.2f}  收 {row['close']:.2f}  "
                f"实体 {change:+.2f}"
            )
            ohlc_box.set_visible(True)
        else:
            ohlc_box.set_visible(False)
        fig.canvas.draw_idle()

    # 鼠标滚轮以指针日期为中心缩放横轴。
    def on_scroll(event):
        if event.inaxes not in axes or event.xdata is None:
            return
        left, right = price_ax.get_xlim()
        factor = 0.82 if event.button == "up" else 1.22
        new_left = event.xdata - (event.xdata - left) * factor
        new_right = event.xdata + (right - event.xdata) * factor
        new_left = max(new_left, dates[0] - 1)
        new_right = min(new_right, dates[-1] + 1)
        if new_right - new_left < 5:
            return
        set_all_x_limits(new_left, new_right)
        fig.canvas.draw_idle()

    motion_id = fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    leave_id = fig.canvas.mpl_connect("figure_leave_event", lambda event: (hide_crosshair(), fig.canvas.draw_idle()))
    scroll_id = fig.canvas.mpl_connect("scroll_event", on_scroll)

    # 防止局部对象被垃圾回收后控件失效。
    fig._interactive_controls = {
        "vertical_lines": vertical_lines,
        "horizontal_lines": horizontal_lines,
        "date_box": date_box,
        "value_boxes": value_boxes,
        "ohlc_box": ohlc_box,
        "callback_ids": (motion_id, leave_id, scroll_id),
    }


def plot_combined(
    daily: pd.DataFrame, swing: pd.DataFrame, extreme: pd.DataFrame,
    main_force: pd.DataFrame, start: str, end: str, output: str,
) -> None:
    plt.style.use("dark_background")
    configure_chinese_font()
    price = select_dates(daily, start, end)
    swing_chart = select_dates(swing, start, end)
    extreme_chart = select_dates(extreme, start, end)
    flow_chart = select_dates(main_force, start, end)
    if price.empty:
        raise ValueError(f"展示区间没有日K：{start}～{end}")

    fig, (ax, flow_ax) = plt.subplots(
        2, 1, figsize=(20, 12), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [3.6, 1.25], "hspace": 0.07},
    )
    draw_candles(ax, price)
    ax.plot(swing_chart["time_key"], swing_chart["EMA10"], "#ffd54f", lw=1.0, label="EMA10")
    ax.plot(swing_chart["time_key"], swing_chart["EMA20"], "#42a5f5", lw=0.9, label="EMA20")
    ax.plot(swing_chart["time_key"], swing_chart["MA60"], "#ab47bc", lw=0.9, label="MA60")
    ax.plot(swing_chart["time_key"], swing_chart["MA120"], "#eeeeee", lw=0.7,
            alpha=0.65, label="MA120")

    # 每一种候选/确认分别聚类；不同信号类型绝不互相合并。
    low_candidates = merge_nearby_signal_points(
        swing_chart[swing_chart["低吸候选"]], "low", "min"
    )
    low_confirmations = merge_nearby_signal_points(
        swing_chart[swing_chart["低吸确认"]], "low", "min"
    )
    high_candidates = merge_nearby_signal_points(
        swing_chart[swing_chart["高抛候选"]], "high", "max"
    )
    high_confirmations = merge_nearby_signal_points(
        swing_chart[swing_chart["高抛确认"]], "high", "max"
    )
    bottom_candidates = merge_nearby_signal_points(
        extreme_chart[extreme_chart["抄底候选"]], "low", "min"
    )
    bottom_confirmations = merge_nearby_signal_points(
        extreme_chart[extreme_chart["抄底确认"]], "low", "min"
    )
    top_candidates = merge_nearby_signal_points(
        extreme_chart[extreme_chart["逃顶候选"]], "high", "max"
    )
    top_confirmations = merge_nearby_signal_points(
        extreme_chart[extreme_chart["逃顶确认"]], "high", "max"
    )

    # 只绘制标记，不显示标记文字。
    scatter_signal(ax, low_candidates, "low", -0.30,
                   "o", 62, "#ffee58", "#ff8f00", "低吸候选", -20, False)
    scatter_signal(ax, low_confirmations, "low", -0.60,
                   "^", 135, "#ffca28", "#ff3d00", "低吸确认", -34, False)
    scatter_signal(ax, high_candidates, "high", 0.30,
                   "o", 62, "#64b5f6", "#0d47a1", "高抛候选", 20, False)
    scatter_signal(ax, high_confirmations, "high", 0.60,
                   "v", 135, "#42a5f5", "#002171", "高抛确认", -34, False)

    scatter_signal(ax, bottom_candidates, "low", -0.92,
                   "s", 72, "#66bb6a", "#1b5e20", "抄底候选", -50, False)
    scatter_signal(ax, bottom_confirmations, "low", -1.28,
                   "*", 210, "#00e676", "#004d40", "抄底确认", -66, False)
    scatter_signal(ax, top_candidates, "high", 0.92,
                   "s", 72, "#ff8a80", "#b71c1c", "逃顶候选", 50, False)
    scatter_signal(ax, top_confirmations, "high", 1.28,
                   "*", 210, "#ff5252", "#7f0000", "逃顶确认", -66, False)

    draw_layered_flame(
        flow_ax, flow_chart["time_key"], flow_chart["主力吸筹"],
        ["#e53900", "#ff6d00", "#ffb300", "#fff8b0"], "主力吸筹",
    )
    draw_layered_flame(
        flow_ax, flow_chart["time_key"], flow_chart["主力出货"],
        ["#0d47a1", "#1976d2", "#42a5f5", "#d7efff"], "主力出货",
    )
    flow_ax.axhline(0, color="#aaaaaa", linewidth=0.6)
    max_accumulation = max(float(flow_chart["主力吸筹"].max()), 1.0)
    min_distribution = min(float(flow_chart["主力出货"].min()), -1.0)
    flow_padding = max(max_accumulation - min_distribution, 1.0) * 0.08
    flow_ax.set_ylim(
        min_distribution - flow_padding,
        max_accumulation + flow_padding,
    )
    flow_ax.set_ylabel("Main Force Flow")
    flow_ax.grid(alpha=0.14)
    flow_ax.legend(loc="upper left", ncol=2, fontsize=8)

    date_from = price["time_key"].iloc[0].strftime("%Y-%m-%d")
    date_to = price["time_key"].iloc[-1].strftime("%Y-%m-%d")
    ax.set_title(f"高抛低吸 + 抄底逃顶 联合信号图  {date_from}～{date_to}", fontsize=15)
    ax.set_ylabel("价格")
    ax.grid(alpha=0.13)
    # 主图顶部预留空间，让左上角图例与K线、信号标记保持距离。
    visible_low = float(price["low"].min())
    visible_high = float(price["high"].max())
    visible_range = max(visible_high - visible_low, abs(visible_high) * 0.01, 0.01)
    ax.set_ylim(
        visible_low - visible_range * 0.08,
        visible_high + visible_range * 0.32,
    )
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    ax.tick_params(axis="x", labelbottom=False)
    span = max((price["time_key"].iloc[-1] - price["time_key"].iloc[0]).days, 1)
    interval = 1 if span <= 370 else (2 if span <= 800 else 4)
    flow_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    flow_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    padding = pd.Timedelta(days=max(span * 0.01, 2))
    flow_ax.set_xlim(price["time_key"].iloc[0] - padding,
                     price["time_key"].iloc[-1] + padding)
    fig.autofmt_xdate(rotation=25)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.93, bottom=0.07)
    add_interactive_controls(fig, ax, flow_ax, price)
    fig.savefig(output, bbox_inches="tight")
    print(f"联合图片已保存：{output}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="高抛低吸与抄底逃顶单文件联合版")
    parser.add_argument("--code", default="HK.800700", help="富途证券代码")
    parser.add_argument("--start", default="2025-01-02", help="图片开始日期")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                        help="图片结束日期，默认今天")
    parser.add_argument("--warmup-days", type=int, default=365,
                        help="start之前额外拉取的自然日数，默认365")
    parser.add_argument("--main-force-period", type=int, default=34,
                        help="主力吸筹/出货阶段新高新低观察周期，默认34")
    parser.add_argument("--host", default="127.0.0.1", help="Futu OpenD地址")
    parser.add_argument("--port", type=int, default=11111, help="Futu OpenD端口")
    parser.add_argument("--output", default="高抛低吸_抄底逃顶_单文件联合图.png")
    parser.add_argument("--csv", default="高抛低吸_抄底逃顶_单文件联合数据.csv")
    args = parser.parse_args()

    display_start, display_end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if display_start > display_end:
        raise ValueError("start不能晚于end")
    if args.warmup_days < 0:
        raise ValueError("warmup-days不能小于0")
    fetch_start = (display_start - pd.Timedelta(days=args.warmup_days)).strftime("%Y-%m-%d")

    daily = fetch_futu_daily_kline(
        stock_code=args.code, start=fetch_start, end=args.end,
        host=args.host, port=args.port,
    )
    # 两套识别函数独立执行，只共享同一份原始日K。
    swing_result = calculate_buy_low_sell_high(daily.copy())
    extreme_result = calculate_bottom_top_escape(daily.copy())
    main_force_result = calculate_main_force_flow(
        daily.copy(), period=args.main_force_period
    )
    merged = merge_results(daily, swing_result, extreme_result, args.start, args.end)
    merged.to_csv(args.csv, index=False, encoding="utf-8-sig")

    print(f"Futu拉取区间：{fetch_start}～{args.end}（预热{args.warmup_days}天）")
    print(f"联合数据：{args.csv}")
    print(
        f"高抛低吸：低吸候选{int(merged['高抛低吸_低吸候选'].sum())}、"
        f"低吸确认{int(merged['高抛低吸_低吸确认'].sum())}、"
        f"高抛候选{int(merged['高抛低吸_高抛候选'].sum())}、"
        f"高抛确认{int(merged['高抛低吸_高抛确认'].sum())}"
    )
    print(
        f"抄底逃顶：抄底候选{int(merged['抄底逃顶_抄底候选'].sum())}、"
        f"抄底确认{int(merged['抄底逃顶_抄底确认'].sum())}、"
        f"逃顶候选{int(merged['抄底逃顶_逃顶候选'].sum())}、"
        f"逃顶确认{int(merged['抄底逃顶_逃顶确认'].sum())}"
    )
    plot_combined(
        daily, swing_result, extreme_result, main_force_result,
        args.start, args.end, args.output,
    )


if __name__ == "__main__":
    main()
