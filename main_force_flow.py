"""恒生科技 Main Force Flow 主力吸筹/出货指标（单文件完整版）。

依赖：
    pip install futu-api pandas numpy matplotlib

运行前请启动 Futu OpenD，默认连接 127.0.0.1:11111。
示例：
    python main_force_flow.py
    python main_force_flow.py --code HK.800700 --start 2023-01-02 --end 2026-10-05
"""
from __future__ import annotations

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def configure_chinese_font() -> str | None:
    """按实际中文字形检测字体，而不是只凭字体名称猜测。"""
    preferred_names = [
        "Microsoft YaHei", "Microsoft JhengHei", "SimHei", "SimSun",
        "PingFang SC", "Heiti SC", "Noto Sans CJK SC", "Noto Sans CJK JP",
        "Source Han Sans SC", "Source Han Sans CN", "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    fonts = list(font_manager.fontManager.ttflist)
    fonts.sort(key=lambda item: (
        preferred_names.index(item.name) if item.name in preferred_names else len(preferred_names),
        item.name,
    ))

    # 检查“中、吸、货”三个字，避免选到同名字但不含中文字形的字体文件。
    required_codepoints = [ord("中"), ord("吸"), ord("货")]
    selected = None
    for font in fonts:
        try:
            cmap = ft2font.FT2Font(font.fname).get_charmap()
            if all(codepoint in cmap for codepoint in required_codepoints):
                selected = font
                break
        except (RuntimeError, OSError):
            continue

    if selected:
        # 同时设置 family 和 sans-serif；必须在 plt.style.use() 之后调用。
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected.name, "DejaVu Sans"]
        print(f"Matplotlib 中文字体：{selected.name} ({selected.fname})")
        result = selected.name
    else:
        print("警告：系统字体中没有找到包含中文字形的字体。")
        print("Windows请安装/启用微软雅黑；Linux请安装 fonts-noto-cjk。")
        result = None

    plt.rcParams["axes.unicode_minus"] = False
    return result


def fetch_futu_daily_kline(
    stock_code: str,
    start: str,
    end: str,
    host: str = "127.0.0.1",
    port: int = 11111,
) -> pd.DataFrame:
    """从 Futu OpenD 拉取完整日K，并自动处理分页。"""
    try:
        from futu import KLType, OpenQuoteContext, RET_OK
    except ImportError as exc:
        raise RuntimeError("缺少 futu-api，请先执行：pip install futu-api") from exc

    quote_ctx = OpenQuoteContext(host=host, port=port)
    pages: list[pd.DataFrame] = []
    page_req_key = None

    try:
        while True:
            ret, page, page_req_key = quote_ctx.request_history_kline(
                code=stock_code,
                start=start,
                end=end,
                ktype=KLType.K_DAY,
                page_req_key=page_req_key,
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
        raise RuntimeError(
            f"富途没有返回日K：code={stock_code}, start={start}, end={end}"
        )

    data = pd.concat(pages, ignore_index=True)
    required = ["time_key", "open", "close", "high", "low"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"富途返回数据缺少字段：{missing}")

    data["time_key"] = pd.to_datetime(data["time_key"], errors="coerce")
    for column in ["open", "close", "high", "low", "volume", "turnover"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = (
        data.dropna(subset=required)
        .sort_values("time_key")
        .drop_duplicates("time_key", keep="last")
        .reset_index(drop=True)
    )
    return data


def tdx_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """复刻通达信 SMA(X,N,M)，并非 pandas rolling mean。

    递推公式：Y_t = (M * X_t + (N-M) * Y_{t-1}) / N
    """
    if n <= 0 or not 0 < m <= n:
        raise ValueError("tdx_sma 要求 n > 0 且 0 < m <= n")

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    previous = np.nan

    for index, value in enumerate(values):
        if np.isnan(value):
            continue
        if np.isnan(previous):
            previous = value
        else:
            previous = (m * value + (n - m) * previous) / n
        result[index] = previous

    return pd.Series(result, index=series.index)


def calculate_main_force_flow(data: pd.DataFrame, period: int = 34) -> pd.DataFrame:
    """计算截图同类 Main Force Flow 吸筹/出货指标。

    逻辑：
    1. 使用前一交易日 OHLC 均价作为比较基准；
    2. 触及 period 日新低时计算吸筹脉冲；
    3. 触及 period 日新高时计算出货脉冲；
    4. 使用通达信 SMA 和 EMA 平滑；
    5. 仅保留核心值继续增强的日期，其余日期置零。

    注意：该公式不使用真实主力资金流，只是价格极值行为指标。
    """
    if period < 2:
        raise ValueError("period 必须大于等于2")

    result = data.sort_values("time_key").reset_index(drop=True).copy()
    for column in ["open", "close", "high", "low"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    previous_average_price = (
        (result["low"] + result["open"] + result["close"] + result["high"]) / 4
    ).shift(1)

    # ------------------------- 吸筹侧 -------------------------
    low_difference = result["low"] - previous_average_price
    low_numerator = tdx_sma(low_difference.abs(), period, 1)

    # 常见公式在吸筹侧使用13日分母，因此吸筹火焰一般比出货火焰更高。
    low_denominator = tdx_sma(low_difference.clip(lower=0), 13, 1)
    low_denominator = low_denominator.replace(0, np.nan)
    low_ratio = low_numerator / low_denominator
    low_base = low_ratio.ewm(span=period, adjust=False, min_periods=1).mean()

    period_low = result["low"].rolling(period, min_periods=period).min()
    accumulation_trigger = result["low"] <= period_low
    accumulation_input = pd.Series(
        np.where(accumulation_trigger, low_base, 0.0), index=result.index
    )
    accumulation_core = accumulation_input.ewm(
        span=3, adjust=False, min_periods=1
    ).mean()

    result["主力吸筹"] = accumulation_core.where(
        accumulation_core > accumulation_core.shift(1), 0.0
    )

    # ------------------------- 出货侧 -------------------------
    high_difference = result["high"] - previous_average_price
    high_numerator = tdx_sma(high_difference.abs(), period, 1)

    # MIN(HIGH-REF_PRICE, 0) 保留负号，因此出货序列位于零轴下方。
    high_denominator = tdx_sma(high_difference.clip(upper=0), period, 1)
    high_denominator = high_denominator.replace(0, np.nan)
    high_ratio = high_numerator / high_denominator
    high_base = high_ratio.ewm(span=period, adjust=False, min_periods=1).mean()

    period_high = result["high"].rolling(period, min_periods=period).max()
    distribution_trigger = result["high"] >= period_high
    distribution_input = pd.Series(
        np.where(distribution_trigger, high_base, 0.0), index=result.index
    )
    distribution_core = distribution_input.ewm(
        span=3, adjust=False, min_periods=1
    ).mean()

    result["主力出货"] = distribution_core.where(
        distribution_core > distribution_core.shift(1), 0.0
    )

    result["主力吸筹"] = (
        result["主力吸筹"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0, upper=100)
    )
    result["主力出货"] = (
        result["主力出货"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=-100, upper=0)
    )

    result["MA20"] = result["close"].rolling(20).mean()
    result["MA60"] = result["close"].rolling(60).mean()
    return result


def draw_candles(ax, data: pd.DataFrame) -> None:
    """使用 Matplotlib 绘制日K蜡烛图。"""
    x_values = mdates.date2num(data["time_key"])
    candle_width = 0.65

    for x, open_price, close_price, high_price, low_price in zip(
        x_values,
        data["open"],
        data["close"],
        data["high"],
        data["low"],
    ):
        color = "#ef5350" if close_price >= open_price else "#26a69a"
        ax.vlines(x, low_price, high_price, color=color, linewidth=0.65, alpha=0.9)
        body_bottom = min(open_price, close_price)
        body_height = max(abs(close_price - open_price), 0.5)
        ax.add_patch(
            Rectangle(
                (x - candle_width / 2, body_bottom),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.4,
            )
        )


def draw_layered_flame(ax, x, values, colors, label: str) -> None:
    """通过四层填充模拟截图中的火焰效果。"""
    y = np.asarray(values, dtype=float)
    layers = [1.00, 0.72, 0.46, 0.22]

    for index, (scale, color) in enumerate(zip(layers, colors)):
        ax.fill_between(
            x,
            0,
            y * scale,
            color=color,
            alpha=0.96,
            linewidth=0,
            label=label if index == 0 else None,
        )


def plot_main_force_flow(
    data: pd.DataFrame,
    display_start: str,
    display_end: str,
    output: str | None = "恒生科技_Main Force Flow复刻图.png",
) -> None:
    """按用户指定日期范围绘图、可选保存图片，并通过 plt.show() 显示。"""
    plt.style.use("dark_background")
    # style.use 会覆盖字体 rcParams，因此中文字体必须在它之后设置。
    configure_chinese_font()
    start_time = pd.Timestamp(display_start)
    end_time = pd.Timestamp(display_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    chart_data = data.loc[
        (data["time_key"] >= start_time) & (data["time_key"] <= end_time)
    ].copy()
    if chart_data.empty:
        raise ValueError(
            f"指定展示区间没有日K数据：{display_start} ～ {display_end}"
        )
    fig = plt.figure(figsize=(18, 10), dpi=150)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.06)
    price_ax = fig.add_subplot(grid[0])
    indicator_ax = fig.add_subplot(grid[1], sharex=price_ax)

    draw_candles(price_ax, chart_data)
    price_ax.plot(
        chart_data["time_key"], chart_data["MA20"],
        color="#ffd54f", linewidth=1.0, label="MA20"
    )
    price_ax.plot(
        chart_data["time_key"], chart_data["MA60"],
        color="#42a5f5", linewidth=1.0, label="MA60"
    )

    date_from = chart_data["time_key"].iloc[0].strftime("%Y-%m-%d")
    date_to = chart_data["time_key"].iloc[-1].strftime("%Y-%m-%d")
    price_ax.set_title(f"恒生科技 Main Force Flow 复刻  {date_from} ～ {date_to}")
    price_ax.set_ylabel("指数点位")
    price_ax.grid(alpha=0.14)
    price_ax.legend(loc="upper left", ncol=2)
    price_ax.tick_params(axis="x", labelbottom=False)

    draw_layered_flame(
        indicator_ax,
        chart_data["time_key"],
        chart_data["主力吸筹"],
        ["#e53900", "#ff6d00", "#ffb300", "#fff8b0"],
        "主力吸筹",
    )
    draw_layered_flame(
        indicator_ax,
        chart_data["time_key"],
        chart_data["主力出货"],
        ["#0d47a1", "#1976d2", "#42a5f5", "#d7efff"],
        "主力出货",
    )

    indicator_ax.axhline(0, color="#aaaaaa", linewidth=0.6)
    max_accumulation = max(float(chart_data["主力吸筹"].max()), 1.0)
    min_distribution = min(float(chart_data["主力出货"].min()), -1.0)
    y_padding = max(max_accumulation - min_distribution, 1.0) * 0.08
    indicator_ax.set_ylim(
        min_distribution - y_padding,
        max_accumulation + y_padding,
    )
    indicator_ax.set_ylabel("Main Force Flow")
    indicator_ax.grid(alpha=0.14)
    indicator_ax.legend(loc="upper left", ncol=2)

    date_span = max(
        (chart_data["time_key"].iloc[-1] - chart_data["time_key"].iloc[0]).days,
        1,
    )
    month_interval = 1 if date_span <= 370 else (2 if date_span <= 800 else 4)
    indicator_ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=month_interval)
    )
    indicator_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    x_padding = pd.Timedelta(days=max(date_span * 0.01, 2))
    indicator_ax.set_xlim(
        chart_data["time_key"].iloc[0] - x_padding,
        chart_data["time_key"].iloc[-1] + x_padding,
    )

    fig.autofmt_xdate(rotation=25)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.10)

    if output:
        fig.savefig(output, bbox_inches="tight")
        print(f"图片已保存：{output}")

    # 按用户要求使用 show，不再调用 plt.close(fig)。
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 Futu 拉取日K并显示 Main Force Flow 主力吸筹/出货指标"
    )
    parser.add_argument("--code", default="HK.800700", help="富途证券代码")
    parser.add_argument("--start", default="2023-01-02", help="开始日期 YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="结束日期 YYYY-MM-DD，默认今天",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Futu OpenD 地址")
    parser.add_argument("--port", type=int, default=11111, help="Futu OpenD 端口")
    parser.add_argument("--period", type=int, default=34, help="新高/新低观察周期")
    parser.add_argument(
        "--warmup-days", type=int, default=365,
        help="在start之前额外拉取的自然日数，仅用于指标预热，默认365天"
    )
    parser.add_argument("--output", default="恒生科技_Main Force Flow复刻图.png")
    parser.add_argument("--csv", default="恒生科技_Main Force Flow复刻数据.csv")
    args = parser.parse_args()

    display_start = pd.Timestamp(args.start)
    display_end = pd.Timestamp(args.end)
    if display_start > display_end:
        raise ValueError("start 不能晚于 end")
    if args.warmup_days < 0:
        raise ValueError("warmup-days 不能小于0")
    fetch_start = (display_start - pd.Timedelta(days=args.warmup_days)).strftime("%Y-%m-%d")

    daily_data = fetch_futu_daily_kline(
        stock_code=args.code,
        start=fetch_start,
        end=args.end,
        host=args.host,
        port=args.port,
    )
    # 在包含预热期的完整数据上计算，确保展示区间第一天的SMA/EMA已经稳定。
    result = calculate_main_force_flow(daily_data, period=args.period)
    display_result = result.loc[
        (result["time_key"] >= display_start) & (result["time_key"] <= display_end)
    ].copy()
    display_result.to_csv(args.csv, index=False, encoding="utf-8-sig")

    print(
        f"Futu拉取区间：{fetch_start} ～ {args.end} "
        f"（含{args.warmup_days}天预热）"
    )
    print(
        f"图片/CSV展示区间：{args.start} ～ {args.end}，"
        f"实际包含 {len(display_result)} 根日K"
    )
    print(f"指标数据已保存：{args.csv}")
    print(
        display_result[["time_key", "close", "主力吸筹", "主力出货"]]
        .tail(20)
        .to_string(index=False)
    )

    plot_main_force_flow(
        data=result,
        display_start=args.start,
        display_end=args.end,
        output=args.output,
    )


if __name__ == "__main__":
    main()
