from __future__ import annotations

"""
Event-anchored directional Granger tests for consensus events based on 0808 PDF Instructions.
Modified: 共识窗口时间改为手动输入，不再自动从CSV中合并窗口。
"""

import hashlib
import math
import os
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from scipy.stats import f as f_dist

# ==========================================
# 代码内参数配置
# ==========================================
DATA_DIR = Path(os.getenv("PREMADE_DISHES_DIR", "/root/autodl-tmp/预制菜"))
INPUT_EXCEL = str(Path(os.getenv("PREMADE_DISHES_OUTPUT", str(DATA_DIR / "预制菜output.xlsx"))))
ANALYSIS_DIR = DATA_DIR / "advanced_analysis"
CONSENSUS_SUMMARY = str(ANALYSIS_DIR / "3天刻度_全情感共识汇总数据.csv")
CONSENSUS_DETAILS = str(ANALYSIS_DIR / "3天刻度_共识窗口全部议题明细.csv")
OUTPUT_DIR = str(DATA_DIR / "event_anchored_granger_results")

# 指数衰减存量参数配置
ENABLE_EXPONENTIAL_DECAY = False
DECAY_GAMMA = 0.975
MAX_LAG_LIMIT = 1

# 显著性阈值及 Bootstrap 重抽样配置
N_BOOTSTRAP = 100 # Bootstrap 重抽样次数
BOOTSTRAP_SEED = 555

PUBLIC_THEMES: list[str] = []
SOURCE_THEMES: list[str] = []
ALIAS_MAP: dict[tuple[str, str], str] = {}

MANUAL_WINDOWS: list[tuple[str, str]] = []

FULL_PAIR_START: str | None = None
FULL_PAIR_END: str | None = None

CONNECTED_EVENT_MODE = "时间重叠或首尾相接合并"
INVALID_SUMMARY_PATTERN = "未提及|未涉及|无法提炼|无法概括|无法得出|没有涉及|不涉及"
PUBLIC_DUPLICATE_SIMILARITY = 0.75
DATE_PATTERN = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")


def run_full_pair_granger_tests(
    series_store: DailyTopicSeries,
    start: str,
    end: str,
    output_csv: str,
) -> None:
    """
    全议题对全局 VAR 分析。
    对每个指标：议题、正面情感、负面情感、平均情感
    分别分析政府/媒体/公众视角下的议题对。
    """
    print("[*] 正在执行全议题对全局 VAR 分析 ...")

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    indicators = DailyTopicSeries.INDICATORS
    all_results: list[dict[str, object]] = []
    pair_index = 0
    total_pairs = len(indicators) * 4 * len(PUBLIC_THEMES) * len(SOURCE_THEMES)

    def add_result(
        indicator: str,
        relation_type: str,
        cause_actor: str,
        effect_actor: str,
        cause_topic: str,
        effect_topic: str,
        cause_alias: str,
        effect_alias: str,
    ) -> None:
        nonlocal pair_index
        pair_index += 1
        print(
            f"    [{pair_index}/{total_pairs}] {indicator} | {relation_type}: "
            f"{cause_alias} -> {effect_alias}"
        )

        x_series = series_store.get(cause_actor, cause_topic, indicator).loc[start_ts:end_ts]
        y_series = series_store.get(effect_actor, effect_topic, indicator).loc[start_ts:end_ts]

        # 平稳化处理
        x_clean, _ = _stationarize(x_series)
        y_clean, _ = _stationarize(y_series)

        df = pd.concat([x_clean, y_clean], axis=1).dropna()
        if len(df) <= 10:
            row = {
                "分析指标": indicator,
                "分析视角": cause_actor,
                "议题X": cause_alias,
                "议题Y": effect_alias,
                "全局因果关系": "样本不足",
                "X→Y的具体时间段": "无",
                "Y→X的具体时间段": "无",
                "全局VAR最优滞后阶数": "—",
                "全局P值_X→Y": "—",
                "全局P值_Y→X": "—",
                "第十期方差分解_X对Y(%)": "—",
                "第十期方差分解_Y对X(%)": "—",
                "第十期累计脉冲响应_X→Y": "—",
                "第十期累计脉冲响应_Y→X": "—",
            }
            all_results.append(row)
            return

        x_work = df.iloc[:, 0]
        y_work = df.iloc[:, 1]

        diag = compute_var_diagnostics(x_work, y_work)
        if not diag:
            row = {
                "分析指标": indicator,
                "分析视角": cause_actor,
                "议题X": cause_alias,
                "议题Y": effect_alias,
                "全局因果关系": "模型无法估计",
                "X→Y的具体时间段": "无",
                "Y→X的具体时间段": "无",
                "全局VAR最优滞后阶数": "—",
                "全局P值_X→Y": "—",
                "全局P值_Y→X": "—",
                "第十期方差分解_X对Y(%)": "—",
                "第十期方差分解_Y对X(%)": "—",
                "第十期累计脉冲响应_X→Y": "—",
                "第十期累计脉冲响应_Y→X": "—",
            }
            all_results.append(row)
            return

        best_lag = diag["best_lag"]
        p_xy = diag["p_xy"]
        p_yx = diag["p_yx"]

        periods_xy = find_significant_periods(x_work, y_work, "X->Y", best_lag)
        periods_yx = find_significant_periods(x_work, y_work, "Y->X", best_lag)

        global_causality = _format_global_causality(p_xy, p_yx)

        if pd.isna(p_xy) or pd.isna(p_yx):
            global_p_str = "—"
        else:
            global_p_str = f"X→Y p={p_xy:.3f}; Y→X p={p_yx:.3f}"

        fevd_str = (
            f"X对Y方差贡献: {diag['var_y_by_x']:.2f}%; "
            f"Y对X方差贡献: {diag['var_x_by_y']:.2f}%"
        )
        irf_str = (
            f"X→Y累计响应: {diag['cum_xy']:.4f}; "
            f"Y→X累计响应: {diag['cum_yx']:.4f}"
        )

        row = {
            "分析指标": indicator,
            "分析视角": cause_actor,
            "议题X": cause_alias,
            "议题Y": effect_alias,
            "全局因果关系": global_causality,
            "X→Y的具体时间段": _format_periods(periods_xy),
            "Y→X的具体时间段": _format_periods(periods_yx),
            "全局VAR最优滞后阶数": best_lag,
            "全局P值_X→Y": f"{p_xy:.3f}" if not pd.isna(p_xy) else "—",
            "全局P值_Y→X": f"{p_yx:.3f}" if not pd.isna(p_yx) else "—",
            "第十期方差分解_X对Y(%)": f"{diag['var_y_by_x']:.2f}" if not pd.isna(diag['var_y_by_x']) else "—",
            "第十期方差分解_Y对X(%)": f"{diag['var_x_by_y']:.2f}" if not pd.isna(diag['var_x_by_y']) else "—",
            "第十期累计脉冲响应_X→Y": f"{diag['cum_xy']:.4f}" if not pd.isna(diag['cum_xy']) else "—",
            "第十期累计脉冲响应_Y→X": f"{diag['cum_yx']:.4f}" if not pd.isna(diag['cum_yx']) else "—",
        }
        all_results.append(row)

    for indicator in indicators:
        # 1. 政府 -> 公众
        for gov_topic in SOURCE_THEMES:
            for pub_topic in PUBLIC_THEMES:
                x_alias = ALIAS_MAP.get(("政府", gov_topic), f"政府:{gov_topic}")
                y_alias = f"公众:{pub_topic}"
                add_result(
                    indicator,
                    "政府->公众",
                    "政府",
                    "公众",
                    gov_topic,
                    pub_topic,
                    x_alias,
                    y_alias,
                )

        # 2. 媒体 -> 公众
        for med_topic in SOURCE_THEMES:
            for pub_topic in PUBLIC_THEMES:
                x_alias = ALIAS_MAP.get(("媒体", med_topic), f"媒体:{med_topic}")
                y_alias = f"公众:{pub_topic}"
                add_result(
                    indicator,
                    "媒体->公众",
                    "媒体",
                    "公众",
                    med_topic,
                    pub_topic,
                    x_alias,
                    y_alias,
                )

        # 3. 公众 -> 政府
        for pub_topic in PUBLIC_THEMES:
            for gov_topic in SOURCE_THEMES:
                x_alias = f"公众:{pub_topic}"
                y_alias = ALIAS_MAP.get(("政府", gov_topic), f"政府:{gov_topic}")
                add_result(
                    indicator,
                    "公众->政府",
                    "公众",
                    "政府",
                    pub_topic,
                    gov_topic,
                    x_alias,
                    y_alias,
                )

        # 4. 公众 -> 媒体
        for pub_topic in PUBLIC_THEMES:
            for med_topic in SOURCE_THEMES:
                x_alias = f"公众:{pub_topic}"
                y_alias = ALIAS_MAP.get(("媒体", med_topic), f"媒体:{med_topic}")
                add_result(
                    indicator,
                    "公众->媒体",
                    "公众",
                    "媒体",
                    pub_topic,
                    med_topic,
                    x_alias,
                    y_alias,
                )

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[*] 全议题对全局 VAR 分析结果已输出到 {output_csv}，共 {len(df_all)} 行")


def compute_exponential_decay_stock(series: pd.Series, gamma: float) -> pd.Series:
    """计算指数衰减存量信号 S_t = gamma * S_{t-1} + X_t"""
    values = series.to_numpy(dtype=float)
    stock = np.zeros_like(values)
    noise = np.random.normal(0, 0.01, size=len(values))
    if len(values) > 0:
        stock[0] = values[0] + noise[0]
        for t in range(1, len(values)):
            stock[t] = gamma * stock[t - 1] + values[t] + noise[t]
    return pd.Series(stock, index=series.index, name=series.name)


def _to_datetime_mixed(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def _parse_source_date(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.replace(r"(?:发布于|编辑于)\s*", "", regex=True).str.strip()
    extracted = raw.str.extract(
        r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}(?:\s+\d{1,2}:\d{1,2})?)"
    )[0]
    extracted = (
        extracted.str.replace("年", "-", regex=False)
        .str.replace("月", "-", regex=False)
        .str.replace("日", "", regex=False)
    )
    parsed = _to_datetime_mixed(extracted)
    direct = _to_datetime_mixed(series)
    return parsed.fillna(direct)


def _normalize_actor(value: object) -> str | None:
    text = str(value).lower().strip()
    if text in {"public", "公众", "普通用户", "群众"}:
        return "公众"
    if text in {"media", "媒体", "新闻媒体", "自媒体"}:
        return "媒体"
    if text in {"government", "gov", "政府", "政务", "政务机构"}:
        return "政府"
    return None


def _sentiment_polarity(score: float) -> str:
    if score > 0.0: return "正向情感圈层"
    if score < -0.0: return "负向情感圈层"
    return "中立情感圈层"


def _extract_themes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    cleaned = re.sub(r"\([0-9.]+\)", "", str(value))
    return list(dict.fromkeys(
        item.strip()
        for item in cleaned.split("|")
        if item.strip() and item.strip() != "无匹配"
    ))


def _standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "source": "账号分类",
        "distilled_text": "蒸馏总结",
        "ai_summary": "蒸馏总结",
        "评论": "蒸馏总结",
        "time": "时间",
    }
    rename_map = {
        old: new
        for old, new in aliases.items()
        if old in data.columns and new not in data.columns
    }
    return data.rename(columns=rename_map).copy()


def _normalize_text(value: object) -> str:
    return re.sub(r"[\W_]+", "", str(value).strip().lower(), flags=re.UNICODE)


def _remove_similar_public_rows(data: pd.DataFrame) -> pd.DataFrame:
    if "蒸馏总结" not in data.columns or "账号分类" not in data.columns:
        return data.copy()

    work = data.copy()
    actor = work["账号分类"].map(_normalize_actor)
    public = work.loc[actor.eq("公众")].copy()
    other = work.loc[~actor.eq("公众")].copy()
    public["蒸馏总结"] = public["蒸馏总结"].fillna("").astype(str).str.strip()
    public = public[
        public["蒸馏总结"].ne("")
        & ~public["蒸馏总结"].str.contains(INVALID_SUMMARY_PATTERN, regex=True, na=False)
    ].copy()
    public["_normalized_text"] = public["蒸馏总结"].map(_normalize_text)
    public = public[public["_normalized_text"].ne("")]
    public = public.drop_duplicates(subset=["_normalized_text"], keep="first")

    if len(public) > 1:
        texts = public["_normalized_text"].tolist()
        duplicate_positions: set[int] = set()
        for current_pos, current_text in enumerate(texts):
            if current_pos in duplicate_positions:
                continue
            for candidate_pos in range(current_pos + 1, len(texts)):
                similarity = SequenceMatcher(None, current_text, texts[candidate_pos]).ratio()
                if similarity > PUBLIC_DUPLICATE_SIMILARITY:
                    duplicate_positions.add(candidate_pos)
        if duplicate_positions:
            public = public.drop(public.index[list(duplicate_positions)])

    public = public.drop(columns=["_normalized_text"], errors="ignore")
    return pd.concat([public, other], axis=0).sort_index()


def _configure_theme_catalog(data: pd.DataFrame) -> None:
    global PUBLIC_THEMES, SOURCE_THEMES, ALIAS_MAP
    PUBLIC_THEMES = sorted(
        data.loc[data["主体"].eq("公众"), "议题"].dropna().astype(str).unique()
    )
    SOURCE_THEMES = sorted(
        data.loc[data["主体"].isin(["媒体", "政府"]), "议题"]
        .dropna().astype(str).unique()
    )
    ALIAS_MAP = {
        (actor, topic): f"{topic}({actor})"
        for actor in ("政府", "媒体")
        for topic in sorted(
            data.loc[data["主体"].eq(actor), "议题"].dropna().astype(str).unique()
        )
    }


def load_topic_data(file_path: str) -> pd.DataFrame:
    data = _standardize_columns(pd.read_excel(file_path))
    _require_columns(
        data,
        {"账号分类", "蒸馏总结", "匹配的主题标签", "情感得分"},
        "预制菜output.xlsx",
    )
    data = _remove_similar_public_rows(data)

    date_col = next(
        (column for column in ("日期", "时间", "创建时间", "发布时间", "parsed_time") if column in data.columns),
        None,
    )
    if date_col is None:
        raise ValueError("输入文件缺少可识别的日期列。")

    data["日期"] = _parse_source_date(data[date_col])
    data = data.dropna(subset=["日期"]).copy()
    data["日期"] = data["日期"].dt.normalize()
    data["主体"] = data["账号分类"].map(_normalize_actor)
    data = data.dropna(subset=["主体"]).copy()

    data["情感得分"] = pd.to_numeric(data.get("情感得分", 0.0), errors="coerce").fillna(0.0)
    data["情感极性"] = data["情感得分"].map(_sentiment_polarity)
    data["themes_list"] = data["匹配的主题标签"].map(_extract_themes)
    data = data[data["themes_list"].map(bool)].explode("themes_list")
    data = data.rename(columns={"themes_list": "议题"})

    data = data.reset_index(drop=True)
    _configure_theme_catalog(data)
    if not PUBLIC_THEMES:
        raise ValueError("输入文件中没有可用的公众主题。")
    if not SOURCE_THEMES:
        raise ValueError("输入文件中没有可用的媒体/政府主题。")
    print(f"[*] 本次事件公众主题：{PUBLIC_THEMES}")
    print(f"[*] 本次事件媒体/政府主题：{SOURCE_THEMES}")
    return data


def _require_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def _is_yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"是", "true", "1", "yes"})


def _extract_dates(values: pd.Series) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for value in values.dropna().astype(str):
        for matched in DATE_PATTERN.findall(value):
            parsed = pd.to_datetime(matched.replace("/", "-"), errors="coerce")
            if pd.notna(parsed):
                dates.append(pd.Timestamp(parsed).normalize())
    return sorted(set(dates))


def _topic_sort_key(topic: str) -> tuple[int, str]:
    try:
        return PUBLIC_THEMES.index(topic), topic
    except ValueError:
        return len(PUBLIC_THEMES), topic


def _select_peak_row(period_summary: pd.DataFrame) -> pd.Series:
    peak_rows = period_summary[
        _is_yes(period_summary.get("是否为该周期最高潮", pd.Series(False, index=period_summary.index)))
    ]
    candidates = peak_rows if not peak_rows.empty else period_summary
    scores = pd.to_numeric(candidates["共识分"], errors="coerce")
    if scores.notna().any():
        candidates = candidates.loc[scores.eq(scores.max())]
    return candidates.sort_values(["中心时间", "窗口开始时间"], na_position="last").iloc[0]


def _representative_topic(period_summary: pd.DataFrame, details: pd.DataFrame) -> tuple[str | None, str, str | None]:
    peak_row = _select_peak_row(period_summary)
    peak_window_id = str(peak_row["共识窗口ID"]).strip()
    peak_details = details[details["共识窗口ID"].astype(str).str.strip().eq(peak_window_id)].copy()

    ranks = pd.to_numeric(peak_details.get("窗口内排名"), errors="coerce")
    dominant = peak_details[ranks.eq(1)].copy()
    if dominant.empty and "是否主导议题" in peak_details.columns:
        dominant = peak_details[_is_yes(peak_details["是否主导议题"])].copy()
    if not dominant.empty:
        dominant = dominant.sort_values(["议题量", "公众议题"], ascending=[False, True], na_position="last")
        return str(dominant.iloc[0]["公众议题"]).strip(), "最高潮窗口第一名", peak_window_id

    return None, "无可用议题明细", peak_window_id


def build_consensus_events(summary_path: str, details_path: str) -> pd.DataFrame:
    """
    构建共识事件。填写MANUAL_WINDOWS时使用手动窗口；留空时自动合并
    时间重叠或首尾相接的共识窗口，并自动提取代表议题。
    """
    summary = pd.read_csv(summary_path)
    details = pd.read_csv(details_path)

    _require_columns(summary, {"共识窗口ID", "窗口开始时间", "窗口结束时间(不含)", "中心时间", "是否为共识期"},
                     "共识汇总表")
    _require_columns(details, {"共识窗口ID", "公众议题", "议题量", "窗口内排名"}, "共识议题明细表")

    # 只保留共识期窗口
    summary = summary[_is_yes(summary["是否为共识期"])].copy()
    summary["共识窗口ID"] = summary["共识窗口ID"].fillna("").astype(str).str.strip()
    summary = summary[summary["共识窗口ID"].ne("")].copy()

    for column in ["窗口开始时间", "窗口结束时间(不含)", "中心时间"]:
        summary[column] = pd.to_datetime(summary[column], errors="coerce")

    summary = summary.dropna(subset=["窗口开始时间", "窗口结束时间(不含)", "中心时间"])
    summary = summary[summary["窗口结束时间(不含)"] >= summary["窗口开始时间"]].copy()

    details["共识窗口ID"] = details["共识窗口ID"].fillna("").astype(str).str.strip()

    window_specs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if MANUAL_WINDOWS:
        for start_str, end_str in MANUAL_WINDOWS:
            try:
                start = pd.Timestamp(start_str).normalize()
                end = pd.Timestamp(end_str).normalize()
            except Exception as exc:
                print(f"警告：无法解析手动窗口 {start_str} 至 {end_str}：{exc}")
                continue
            if end < start:
                print(f"警告：手动窗口 {start_str} 至 {end_str} 结束时间早于开始时间，跳过。")
                continue
            window_specs.append((start, end))
    else:
        ordered = summary.sort_values(["窗口开始时间", "窗口结束时间(不含)"])
        for _, row in ordered.iterrows():
            start = row["窗口开始时间"]
            end_exclusive = row["窗口结束时间(不含)"]
            end = end_exclusive - pd.Timedelta(days=1)
            if not window_specs or start > window_specs[-1][1] + pd.Timedelta(days=1):
                window_specs.append((start, end))
            else:
                previous_start, previous_end = window_specs[-1]
                window_specs[-1] = (previous_start, max(previous_end, end))
        print(f"MANUAL_WINDOWS为空，已自动合并得到 {len(window_specs)} 个共识事件窗口。")

    event_rows: list[dict[str, object]] = []
    for event_number, (start, end) in enumerate(window_specs, start=1):
        # 筛选与手动窗口重叠的原始共识窗口
        overlap_mask = (
            (summary["窗口开始时间"] <= end) &
            (summary["窗口结束时间(不含)"] > start)
        )
        period_df = summary.loc[overlap_mask].copy()

        if period_df.empty:
            print(f"警告：窗口 {start.date()} 至 {end.date()} 未匹配到任何共识窗口，跳过。")
            continue

        # 提取代表议题
        representative, _, _ = _representative_topic(period_df, details)
        if not representative:
            print(f"警告：窗口 {start.date()} 至 {end.date()} 无法提取代表议题，跳过。")
            continue

        event_rows.append({
            "事件ID": f"共识窗口{event_number}",
            "共识开始时间": start,
            "共识结束时间": end,
            "代表议题_Y": representative,
            "窗口时长": (end - start).days + 1
        })

    events = pd.DataFrame(event_rows)
    if not events.empty:
        return events.sort_values("共识开始时间").reset_index(drop=True)
    return events


class DailyTopicSeries:
    INDICATORS = ["议题", "正面情感", "负面情感", "平均情感"]

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.start = pd.Timestamp(data["日期"].min()).normalize()
        self.end = pd.Timestamp(data["日期"].max()).normalize()
        self.index = pd.date_range(self.start, self.end, freq="D")
        self._cache: dict[tuple[str, str, str], pd.Series] = {}

    def get(self, actor: str, topic: str, indicator: str = "议题") -> pd.Series:
        key = (actor, topic, indicator)
        if key in self._cache:
            return self._cache[key].copy()

        mask = self.data["主体"].eq(actor) & self.data["议题"].eq(topic)
        topic_data = self.data.loc[mask]

        if indicator == "议题":
            daily_count = (
                topic_data.groupby("日期").size().astype(float)
                .reindex(self.index, fill_value=0.0)
            )
            series = self._transform_count(daily_count, actor)

        elif indicator in ("正面情感", "负面情感"):
            polarity = "正向情感圈层" if indicator == "正面情感" else "负向情感圈层"
            sub = topic_data[topic_data["情感极性"].eq(polarity)]
            daily_count = (
                sub.groupby("日期").size().astype(float)
                .reindex(self.index, fill_value=0.0)
            )
            series = self._transform_count(daily_count, actor)

        elif indicator == "平均情感":
            daily_mean = (
                topic_data.groupby("日期")["情感得分"]
                .mean()
                .reindex(self.index)
            )
            series = daily_mean.fillna(0.0).astype(float)

        else:
            raise ValueError(f"未知指标: {indicator}")

        series.name = f"{actor}:{topic}:{indicator}"
        self._cache[key] = series
        return series.copy()

    def _transform_count(self, daily_count: pd.Series, actor: str) -> pd.Series:
        if ENABLE_EXPONENTIAL_DECAY and actor in {"政府", "媒体"}:
            decayed_stock = compute_exponential_decay_stock(daily_count, DECAY_GAMMA)
            return np.log1p(decayed_stock)
        return np.log1p(daily_count)


def _stationarize(series: pd.Series) -> tuple[pd.Series, bool]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.nunique() <= 1:
        return clean, False
    try:
        p_value = float(adfuller(clean, autolag="AIC")[1])
    except Exception:
        p_value = 1.0
    if p_value < 0.5:
        return clean, False
    return clean.diff(), True


def _significance_level(p_val: float) -> str:
    if pd.isna(p_val): return "ns"
    if p_val < 0.01:
        return "***"
    elif p_val < 0.05:
        return "**"
    elif p_val < 0.10:
        return "*"
    else:
        return "ns"


def _merge_periods(periods: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not periods:
        return []
    periods = sorted(
        (pd.Timestamp(s).normalize(), pd.Timestamp(e).normalize()) for s, e in periods
    )
    merged = [list(periods[0])]
    for start, end in periods[1:]:
        last = merged[-1]
        if start <= last[1] + pd.Timedelta(days=1):
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _format_periods(periods: list[tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    if not periods:
        return "无"
    return "; ".join(
        f"{s.strftime('%Y-%m-%d')}~{e.strftime('%Y-%m-%d')}" for s, e in periods
    )


def granger_p_value_from_stats(Y: np.ndarray, X: np.ndarray, lag: int) -> float:
    F_obs, _ = granger_stats(Y, X, lag)
    if np.isnan(F_obs):
        return np.nan
    n = len(Y)
    df_num = lag
    df_den = (n - lag) - (2 * lag + 1)
    if df_den <= 0:
        return np.nan
    return float(f_dist.sf(F_obs, df_num, df_den))


def find_significant_periods(
    x_series: pd.Series,
    y_series: pd.Series,
    direction: str,
    lag: int,
    window: int = 60,
    step: int = 14,
    p_threshold: float = 0.10,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    滚动窗口检测显著格兰杰因果时间段。
    direction: "X->Y" 或 "Y->X"
    """
    df = pd.concat([x_series, y_series], axis=1).dropna()
    n = len(df)
    if n <= lag + 5:
        return []

    def test_sub(sub: pd.DataFrame) -> float:
        x = sub.iloc[:, 0].to_numpy(dtype=float)
        y = sub.iloc[:, 1].to_numpy(dtype=float)
        if direction == "X->Y":
            return granger_p_value_from_stats(y, x, lag)
        else:
            return granger_p_value_from_stats(x, y, lag)

    periods = []
    if n < window:
        if test_sub(df) < p_threshold:
            periods.append((df.index[0], df.index[-1]))
        return _merge_periods(periods)

    start_indices = list(range(0, n - window + 1, step))
    if (n - window) % step != 0:
        start_indices.append(n - window)

    for start_idx in start_indices:
        sub = df.iloc[start_idx:start_idx + window]
        if test_sub(sub) < p_threshold:
            periods.append((sub.index[0], sub.index[-1]))

    return _merge_periods(periods)


def _format_global_causality(p_xy: float, p_yx: float, alpha: float = 0.05) -> str:
    if pd.isna(p_xy) or pd.isna(p_yx):
        return "无法判断"
    sig_xy = p_xy < alpha
    sig_yx = p_yx < alpha
    if sig_xy and sig_yx:
        return "双向因果"
    elif sig_xy:
        return "X→Y"
    elif sig_yx:
        return "Y→X"
    else:
        return "无显著因果关系"


def compute_var_diagnostics(x_series: pd.Series, y_series: pd.Series) -> dict:
    """
    输入为已平稳化后的两个序列。
    返回：最优滞后阶数、两个方向的全局 P 值、
    第十期方差分解、第十期累计脉冲响应。
    """
    df = pd.concat([x_series, y_series], axis=1).dropna()
    df.columns = ["X", "Y"]
    n = len(df)
    print(f"有效样本数: {n}")

    if n <= 10:
        print("    -> 样本数过少(≤10)，跳过")
        return {}

    # 【终极修复1】：微小白噪声注入 (Jitter Trick)
    # 注入 1e-8 级别的噪声在数学上不影响真实的统计和情绪特性，
    # 但能完美避开底层的 Cholesky 分解崩溃和奇异矩阵错误。
    np.random.seed(42)  # 固定随机种子保证结果可复现
    df["X"] = df["X"] + np.random.normal(0, 1e-8, n)
    df["Y"] = df["Y"] + np.random.normal(0, 1e-8, n)

    max_lag = min(MAX_LAG_LIMIT, max(1, n // 3 - 1))
    if max_lag < 1:
        return {}

    try:
        model = VAR(df)
        selected = model.select_order(maxlags=max_lag)
        best_lag = int(selected.aic)
    except Exception as e:
        print(f"    -> 选阶提示: {e} (默认回退到1阶)")
        best_lag = 1

    best_lag = max(1, min(best_lag, max_lag))

    try:
        res = model.fit(best_lag)
    except Exception as e:
        print(f"    -> 模型拟合彻底失败: {e}")
        return {}

    # 全局 Granger 因果检验
    try:
        p_xy = float(res.test_causality("X", "Y", kind="f").pvalue)
    except Exception as e:
        print(f"    -> p_xy检验计算失败: {e}")
        p_xy = np.nan

    try:
        p_yx = float(res.test_causality("Y", "X", kind="f").pvalue)
    except Exception as e:
        print(f"    -> p_yx检验计算失败: {e}")
        p_yx = np.nan

    # 第十期方差分解
    try:
        # 【修复1】：参数名改为 periods=10
        fevd = res.fevd(periods=10)

        # 提取方差贡献矩阵
        decomp = fevd.decomp

        # 兼容处理：statsmodels 的 decomp 形状可能是 (periods, k, k) 或 (k, k, periods)
        # 通常如果是 (periods, k_endog, k_endog)，则期数在第0维
        if decomp.shape[0] == 10 or decomp.shape[0] == 11:
            h_idx = -1  # 取最后一期
            var_y_by_x = float(decomp[h_idx, 1, 0] * 100)  # X(0) 对 Y(1) 的贡献
            var_x_by_y = float(decomp[h_idx, 0, 1] * 100)  # Y(1) 对 X(0) 的贡献
        else:
            h_idx = decomp.shape[2] - 1
            var_y_by_x = float(decomp[1, 0, h_idx] * 100)
            var_x_by_y = float(decomp[0, 1, h_idx] * 100)

    except Exception as e:
        print(f"    -> 方差分解(FEVD)失败: {e}")
        var_y_by_x = np.nan
        var_x_by_y = np.nan

    # 第十期累计脉冲响应（正交化 Cholesky）
    try:
        # 【修复2】：参数名改为 periods=10
        irf = res.irf(periods=10)
        orth_irfs = irf.orth_irfs

        # orth_irfs 的标准形状为 (periods+1, k_endog, k_endog) = (11, 2, 2)
        # 索引 [期数, 响应变量(Y=1), 冲击变量(X=0)]
        cum_xy = float(np.sum(orth_irfs[1:11, 1, 0]))  # X 冲击对 Y
        cum_yx = float(np.sum(orth_irfs[1:11, 0, 1]))  # Y 冲击对 X
    except Exception as e:
        print(f"    -> 脉冲响应(IRF)失败: {e}")
        cum_xy = np.nan
        cum_yx = np.nan

    return {
        "best_lag": best_lag,
        "p_xy": p_xy,
        "p_yx": p_yx,
        "var_y_by_x": var_y_by_x,
        "var_x_by_y": var_x_by_y,
        "cum_xy": cum_xy,
        "cum_yx": cum_yx,
    }

def granger_stats(Y: np.ndarray, X: np.ndarray, lag: int) -> tuple[float, float]:
    """快速矩阵计算 Granger F-stat 与 X特征累积系数"""
    n = len(Y)
    if n <= 2 * lag + 1:
        return np.nan, np.nan

    Y_lag = np.column_stack([Y[lag - i: n - i] for i in range(1, lag + 1)])
    X_lag = np.column_stack([X[lag - i: n - i] for i in range(1, lag + 1)])
    Y_target = Y[lag:]

    X_res = np.column_stack([np.ones(n - lag), Y_lag])
    X_unres = np.column_stack([np.ones(n - lag), Y_lag, X_lag])

    def fit(X_mat, y_vec):
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        resid = y_vec - X_mat @ beta
        ssr = np.sum(resid ** 2)
        return ssr, beta

    ssr_res, _ = fit(X_res, Y_target)
    ssr_unres, beta_unres = fit(X_unres, Y_target)

    df_num = lag
    df_den = (n - lag) - (2 * lag + 1)
    if df_den <= 0 or ssr_unres <= 1e-15:
        return np.nan, np.nan

    F = ((ssr_res - ssr_unres) / df_num) / (ssr_unres / df_den)
    coef = float(np.sum(beta_unres[-lag:]))  # X 滞后项系数之和

    return F, coef


def perform_bootstrap_test(Y_vals: np.ndarray, X_vals: np.ndarray, best_lag: int, F_obs: float) -> tuple[float, float]:
    """双重 Bootstrap：X 扰乱计算 P值，成对抽样计算 SE"""
    n_obs = len(Y_vals)
    block_length = max(best_lag + 1, int(round(n_obs ** (1 / 3))))
    indices = np.arange(n_obs)
    blocks = [indices[i:i + block_length] for i in range(0, n_obs, block_length)]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    F_null = []
    coef_boot = []

    for _ in range(N_BOOTSTRAP):
        # 1. 估算 P 值（破坏 X 依赖）
        shuffled_idx = []
        while len(shuffled_idx) < n_obs:
            shuffled_idx.extend(blocks[rng.choice(len(blocks))])
        shuffled_idx = np.array(shuffled_idx)[:n_obs]

        f, _ = granger_stats(Y_vals, X_vals[shuffled_idx], best_lag)
        if not np.isnan(f):
            F_null.append(f)

        # 2. 估算标准误 (成对 Block 抽样以保护结构)
        sampled_idx = []
        while len(sampled_idx) < n_obs:
            sampled_idx.extend(blocks[rng.choice(len(blocks))])
        sampled_idx = np.array(sampled_idx)[:n_obs]

        _, c = granger_stats(Y_vals[sampled_idx], X_vals[sampled_idx], best_lag)
        if not np.isnan(c):
            coef_boot.append(c)

    p_val = np.sum(np.array(F_null) >= F_obs) / len(F_null) if len(F_null) > 0 else np.nan
    se_boot = float(np.std(coef_boot)) if len(coef_boot) > 1 else np.nan
    return p_val, se_boot


def test_pair(cause_s: pd.Series, effect_s: pd.Series, cause_name: str, effect_name: str, direction_flag: str) -> dict:
    if cause_s.eq(0).all() or effect_s.eq(0).all():
        return {
            "解释变量 (原因)": cause_name,
            "被解释变量 (结果)": effect_name,
            "最优滞后阶数": "—",
            "系数": "—",
            "Bootstrap P值": "—",
            "显著性标记": "—",
            "标准误": "—",
            "作用方向": "数据全零"
        }

    # 检查平稳性，必要时自动差分
    cause_clean, c_diff = _stationarize(cause_s)
    effect_clean, e_diff = _stationarize(effect_s)
    if c_diff: cause_s = cause_s.diff().dropna()
    if e_diff: effect_s = effect_s.diff().dropna()

    df = pd.concat([cause_s, effect_s], axis=1).dropna()
    if len(df) <= 2:
        return {
            "解释变量 (原因)": f"{cause_name}" if c_diff else cause_name,
            "被解释变量 (结果)": f"{effect_name}" if e_diff else effect_name,
            "最优滞后阶数": "—",
            "系数": "—",
            "Bootstrap P值": "—",
            "显著性标记": "—",
            "标准误": "—",
            "作用方向": "窗口过短，无法估计"
        }

    cause_vals = df.iloc[:, 0].values
    effect_vals = df.iloc[:, 1].values

    # AIC 选阶限制 (避免观测数不足导致线性代数崩溃)
    max_lag = min(MAX_LAG_LIMIT, max(1, len(df) // 3 - 1))
    best_lag = 1
    try:
        model = VAR(df)
        res = model.select_order(maxlags=max_lag)
        if res.aic > 0:
            best_lag = int(res.aic)
    except Exception:
        pass
    best_lag = max(1, min(best_lag, max_lag))

    F_obs, coef_obs = granger_stats(effect_vals, cause_vals, best_lag)
    if np.isnan(F_obs):
        return {
            "解释变量 (原因)": cause_name,
            "被解释变量 (结果)": effect_name,
            "最优滞后阶数": best_lag,
            "系数": "—",
            "Bootstrap P值": "—",
            "显著性标记": "—",
            "标准误": "—",
            "作用方向": "矩阵共线，无法估计"
        }

    p_val, se_boot = perform_bootstrap_test(effect_vals, cause_vals, best_lag, F_obs)

    sig = _significance_level(p_val)
    if pd.isna(p_val) or p_val >= 0.1:
        direction_str = "—"
    else:
        is_x_to_y = (direction_flag == "X->Y")
        if coef_obs > 0:
            direction_str = "正向引导" if is_x_to_y else "正向倒逼"
        else:
            direction_str = "负向抑制"

    cause_display = f"{cause_name}" if c_diff else cause_name
    effect_display = f"{effect_name}" if e_diff else effect_name

    return {
        "解释变量 (原因)": cause_display,
        "被解释变量 (结果)": effect_display,
        "最优滞后阶数": best_lag,
        "系数": f"{coef_obs:.4f}" if not np.isnan(coef_obs) else "—",
        "Bootstrap P值": f"{p_val:.3f}" if not np.isnan(p_val) else "—",
        "显著性标记": sig,
        "标准误": f"{se_boot:.4f}" if not np.isnan(se_boot) else "—",
        "作用方向": direction_str
    }


def run_window_granger_tests(events: pd.DataFrame, series_store: DailyTopicSeries, output_csv: str):
    print("[*] 正在执行共识窗口内的 Bootstrap 格兰杰因果检验 (基于0808 PDF要求)...")

    all_results: list[dict[str, object]] = []

    for idx, event in events.iterrows():
        window_id = event["事件ID"]
        start_date = event["共识开始时间"]
        end_date = event["共识结束时间"]
        y_topic = str(event["代表议题_Y"])

        print(f"  -> 处理 {window_id} (时长: {event['窗口时长']} 天, 代表议题: {y_topic})")

        # 提取公众共识议题时间切片
        y_series = series_store.get("公众", y_topic).loc[start_date:end_date]
        y_alias = "Pub_Consensus"

        # 对每一个指定的 Gov/Med 议题进行双向检验
        for (x_actor, x_topic), x_alias in ALIAS_MAP.items():
            x_series = series_store.get(x_actor, x_topic).loc[start_date:end_date]

            # 方向 A: Gov/Med -> Pub_Consensus
            res_xy = test_pair(x_series, y_series, x_alias, y_alias, "X->Y")
            all_results.append({"事件ID": window_id, **res_xy})

            # 方向 B: Pub_Consensus -> Gov/Med (倒逼)
            res_yx = test_pair(y_series, x_series, y_alias, x_alias, "Y->X")
            all_results.append({"事件ID": window_id, **res_yx})

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[*] 结果已成功输出到 {output_csv} (共 {len(events)} 个事件，{len(df_all)} 行)")


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "共识窗口格兰杰检验结果.csv"
    full_pair_csv = output_dir / "全议题对格兰杰检验结果.csv"

    print("[*] 分析开始...")

    required_files = [INPUT_EXCEL, CONSENSUS_SUMMARY, CONSENSUS_DETAILS]
    missing_files = [path for path in required_files if not Path(path).exists()]
    if missing_files:
        raise FileNotFoundError(f"缺少格兰杰分析输入文件：{missing_files}")

    events = build_consensus_events(CONSENSUS_SUMMARY, CONSENSUS_DETAILS)
    if events.empty:
        print("[!] 警告：未构建出任何有效的共识窗口事件。")

    print("[*] 加载并准备议题级时序数据...")
    raw_data = load_topic_data(INPUT_EXCEL)
    series_store = DailyTopicSeries(raw_data)
    full_pair_start = FULL_PAIR_START or series_store.start.strftime("%Y-%m-%d")
    full_pair_end = FULL_PAIR_END or series_store.end.strftime("%Y-%m-%d")

    # 原有：共识窗口内代表议题检验
    if not events.empty:
        run_window_granger_tests(events, series_store, str(csv_path))

    # 新增：全议题对全时间段检验
    run_full_pair_granger_tests(
        series_store,
        full_pair_start,
        full_pair_end,
        str(full_pair_csv),
    )

    print("[*] 分析完成！")


if __name__ == "__main__":
    main()
