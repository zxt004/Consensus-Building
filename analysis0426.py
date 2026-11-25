from __future__ import annotations

"""Consensus-window detection plus the PDF SOP's daily consensus-score export.

The original sliding-window outputs are preserved.  The added
``每日社会共识度得分.csv`` contains exactly one row per calendar day and is the
Y series consumed by ``cross_lagged.py``.  Scores are calculated from all
public posts together rather than separately by sentiment circle.
"""

import os
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats
import ruptures as rpt

warnings.filterwarnings("ignore")

# 1. 更改参数与路径
INPUT_FILE = "/root/autodl-tmp/预制菜/预制菜output.xlsx"
BASE_OUTPUT_DIR = "/root/autodl-tmp/预制菜/advanced_analysis/"

EmotionConsistency = 0.75
len_current_period_indices = 1
min_posts = 10

TARGET_WINDOW_IDS = ["N-7D-20240928"]  # 示例: ["P-7D-20240801", "N-7D-20240815"]

# A score is retained for every day with at least one valid public post so the
# downstream day-level time series remains observable.  Days below the legacy
# 20-post threshold are explicitly flagged in the exported diagnostics.
DAILY_SCORE_MIN_POSTS = 1
DAILY_LOW_SAMPLE_THRESHOLD = min_posts

# 2. 更新主题字典
THEMES_MAPPING = {
    "食品安全与营养认知": ["校园", "食品", "冷冻", "食品安全", "营养", "食堂", "学生", "新鲜"],
    "消费者与企业家立场差异": ["罗永浩", "消费者", "贾国龙", "老板", "顾客"],
    "产品供应链透明度关注":["西贝货", "假货", "西贝", "蜀海", "厨房"],
    "个体消费行为倾向": ["面条", "没吃过", "不吃", "不去西贝", "反正", "去过", "以后", "好吃"],
    "社会发展隐喻讨论": ["快乐教育", "美国", "孩子", "大学", "教学", "学习", "预制菜"],
    "国际品牌参照与消费预期":["萨莉亚", "香港", "照片", "老周", "360", "罗永浩", "医生", "没有萨莉亚"],
    "历史经验与系统信任":["奶粉", "牛奶", "三鹿", "狗屎", "三聚氰胺", "内蒙", "标准", "现挤", "国标"],
    "语言表达与身份标签": ["网络黑社会", "老罗", "真正·黑社会", "罗永浩", "龙哥", "内蒙"],
    "技术风险认知":["鱼刺", "辐射", "残留", "检测", "物质", "能量", "鱼肉", "去除"]
}

MEDIA_GOV_THEMES = {
    "西贝预制菜舆论风波": {
        "description": "围绕西贝创始人贾国龙相关言论所引发的舆论争议展开回应与报道，关注企业家表态与公众反应之间的张力及其对品牌声誉的影响",
        "keywords": []
    },
    "预制菜产业发展与规范": {
        "description": "从宏观层面关注预制菜行业的规模扩张、产业链布局及政策监管框架的建立，强调在发展中逐步完善标准体系与行业规范",
        "keywords": []
    },
    "校园预制菜争议与管理": {
        "description": "聚焦预制菜进入学校食堂所引发的家长担忧，报道地方叫停措施、教育管理部门的回应及校园供餐安全标准的讨论",
        "keywords": []
    },
    "消费者体验与权益诉求": {
        "description": "关注消费者对预制菜产品品质、性价比及知情权的主张，报道相关投诉与维权事件",
        "keywords": []
    },
    "预制菜营销伦理争议": {
        "description": "围绕餐饮品牌在宣传中是否夸大预制菜的正向属性、是否模糊现制与预制的界限展开讨论，指向营销行为的伦理边界与监管空间",
        "keywords": []
    }
}

VALID_THEMES = list(THEMES_MAPPING.keys())
TOTAL_THEMES_COUNT = len(VALID_THEMES)

WINDOW_CONFIG = {
    # "3天平滑滑动": {"window_days": 3, "step_days": 1},
    # "7天平滑滑动": {"window_days": 7, "step_days": 2}
    "14天平滑滑动": {"window_days": 14, "step_days": 2}
}

def get_stage(date_obj: pd.Timestamp) -> str:
    if pd.isna(date_obj):
        return "未知阶段"
    date_obj = pd.to_datetime(date_obj)
    if date_obj < pd.Timestamp("2023-03-03"):
        return "阶段1"
    elif date_obj < pd.Timestamp("2024-03-30"):
        return "阶段2"
    elif date_obj < pd.Timestamp("2024-07-12"):
        return "阶段3"
    else:
        return "阶段4"

stages = [
    "阶段1",
    "阶段2",
    "阶段3",
    "阶段4"
]

def extract_valid_themes(theme_str: str) -> list[str]:
    """从匹配的主题标签字符串中提取属于 VALID_THEMES 的主题列表"""
    if pd.isna(theme_str):
        return []
    cleaned = re.sub(r'\([0-9.]+\)', '', str(theme_str))
    extracted = [t.strip() for t in cleaned.split('|') if t.strip()]
    return [t for t in extracted if t in VALID_THEMES]

def get_polarity_prefix(polarity: str) -> str:
    return {
        "正向情感圈层": "P",
        "负向情感圈层": "N",
        "中立情感圈层": "U"
    }.get(polarity, "A")


def load_and_preprocess_data(file_path: str, keep_account_types: list = None) -> pd.DataFrame:
    """
    加载并预处理数据。
    keep_account_types: 若为 None，则保留所有账号分类；若为列表，则只保留指定分类（如 ['public']）。
    """
    print(f"\n加载数据与预处理...")

    df = pd.read_excel(file_path)

    # 账号分类过滤（仅当 keep_account_types 不为 None 时执行）
    if keep_account_types is not None and "账号分类" in df.columns:
        original_len = len(df)
        df = df[df["账号分类"].isin(keep_account_types)].copy()
        print(f"依据 '账号分类 in {keep_account_types}' 过滤：从 {original_len} 条筛选出 {len(df)} 条记录。")
    elif keep_account_types is None:
        print("保留所有账号分类，不进行账号过滤。")
    else:
        print("警告：数据中未找到 '账号分类' 字段，无法执行过滤。")

    # 去重与无效文本过滤（适用于所有分类）
    if "蒸馏总结" in df.columns:
        df = df.dropna(subset=["蒸馏总结"]).drop_duplicates(subset=["蒸馏总结"])
        invalid_patterns = "未提及|未涉及|无法提炼|无法概括|无法得出|没有涉及|不涉及"
        df = df[~df["蒸馏总结"].astype(str).str.contains(invalid_patterns, regex=True, na=False)]

    # 时间解析（所有分类）
    time_col_found = False
    for tcand in ("日期", "时间", "创建时间", "发布时间", "parsed_time"):
        if tcand in df.columns:
            clean_time = df[tcand].astype(str).str.replace(r'(?:发布于|编辑于)\s*', '', regex=True).str.strip()
            time_strs = clean_time.str.extract(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}(?:\s+\d{1,2}:\d{1,2})?)')[0]
            time_strs = time_strs.str.replace("年", "-").str.replace("月", "-").str.replace("日", "")
            df["日期"] = pd.to_datetime(time_strs, errors="coerce")
            time_col_found = True
            break

    if not time_col_found or "日期" not in df.columns:
        df["日期"] = pd.NaT

    df['情感得分'] = pd.to_numeric(df.get('情感得分', 0), errors='coerce').fillna(0)

    def get_polarity(score):
        if score > 0.0:
            return "正向情感圈层"
        elif score < -0.0:
            return "负向情感圈层"
        else:
            return "中立情感圈层"

    df['SentimentPolarity'] = df['情感得分'].apply(get_polarity)

    df = df.dropna(subset=['日期'])

    # 主题提取（所有分类都尝试提取，但仅对公众有效，对 government/media 可能为空）
    if "匹配的主题标签" in df.columns:
        df["themes_list"] = df["匹配的主题标签"].apply(extract_valid_themes)
        # 仅当 keep_account_types 指定为 ['public'] 时，才过滤掉无主题的行（避免误删 government/media）
        is_public_only = (keep_account_types is not None and set(keep_account_types) == {'public'})
        if is_public_only:
            df = df[df["themes_list"].map(len) > 0]
            print(f"仅保留 public 且有主题标签的数据，剩余 {len(df)} 条。")
        else:
            print("保留所有分类，即使 themes_list 为空（后续导出会重新匹配）。")
    else:
        print("警告：数据中未找到 '匹配的主题标签' 字段，后续主题提取可能失效。")

    print(f"预处理完成，共 {len(df)} 条记录。")
    return df


def calculate_window_metrics(group_df: pd.DataFrame, min_posts: int = 15) -> dict:
    total_posts = len(group_df)

    dominant_theme = None
    dominant_theme_pct = np.nan
    topic_ranking = ""
    topic_count = 0
    mean_sentiment = group_df['情感得分'].mean() if total_posts > 0 else np.nan

    if total_posts < min_posts:
        return {
            'TopicFocus': np.nan, 'EmotionConsistency': np.nan, 'ConsensusScore': np.nan,
            'PostCount': total_posts, 'DominantTheme': dominant_theme,
            'DominantThemePct': dominant_theme_pct, 'TopicRanking': topic_ranking,
            'TopicRankingWithPct': topic_ranking, 'TopicStats': [],
            'TopicCount': topic_count, 'MeanSentiment': mean_sentiment
        }

    themes_exploded = group_df.explode('themes_list')
    theme_counts = themes_exploded['themes_list'].value_counts()

    if len(theme_counts) == 0:
        topic_focus = 0.0
    else:
        p_i = theme_counts.values / theme_counts.sum()
        entropy = scipy.stats.entropy(p_i, base=np.e)
        max_entropy = np.log(TOTAL_THEMES_COUNT)
        topic_focus = 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        dominant_theme = theme_counts.index[0]
        dominant_theme_pct = theme_counts.values[0] / theme_counts.sum()
        topic_count = len(theme_counts)
        topic_ranking = "；".join(
            f"{rank}.{theme}({int(count)})"
            for rank, (theme, count) in enumerate(theme_counts.items(), start=1)
        )
        total_theme_mentions = int(theme_counts.sum())
        topic_stats = [
            {
                'theme': theme,
                'count': int(count),
                'share': float(count / total_theme_mentions),
                'rank': rank
            }
            for rank, (theme, count) in enumerate(theme_counts.items(), start=1)
        ]
        topic_ranking_with_pct = "；".join(
            f"{item['rank']}.{item['theme']}({item['count']}, {item['share']:.2%})"
            for item in topic_stats
        )

    if len(theme_counts) == 0:
        topic_stats = []
        topic_ranking_with_pct = ""

    sentiment_std = group_df['情感得分'].std(ddof=0)
    if pd.isna(sentiment_std): sentiment_std = 1.0

    emotion_consistency = max(0.0, 1 - (sentiment_std ** 2))

    if topic_focus + emotion_consistency > 0:
        consensus_score = (2 * topic_focus * emotion_consistency) / (topic_focus + emotion_consistency)
    else:
        consensus_score = 0.0

    return {
        'TopicFocus': topic_focus,
        'EmotionConsistency': emotion_consistency,
        'ConsensusScore': consensus_score,
        'PostCount': total_posts,
        'DominantTheme': dominant_theme,
        'DominantThemePct': dominant_theme_pct,
        'TopicRanking': topic_ranking,
        'TopicRankingWithPct': topic_ranking_with_pct,
        'TopicStats': topic_stats,
        'TopicCount': topic_count,
        'MeanSentiment': mean_sentiment
    }


def build_daily_consensus_scores(
        df: pd.DataFrame,
        score_min_posts: int = DAILY_SCORE_MIN_POSTS,
        low_sample_threshold: int = DAILY_LOW_SAMPLE_THRESHOLD,
) -> pd.DataFrame:
    """Build the true daily Y series required by the 0801 PDF SOP.

    The same topic-focus, emotion-consistency, and harmonic-mean formula used
    by the legacy rolling analysis is applied to each calendar day.  Missing
    public-data days remain present with a missing score; they must not be
    interpolated before ADF or Granger testing.
    """
    columns = [
        '日期', '社会共识度得分', '每日社会共识度得分', '话题聚焦度',
        '情感一致性', '当日文本数', '当日议题标签量', '主导话题',
        '主导话题占比', '低样本日',
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    if score_min_posts < 1:
        raise ValueError("score_min_posts 必须大于 0。")
    if low_sample_threshold < score_min_posts:
        raise ValueError("low_sample_threshold 不能小于 score_min_posts。")

    working = df.copy()
    working['日期'] = pd.to_datetime(working['日期'], errors='coerce').dt.normalize()
    working = working.dropna(subset=['日期'])
    if working.empty:
        return pd.DataFrame(columns=columns)

    calendar = pd.date_range(working['日期'].min(), working['日期'].max(), freq='D')
    rows = []
    for date in calendar:
        day = working[working['日期'].eq(date)]
        if day.empty:
            rows.append(
                {
                    '日期': date,
                    '社会共识度得分': np.nan,
                    '每日社会共识度得分': np.nan,
                    '话题聚焦度': np.nan,
                    '情感一致性': np.nan,
                    '当日文本数': 0,
                    '当日议题标签量': 0,
                    '主导话题': None,
                    '主导话题占比': np.nan,
                    '低样本日': True,
                }
            )
            continue

        metrics = calculate_window_metrics(day, min_posts=score_min_posts)
        score = metrics['ConsensusScore']
        rows.append(
            {
                '日期': date,
                '社会共识度得分': score,
                '每日社会共识度得分': score,
                '话题聚焦度': metrics['TopicFocus'],
                '情感一致性': metrics['EmotionConsistency'],
                '当日文本数': int(metrics['PostCount']),
                '当日议题标签量': int(
                    sum(item['count'] for item in metrics.get('TopicStats', []))
                ),
                '主导话题': metrics['DominantTheme'],
                '主导话题占比': metrics['DominantThemePct'],
                '低样本日': int(metrics['PostCount']) < low_sample_threshold,
            }
        )

    daily = pd.DataFrame(rows, columns=columns)
    numeric_columns = ['社会共识度得分', '每日社会共识度得分', '话题聚焦度', '情感一致性']
    daily[numeric_columns] = daily[numeric_columns].round(6)
    return daily


def analyze_consensus_periods(df: pd.DataFrame, polarity: str, window_label: str, window_days: int,
                              step_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub_df = df[df['SentimentPolarity'] == polarity].copy()
    if sub_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    print(f"\n========== 分析维度: [{polarity}] | 尺度: {window_label} ==========")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"Consensus_{polarity}_{window_label}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_date = sub_df['日期'].min().floor('D')
    end_date = sub_df['日期'].max().ceil('D')

    records = []
    current_date = start_date

    while current_date + pd.Timedelta(days=window_days) <= end_date + pd.Timedelta(days=step_days):
        window_end = current_date + pd.Timedelta(days=window_days)
        mask = (sub_df['日期'] >= current_date) & (sub_df['日期'] < window_end)
        window_df = sub_df.loc[mask]

        metrics = calculate_window_metrics(window_df, min_posts=min_posts)
        metrics['WindowStart'] = current_date
        metrics['WindowEnd'] = window_end - pd.Timedelta(days=1)
        metrics['WindowEndExclusive'] = window_end
        metrics['日期'] = current_date + pd.Timedelta(days=window_days / 2)
        metrics['ConsensusWindowID'] = (
            f"{get_polarity_prefix(polarity)}-{window_days}D-{current_date.strftime('%Y%m%d')}"
        )

        records.append(metrics)
        current_date += pd.Timedelta(days=step_days)

    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.dropna(subset=['ConsensusScore']).copy()
    metrics_df = metrics_df.reset_index(drop=True)

    if metrics_df.empty:
        print(f"[{polarity}] 数据稀疏，无满足要求的连续窗口。")
        return pd.DataFrame(), pd.DataFrame()

    TopicFocus_Threshold = metrics_df['TopicFocus'].quantile(0.5)
    EmotionConsistency_Threshold = max(0.5, metrics_df['EmotionConsistency'].quantile(0.5))

    metrics_df['IsConsensus'] = (metrics_df['TopicFocus'] >= TopicFocus_Threshold) & \
                                (metrics_df['EmotionConsistency'] >= EmotionConsistency_Threshold)

    metrics_df['Stage'] = metrics_df['日期'].apply(get_stage)

    # 标记周期和最高潮，并新增变点记录字段
    metrics_df['IsPeak'] = '否'

    metrics_df['共识最高潮日'] = pd.Series(np.nan, index=metrics_df.index, dtype=object)
    metrics_df['共识爆发点'] = pd.Series(np.nan, index=metrics_df.index, dtype=object)
    metrics_df['共识消散点'] = pd.Series(np.nan, index=metrics_df.index, dtype=object)

    signal = metrics_df['ConsensusScore'].values
    changepoints = []

    if len(signal) > 5:
        algo = rpt.Pelt(model="l2", min_size=2).fit(signal)
        penalty = np.log(len(signal)) * signal.var() * 3
        try:
            cps_indices = algo.predict(pen=penalty)
            for cp_idx in cps_indices:
                if cp_idx >= len(metrics_df):
                    continue
                window = 3
                start_before = max(0, cp_idx - window)
                end_after = min(len(signal), cp_idx + window)
                mean_before = signal[start_before:cp_idx].mean() if cp_idx > 0 else 0
                mean_after = signal[cp_idx:end_after].mean()
                cp_direction = "上升" if mean_after > mean_before else "下降"
                changepoints.append({
                    'date': metrics_df.iloc[cp_idx]['日期'],
                    'idx': cp_idx,
                    'direction': cp_direction,
                    'mean_before': mean_before,
                    'mean_after': mean_after
                })
        except Exception as e:
            pass

    periods = []
    current_period_indices = []

    for idx, row in metrics_df.iterrows():
        if row['IsConsensus']:
            if not current_period_indices:
                # 如果当前没有正在记录的共识期，直接开启新周期
                current_period_indices.append(idx)
            else:
                # 获取上一行的索引和窗口开始时间
                prev_idx = current_period_indices[-1]
                prev_window_start = metrics_df.loc[prev_idx, 'WindowStart']
                curr_window_start = row['WindowStart']

                # 计算与上一行之间的时间差
                delta_days = (curr_window_start - prev_window_start).days

                # 判定连续性：如果当前行的“窗口开始时间”与上一行相比，刚好等于滑动步长，则为连续
                if delta_days == step_days:
                    current_period_indices.append(idx)
                else:
                    # 如果时间跨度大于步长，说明中间断开了，需要强制开启一个新的共识阶段
                    if len(current_period_indices) >= len_current_period_indices:
                        periods.append(metrics_df.loc[current_period_indices])
                    # 开启全新的共识段，将当前行作为新段的起点
                    current_period_indices = [idx]
        else:
            # 如果当前行不是共识期，结算已有的连续期
            if len(current_period_indices) >= len_current_period_indices:
                periods.append(metrics_df.loc[current_period_indices])
            current_period_indices = []

    # 循环结束后，处理最后一段可能存在的连续期
    if len(current_period_indices) >= len_current_period_indices:
        periods.append(metrics_df.loc[current_period_indices])

    metrics_df['ConsensusPeriodID'] = ''
    polarity_prefix = get_polarity_prefix(polarity)
    for period_number, period_df in enumerate(periods, start=1):
        metrics_df.loc[period_df.index, 'ConsensusPeriodID'] = f'{polarity_prefix}-S{period_number:03d}'

    report_lines = []
    # stages = [
    #     "阶段1 (以前~2023.8)", "阶段2 (2023.9~2024.2)",
    #     "阶段3 (2024.3~2025.8)", "阶段4 (2025.9~至今)"
    # ]
    # 修改后

    for stage in stages:
        report_lines.append(f"\n【 {stage} 】")
        stage_periods = [p for p in periods if p['Stage'].iloc[0] == stage]
        stage_cps = [cp for cp in changepoints if get_stage(cp['date']) == stage]

        if not stage_periods and not stage_cps:
            report_lines.append("无短暂共识持续期或显著转折点。")
            continue

        if stage_periods:
            for idx, p in enumerate(stage_periods, 1):
                start_idx = p.index[0]
                end_idx = p.index[-1]
                period_id = metrics_df.loc[start_idx, 'ConsensusPeriodID']
                start_date = p['日期'].iloc[0].strftime('%Y.%m.%d')
                end_date = p['日期'].iloc[-1].strftime('%Y.%m.%d')

                peak_idx = p['ConsensusScore'].idxmax()
                peak_row = p.loc[peak_idx]
                peak_date_report = peak_row['日期'].strftime('%Y.%m.%d')
                peak_date_csv = peak_row['日期'].strftime('%Y-%m-%d')
                peak_score = peak_row['ConsensusScore']

                # 标记该周期的最高潮点，并将相关信息同步广播到CSV对应的所有行
                metrics_df.loc[peak_idx, 'IsPeak'] = '是'
                metrics_df.loc[p.index, '共识最高潮日'] = peak_date_csv

                report_lines.append(
                    f"  * 圈层共识持续期 {period_id}：核心时间点 {start_date} - {end_date} "
                    f"(横跨 {len(p)} 个滑动窗口)")
                report_lines.append(f"    - 共识最高潮：{peak_date_report} (共识分={peak_score:.2f})")

                # 定义时间匹配阈值（容忍变点在周期前后浮动的天数）
                time_tolerance_days = 3 * step_days

                # 爆发点匹配
                matching_start_cps = [
                    cp for cp in changepoints
                    if cp['direction'] == '上升'
                       and abs((cp['date'] - p['日期'].iloc[0]).days) <= time_tolerance_days
                       and get_stage(cp['date']) == stage
                ]

                if matching_start_cps:
                    cps_str_report = ", ".join([d['date'].strftime('%Y.%m.%d') for d in matching_start_cps])
                    cps_str_csv = ", ".join([d['date'].strftime('%Y-%m-%d') for d in matching_start_cps])
                    report_lines.append(f"    - 共识爆发点 (上升变点)：{cps_str_report}")
                    metrics_df.loc[p.index, '共识爆发点'] = cps_str_csv
                else:
                    report_lines.append(f"    - 共识爆发点：平缓过渡")
                    metrics_df.loc[p.index, '共识爆发点'] = "平缓过渡"

                # 消散点匹配
                matching_end_cps = [
                    cp for cp in changepoints
                    if cp['direction'] == '下降'
                       and abs((cp['date'] - p['日期'].iloc[-1]).days) <= time_tolerance_days
                       and get_stage(cp['date']) == stage  # 强制要求变点不能跨越阶段
                ]

                if matching_end_cps:
                    cps_str_report = ", ".join([d['date'].strftime('%Y.%m.%d') for d in matching_end_cps])
                    cps_str_csv = ", ".join([d['date'].strftime('%Y-%m-%d') for d in matching_end_cps])
                    report_lines.append(f"    - 共识消散点 (下降变点)：{cps_str_report}")
                    metrics_df.loc[p.index, '共识消散点'] = cps_str_csv
                else:
                    report_lines.append(f"    - 共识消散点：自然衰减")
                    metrics_df.loc[p.index, '共识消散点'] = "自然衰减"
        else:
            report_lines.append("  * 本阶段内无满足长度的圈层共识区间。")

    print("\n".join(report_lines))

    topic_detail_rows = []
    for _, row in metrics_df.iterrows():
        topic_stats = row.get('TopicStats', [])
        total_mentions = sum(item['count'] for item in topic_stats)
        for item in topic_stats:
            topic_detail_rows.append({
                '情感极性': polarity,
                '所属窗口': window_label,
                '所属阶段': row['Stage'],
                '共识窗口ID': row['ConsensusWindowID'],
                '共识周期ID': row['ConsensusPeriodID'],
                '窗口开始时间': row['WindowStart'],
                '窗口结束时间': row['WindowEnd'],
                '窗口结束时间(不含)': row['WindowEndExclusive'],
                '中心时间': row['日期'],
                '是否为共识期': bool(row['IsConsensus']),
                '公众议题': item['theme'],
                '议题量': item['count'],
                '议题标签总量': total_mentions,
                '窗口内占比': item['share'],
                '窗口内排名': item['rank'],
                '是否主导议题': item['rank'] == 1,
                '共识窗口全部议题': row['TopicRankingWithPct']
            })

    topic_details_df = pd.DataFrame(topic_detail_rows)
    if not topic_details_df.empty:
        topic_details_df = topic_details_df[topic_details_df['是否为共识期']].copy()
        for col in ['窗口开始时间', '窗口结束时间', '窗口结束时间(不含)', '中心时间']:
            topic_details_df[col] = topic_details_df[col].dt.strftime('%Y-%m-%d')
        topic_details_df['是否为共识期'] = '是'
        topic_details_df['是否主导议题'] = topic_details_df['是否主导议题'].map({True: '是', False: '否'})
        topic_details_df.to_csv(
            os.path.join(output_dir, "共识窗口全部议题明细.csv"),
            index=False, encoding='utf-8-sig'
        )

    # 填补充加的字段
    metrics_df['情感极性'] = polarity
    metrics_df['所属窗口'] = window_label

    # 将字段名换成全中文
    rename_cols = {
        'ConsensusWindowID': '共识窗口ID',
        'ConsensusPeriodID': '共识周期ID',
        'WindowStart': '窗口开始时间',
        'WindowEnd': '窗口结束时间',
        'WindowEndExclusive': '窗口结束时间(不含)',
        '日期': '中心时间',
        'PostCount': '该窗口总文本数',
        'DominantTheme': '主导话题',
        'DominantThemePct': '主导话题占比',
        'TopicRanking': '共识窗口议题排序',
        'TopicRankingWithPct': '共识窗口全部议题',
        'TopicCount': '议题种类数',
        'MeanSentiment': '平均情感指数',
        'TopicFocus': '话题聚焦度',
        'EmotionConsistency': '情感一致性',
        'ConsensusScore': '共识分',
        'IsConsensus': '是否为共识期',
        'Stage': '所属阶段',
        'IsPeak': '是否为该周期最高潮'
    }
    metrics_df = metrics_df.rename(columns=rename_cols)

    # 格式化日期和数值
    for col in ['中心时间', '窗口开始时间', '窗口结束时间', '窗口结束时间(不含)']:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].dt.strftime('%Y-%m-%d')

    if '主导话题占比' in metrics_df.columns:
        metrics_df['主导话题占比'] = metrics_df['主导话题占比'].apply(
            lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "0%")

    for col in ['平均情感指数', '话题聚焦度', '情感一致性', '共识分']:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: round(x, 4) if pd.notna(x) else np.nan)

    if '是否为共识期' in metrics_df.columns:
        metrics_df['是否为共识期'] = metrics_df['是否为共识期'].map({True: '是', False: '否'}).fillna('否')

    # 保留分析报告 TXT
    with open(os.path.join(output_dir, "consensus_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"【{polarity}】圈层共识与转折点分析报告 ({window_label})\n")
        f.write("\n".join(report_lines))

    # 列重排，使其更易读（加入新增的三个字段）
    ordered_cols = [
        '情感极性', '所属窗口', '所属阶段', '共识窗口ID', '共识周期ID',
        '窗口开始时间', '窗口结束时间', '窗口结束时间(不含)', '中心时间',
        '该窗口总文本数', '主导话题', '主导话题占比', '共识窗口议题排序',
        '共识窗口全部议题', '议题种类数', '平均情感指数',
        '话题聚焦度', '情感一致性', '共识分', '是否为共识期', '是否为该周期最高潮',
        '共识最高潮日', '共识爆发点', '共识消散点'
    ]
    metrics_df = metrics_df[[c for c in ordered_cols if c in metrics_df.columns]]

    return metrics_df, topic_details_df


def export_consensus_milestones(periods_list, changepoints, polarity, window_label):
    """
    提取每个周期的最高潮、爆发点、消散点并转换为DataFrame
    """
    milestones = []

    for idx, p in enumerate(periods_list, 1):
        start_idx = p.index[0]
        end_idx = p.index[-1]

        # 1. 获取最高潮
        peak_idx = p['ConsensusScore'].idxmax()
        peak_row = p.loc[peak_idx]

        # 2. 匹配爆发点 (上升变点)
        eruption_pts = [cp['date'].strftime('%Y-%m-%d') for cp in changepoints
                        if cp['direction'] == '上升' and abs(cp['idx'] - start_idx) <= 5]
        eruption_str = ", ".join(eruption_pts) if eruption_pts else "平缓过渡"

        # 3. 匹配消散点 (下降变点)
        dissipation_pts = [cp['date'].strftime('%Y-%m-%d') for cp in changepoints
                           if cp['direction'] == '下降' and abs(cp['idx'] - (end_idx + 1)) <= 5]
        dissipation_str = ", ".join(dissipation_pts) if dissipation_pts else "自然衰减"

        milestones.append({
            "情感圈层": polarity,
            "时间尺度": window_label,
            "周期编号": f"周期_{idx}",
            "周期范围": f"{p['日期'].iloc[0].strftime('%Y-%m-%d')} 至 {p['日期'].iloc[-1].strftime('%Y-%m-%d')}",
            "共识最高潮时间": peak_row['日期'].strftime('%Y-%m-%d'),
            "最高潮共识分": round(peak_row['ConsensusScore'], 4),
            "最高潮主导话题": peak_row['DominantTheme'],
            "共识爆发点": eruption_str,
            "共识消散点": dissipation_str
        })

    return pd.DataFrame(milestones)


# 导出指定共识窗口的评论数据
def export_specific_window_comments(raw_df: pd.DataFrame, all_metrics_df: pd.DataFrame, output_dir: str,
                                    target_window_ids: list = None):
    """导出指定共识窗口ID（或所有共识期窗口）的原始评论数据"""
    print("\n========== [新增功能] 开始导出共识窗口评论数据 ==========")
    out_dir = os.path.join(output_dir, "共识窗口评论明细")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    consensus_windows = all_metrics_df[all_metrics_df['是否为共识期'] == '是'].copy()
    if consensus_windows.empty:
        print("未找到任何共识期窗口，跳过导出。")
        return

    if target_window_ids:
        consensus_windows = consensus_windows[consensus_windows['共识窗口ID'].isin(target_window_ids)]
        if consensus_windows.empty:
            print(f"指定的窗口ID {target_window_ids} 未在共识期中找到。")
            return

    exported_count = 0
    raw_df_copy = raw_df.copy()
    raw_df_copy['日期'] = pd.to_datetime(raw_df_copy['日期'], errors='coerce')

    for _, row in consensus_windows.iterrows():
        window_id = row['共识窗口ID']
        polarity = row['情感极性']
        start_date = pd.to_datetime(row['窗口开始时间'])
        end_date = pd.to_datetime(row['窗口结束时间(不含)'])

        mask = (raw_df_copy['日期'] >= start_date) & (raw_df_copy['日期'] < end_date) & (
                raw_df_copy['SentimentPolarity'] == polarity)
        window_data = raw_df_copy[mask]

        if not window_data.empty:
            file_name = f"{window_id}_{polarity}_评论数据.csv"
            file_path = os.path.join(out_dir, file_name)
            window_data.to_csv(file_path, index=False, encoding='utf-8-sig')
            exported_count += 1

    print(f"成功导出 {exported_count} 个共识窗口的评论数据至: {out_dir}")


def export_accounts_topic_and_sentiment(df: pd.DataFrame, output_dir: str):
    """
    导出政府、媒体、公众的议题分布（全部匹配议题）及情感倾向占比。
    使用清洗后的 df，保证与 '4个主题整体占比.csv' 数据基础一致。
    """
    print("\n========== [修改后] 导出政府/媒体/公众的议题分布与情感占比 ==========")

    # 确保必要字段存在
    required_cols = ["账号分类", "蒸馏总结", "情感得分", "themes_list"]
    for col in required_cols:
        if col not in df.columns:
            print(f"错误：缺少列 '{col}'，无法执行。")
            return

    # 复制数据框，避免修改原数据
    df_work = df.copy()

    # 对政府/媒体行，使用新主题词典重新匹配主题
    gov_media_mask = df_work['账号分类'].isin(['government', 'media'])

    def match_themes(text, theme_dict):
        """根据文本中包含的关键词匹配主题，返回主题列表（去重）"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        matched = []
        for theme, info in theme_dict.items():
            # 只要文本中包含任意一个关键词即匹配该主题
            if any(kw in text for kw in info['keywords']):
                matched.append(theme)
        return list(set(matched))  # 去重

    # 为政府/媒体重新生成 themes_list
    df_work.loc[gov_media_mask, 'themes_list'] = df_work.loc[gov_media_mask, '蒸馏总结'].apply(
        lambda x: match_themes(x, MEDIA_GOV_THEMES)
    )

    # 对于公众，保留原有的 themes_list（已在 load_and_preprocess_data 中生成）
    # 但需确保公众行也有 themes_list，且非空
    # （已由预处理保证，但为了安全，保留非空）

    # 过滤掉 themes_list 为空的记录
    df_work = df_work[df_work['themes_list'].map(len) > 0].copy()
    if df_work.empty:
        print("没有有效主题数据，跳过导出。")
        return

    # ---------- 1. 议题分布（全部匹配主题） ----------
    exploded = df_work.explode('themes_list')
    topic_counts = exploded.groupby(['账号分类', 'themes_list']).size().reset_index(name='频次')
    total_per_cat = exploded.groupby('账号分类').size().reset_index(name='总数')
    topic_counts = topic_counts.merge(total_per_cat, on='账号分类')
    topic_counts['占比'] = (topic_counts['频次'] / topic_counts['总数']).apply(lambda x: f"{x:.2%}")
    topic_counts = topic_counts[['账号分类', 'themes_list', '频次', '占比']]
    topic_counts.rename(columns={'themes_list': '主题'}, inplace=True)
    topic_counts.sort_values(['账号分类', '频次'], ascending=[True, False], inplace=True)

    topic_out = os.path.join(output_dir, "账号分类议题分布.csv")
    topic_counts.to_csv(topic_out, index=False, encoding='utf-8-sig')
    print(f"议题分布已导出至: {topic_out}")

    # 2. 情感倾向占比
    # 使用同一份清洗后的 df_work（已保留情感得分）
    # 重新定义情感倾向函数（与 load_and_preprocess_data 中一致）
    def get_polarity(score):
        if score > 0.0:
            return "正向"
        elif score < -0.0:
            return "负向"
        else:
            return "中立"

    df_senti = df_work.copy()
    df_senti['情感倾向'] = df_senti['情感得分'].apply(get_polarity)
    senti_counts = df_senti.groupby(['账号分类', '情感倾向']).size().reset_index(name='数量')
    total_per_cat2 = df_senti.groupby('账号分类').size().reset_index(name='总数')
    senti_counts = senti_counts.merge(total_per_cat2, on='账号分类')
    senti_counts['占比'] = (senti_counts['数量'] / senti_counts['总数']).apply(lambda x: f"{x:.2%}")
    senti_counts = senti_counts[['账号分类', '情感倾向', '数量', '占比']]
    senti_counts.sort_values(['账号分类', '情感倾向'], inplace=True)

    senti_out = os.path.join(output_dir, "账号分类情感倾向占比.csv")
    senti_counts.to_csv(senti_out, index=False, encoding='utf-8-sig')
    print(f"情感倾向占比已导出至: {senti_out}")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"找不到输入文件: {INPUT_FILE}")
        return

    Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 用于共识分析的数据（仅 public）
    df_public = load_and_preprocess_data(INPUT_FILE, keep_account_types=['public'])

    # 用于最后导出的数据（所有账号分类）
    df_all = load_and_preprocess_data(INPUT_FILE, keep_account_types=None)

    # 后续所有基于公众的分析使用 df_public
    daily_scores = build_daily_consensus_scores(df_public)
    daily_score_path = os.path.join(BASE_OUTPUT_DIR, "每日社会共识度得分.csv")
    daily_scores.to_csv(daily_score_path, index=False, encoding='utf-8-sig')
    missing_days = int(daily_scores['社会共识度得分'].isna().sum())
    low_sample_days = int(daily_scores['低样本日'].sum())
    print(
        f"已导出PDF日际分析所需的每日社会共识度得分至: {daily_score_path} (共 {len(daily_scores)} 天，缺失 {missing_days} 天，低样本 {low_sample_days} 天)")

    if not df_public.empty and 'themes_list' in df_public.columns:
        exploded_themes = df_public.explode('themes_list')
        theme_counts = exploded_themes['themes_list'].value_counts().reset_index()
        theme_counts.columns = ['主题', '频次']
        total_themes = theme_counts['频次'].sum()
        theme_counts['占比'] = (theme_counts['频次'] / total_themes).apply(lambda x: f"{x * 100:.2f}%")
        themes_overall_path = os.path.join(BASE_OUTPUT_DIR, f"{TOTAL_THEMES_COUNT}个主题整体占比.csv")
        theme_counts.to_csv(themes_overall_path, index=False, encoding='utf-8-sig')
        print(f"已成功计算并导出【{TOTAL_THEMES_COUNT}个主题整体占比】至: {themes_overall_path}")

    polarities = ["正向情感圈层", "负向情感圈层"]
    # df_7_days = pd.DataFrame()
    df_14_days = pd.DataFrame()
    consensus_topic_details = pd.DataFrame()
    all_metrics_df = pd.DataFrame()  # 用于收集所有窗口数据以供新功能1调用

    for polarity in polarities:
        for label, config in WINDOW_CONFIG.items():
            res_df, topic_details_df = analyze_consensus_periods(
                df=df_public, polarity=polarity, window_label=label,
                window_days=config['window_days'], step_days=config['step_days']
            )
            if not res_df.empty:
                all_metrics_df = pd.concat([all_metrics_df, res_df], ignore_index=True)
                if "7天" in label:
                    df_7_days = pd.concat([df_7_days, res_df], ignore_index=True)
                elif "3天" in label:
                    df_3_days = pd.concat([df_3_days, res_df], ignore_index=True)
                elif "14天" in label:
                    df_14_days = pd.concat([df_14_days, res_df], ignore_index=True)
            if not topic_details_df.empty:
                consensus_topic_details = pd.concat([consensus_topic_details, topic_details_df], ignore_index=True)

    # if not df_7_days.empty:
    #     out_7 = os.path.join(BASE_OUTPUT_DIR, "7天刻度_全情感共识汇总数据.csv")
    #     df_7_days.to_csv(out_7, index=False, encoding='utf-8-sig')
    #     print(f"\n成功将所有情感的7天刻度数据合并导出至: {out_7} (共 {len(df_7_days)} 条记录)")

    if not df_14_days.empty:
        out_14 = os.path.join(BASE_OUTPUT_DIR, "14天刻度_全情感共识汇总数据.csv")
        df_14_days.to_csv(out_14, index=False, encoding='utf-8-sig')
        print(f"\n成功将所有情感的14天刻度数据合并导出至: {out_14} (共 {len(df_14_days)} 条记录)")

    # if not df_3_days.empty:
    #     out_3 = os.path.join(BASE_OUTPUT_DIR, "3天刻度_全情感共识汇总数据.csv")
    #     df_3_days.to_csv(out_3, index=False, encoding='utf-8-sig')
    #     print(f"\n成功将所有情感的3天刻度数据合并导出至: {out_3} (共 {len(df_3_days)} 条记录)")

    # if not consensus_topic_details.empty:
    #     detail_path = os.path.join(BASE_OUTPUT_DIR, "7天刻度_共识窗口全部议题明细.csv")
    #     consensus_topic_details.to_csv(detail_path, index=False, encoding='utf-8-sig')
    #     print(
    #         f"成功导出逐共识窗口逐议题长表至: {detail_path} "
    #         f"(共 {len(consensus_topic_details)} 条记录)"
    #     )

    if not consensus_topic_details.empty:
        detail_path = os.path.join(BASE_OUTPUT_DIR, "14天刻度_共识窗口全部议题明细.csv")
        consensus_topic_details.to_csv(detail_path, index=False, encoding='utf-8-sig')
        print(
            f"成功导出逐共识窗口逐议题长表至: {detail_path} "
            f"(共 {len(consensus_topic_details)} 条记录)"
        )

    export_specific_window_comments(df_public, all_metrics_df, BASE_OUTPUT_DIR, TARGET_WINDOW_IDS)


if __name__ == "__main__":
    main()