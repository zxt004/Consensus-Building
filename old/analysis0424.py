from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats
import ruptures as rpt

warnings.filterwarnings("ignore")

INPUT_CSV = "/root/pmd/processed_data/ai_summarized_with_themes.csv"
BASE_OUTPUT_DIR = "./processed_data/advanced_analysis/"

THEMES_MAPPING = {
    "食品安全与营养争议": ["校园", "食品", "冷冻", "食品安全", "营养", "食堂", "学生", "新鲜"],
    "消费者与企业家的对立立场": ["罗永浩", "消费者", "贾国龙", "老板", "顾客"],
    "产品供应链真实性质疑": ["西贝货", "假货", "西贝", "蜀海", "厨房"],
    "社会发展隐喻争议": ["快乐教育", "美国", "孩子", "大学", "教学", "学习", "预制菜"],
    "个体消费抗拒行为": ["面条", "没吃过", "不吃", "不去西贝", "反正", "去过", "以后", "好吃"],
    "历史创伤与信任危机": ["奶粉", "牛奶", "三鹿", "狗屎", "三聚氰胺", "内蒙", "标准", "现挤", "国标"],
    "语言暴力与身份污名化": ["网络黑社会", "老罗", "真正·黑社会", "罗永浩", "龙哥", "内蒙"],
    "国际品牌参照下的预期管理": ["萨莉亚", "香港", "照片", "老周", "360", "罗永浩", "医生", "没有萨莉亚"],
    "技术风险感知": ["鱼刺", "辐射", "残留", "检测", "物质", "能量", "鱼肉", "去除"]
}
VALID_THEMES = list(THEMES_MAPPING.keys())
TOTAL_THEMES_COUNT = len(VALID_THEMES)

FREQ_CONFIG = {
    "双周": {"freq": "2W"},
    "月": {"freq": "ME"}
}


def get_stage(date_obj: pd.Timestamp) -> str:
    if pd.isna(date_obj): return "未知阶段"
    if date_obj <= pd.to_datetime('2023-08-31'):
        return "阶段1 (以前~2023.8)"
    elif date_obj <= pd.to_datetime('2024-02-29'):
        return "阶段2 (2023.9~2024.2)"
    elif date_obj <= pd.to_datetime('2025-08-31'):
        return "阶段3 (2024.3~2025.8)"
    else:
        return "阶段4 (2025.9~至今)"


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    print(f"\n加载数据与预处理...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "ai_summary" in df.columns:
        df = df.dropna(subset=["ai_summary"]).drop_duplicates(subset=["ai_summary"])
        invalid_patterns = "未提及|未涉及|无法提炼|无法概括|无法得出|没有涉及|不涉及"
        df = df[~df["ai_summary"].astype(str).str.contains(invalid_patterns, regex=True, na=False)]

    time_col_found = False
    for tcand in ("时间", "创建时间", "发布时间", "parsed_time"):
        if tcand in df.columns:
            clean_time = df[tcand].astype(str).str.replace(r'(?:发布于|编辑于)\s*', '', regex=True).str.strip()
            time_strs = clean_time.str.extract(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}(?:\s+\d{1,2}:\d{1,2})?)')[0]
            time_strs = time_strs.str.replace("年", "-").str.replace("月", "-").str.replace("日", "")
            df["时间"] = pd.to_datetime(time_strs, errors="coerce")
            time_col_found = True
            break

    if not time_col_found or "时间" not in df.columns:
        df["时间"] = pd.NaT

    df['sentiment_score'] = pd.to_numeric(df.get('sentiment_score', 0), errors='coerce').fillna(0)
    df = df.dropna(subset=['时间'])
    df = df[df["匹配的主题标签"].notna()]
    df = df[df["匹配的主题标签"].astype(str).str.strip() != "无匹配"]

    def extract_valid_themes(theme_str: str) -> list[str]:
        cleaned = re.sub(r'\([0-9.]+\)', '', str(theme_str))
        extracted = [t.strip() for t in cleaned.split('|') if t.strip()]
        return [t for t in extracted if t in VALID_THEMES]

    df["themes_list"] = df["匹配的主题标签"].apply(extract_valid_themes)
    df = df[df["themes_list"].map(len) > 0]

    return df


def calculate_window_metrics(group_df: pd.DataFrame) -> pd.Series:
    total_posts = len(group_df)

    if total_posts < 20:
        return pd.Series({'TopicFocus': np.nan, 'EmotionConsistency': np.nan, 'ConsensusScore': np.nan, 'PostCount': total_posts})

    themes_exploded = group_df.explode('themes_list')
    theme_counts = themes_exploded['themes_list'].value_counts()

    if len(theme_counts) == 0:
        topic_focus = 0.0
    else:
        p_i = theme_counts.values / theme_counts.sum()
        entropy = scipy.stats.entropy(p_i, base=np.e)
        max_entropy = np.log(TOTAL_THEMES_COUNT)
        topic_focus = 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    sentiment_std = group_df['sentiment_score'].std(ddof=0)
    if pd.isna(sentiment_std): sentiment_std = 1.0

    # 使用方差来计算情感一致性，对极端发散的惩罚更符合逻辑 (假设 score 归一化在 [-1, 1] 附近)
    emotion_consistency = max(0.0, 1 - (sentiment_std ** 2))

    if topic_focus + emotion_consistency > 0:
        consensus_score = (2 * topic_focus * emotion_consistency) / (topic_focus + emotion_consistency)
    else:
        consensus_score = 0.0

    return pd.Series({
        'TopicFocus': topic_focus,
        'EmotionConsistency': emotion_consistency,
        'ConsensusScore': consensus_score,
        'PostCount': total_posts
    })


def analyze_consensus_periods(df: pd.DataFrame, freq_label: str, freq: str):
    print(f"\n========== 按照 {freq_label} ({freq}) 刻度进行共识分析 ==========")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"Consensus_{freq_label}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    grouper = pd.Grouper(key='时间', freq=freq)
    metrics_df = df.groupby(grouper).apply(calculate_window_metrics).reset_index()

    metrics_df = metrics_df.dropna(subset=['ConsensusScore']).copy()
    metrics_df = metrics_df.reset_index(drop=True)  # 重置索引便于后续变点位置计算

    if metrics_df.empty:
        print(f"数据稀疏，无满足要求的 {freq_label} 窗口。")
        return

    # 方案A判定：双阈值
    metrics_df['IsConsensus'] = (metrics_df['TopicFocus'] >= 0.4) & (metrics_df['EmotionConsistency'] >= 0.4)
    metrics_df['Stage'] = metrics_df['时间'].apply(get_stage)

    # 变点检测及方向判断
    signal = metrics_df['ConsensusScore'].values
    changepoints = []

    if len(signal) > 5:
        algo = rpt.Pelt(model="l2", min_size=2).fit(signal)
        penalty = np.log(len(signal)) * signal.var() * 3
        try:
            cps_indices = algo.predict(pen=penalty)

            for cp_idx in cps_indices:
                if cp_idx >= len(metrics_df):
                    continue  # 跳过序列末尾的虚拟变点

                # 计算变点前后的均值以判断方向（窗口大小设为3）
                window = 3
                start_before = max(0, cp_idx - window)
                end_after = min(len(signal), cp_idx + window)

                mean_before = signal[start_before:cp_idx].mean() if cp_idx > 0 else 0
                mean_after = signal[cp_idx:end_after].mean()

                cp_direction = "上升" if mean_after > mean_before else "下降"

                changepoints.append({
                    'date': metrics_df.iloc[cp_idx]['时间'],
                    'idx': cp_idx,
                    'direction': cp_direction,
                    'mean_before': mean_before,
                    'mean_after': mean_after
                })
        except Exception as e:
            print(f"变点检测失败: {e}")

    # 提取短暂共识持续期
    periods = []
    current_period_indices = []

    for idx, row in metrics_df.iterrows():
        if row['IsConsensus']:
            current_period_indices.append(idx)
        else:
            if len(current_period_indices) >= 1:  # 至少连续2个窗口
                periods.append(metrics_df.loc[current_period_indices])
            current_period_indices = []

    if len(current_period_indices) >= 1:
        periods.append(metrics_df.loc[current_period_indices])

    # 输出阶段统计结果
    stages = [
        "阶段1 (以前~2023.8)",
        "阶段2 (2023.9~2024.2)",
        "阶段3 (2024.3~2025.8)",
        "阶段4 (2025.9~至今)"
    ]

    report_lines = []

    for stage in stages:
        report_lines.append(f"\n【 {stage} 】")

        # 过滤属于该阶段的持续期和变点
        stage_periods = [p for p in periods if p['Stage'].iloc[0] == stage]
        stage_cps = [cp for cp in changepoints if get_stage(cp['date']) == stage]

        if not stage_periods and not stage_cps:
            report_lines.append("无短暂共识持续期或显著转折点。")
            continue

        if stage_periods:
            for idx, p in enumerate(stage_periods, 1):
                start_idx = p.index[0]
                end_idx = p.index[-1]

                start_date = p['时间'].iloc[0].strftime('%Y.%m.%d')
                end_date = p['时间'].iloc[-1].strftime('%Y.%m.%d')

                peak_row = p.loc[p['ConsensusScore'].idxmax()]
                peak_date = peak_row['时间'].strftime('%Y.%m.%d')
                peak_score = peak_row['ConsensusScore']

                report_lines.append(f"  * 短暂共识持续期{idx}：{start_date} - {end_date} (共 {len(p)} 个窗口)")
                report_lines.append(f"    - 峰值窗口：{peak_date} (共识分={peak_score:.2f})")

                # 首尾匹配变点 (容错率：前后 2 个窗口内)
                # 匹配起点：寻找附近的"上升"变点
                matching_start_cps = [cp for cp in changepoints if
                                      cp['direction'] == '上升' and abs(cp['idx'] - start_idx) <= 2]
                if matching_start_cps:
                    cps_str = ", ".join([d['date'].strftime('%Y.%m.%d') for d in matching_start_cps])
                    report_lines.append(f"    - 共识达成转折点 (上升变点)：{cps_str}")
                else:
                    report_lines.append(f"    - 共识达成转折点：未检测到显著数值突变，属平缓过渡")

                # 匹配终点：寻找附近的"下降"变点 (断裂往往发生在结束周期的下一个窗口)
                matching_end_cps = [cp for cp in changepoints if
                                    cp['direction'] == '下降' and abs(cp['idx'] - (end_idx + 1)) <= 2]
                if matching_end_cps:
                    cps_str = ", ".join([d['date'].strftime('%Y.%m.%d') for d in matching_end_cps])
                    report_lines.append(f"    - 共识断裂转折点 (下降变点)：{cps_str}")
                else:
                    report_lines.append(f"    - 共识断裂转折点：未检测到显著数值突变，属自然衰减")
        else:
            report_lines.append("  * 本阶段内无满足连续长度的共识区间。")

        # 列出阶段内未匹配到具体共识区间的"孤立变点"
        if stage_cps:
            matched_cp_indices = []
            for p in stage_periods:
                start_idx, end_idx = p.index[0], p.index[-1]
                matched_cp_indices.extend([cp['idx'] for cp in changepoints if
                                           abs(cp['idx'] - start_idx) <= 2 or abs(cp['idx'] - (end_idx + 1)) <= 2])

            isolated_cps = [cp for cp in stage_cps if cp['idx'] not in matched_cp_indices]
            if isolated_cps:
                report_lines.append("  * 其他剧烈震荡点（未形成稳定共识区间）：")
                for cp in isolated_cps:
                    report_lines.append(f"    - {cp['date'].strftime('%Y.%m.%d')} (方向: {cp['direction']})")

    print("\n".join(report_lines))

    metrics_df['时间'] = metrics_df['时间'].dt.strftime('%Y-%m-%d')
    metrics_df.to_csv(os.path.join(output_dir, "consensus_metrics_timeline.csv"), index=False, encoding='utf-8-sig')

    with open(os.path.join(output_dir, "consensus_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"共识周期与转折点分析报告 ({freq_label})\n")
        f.write("\n".join(report_lines))


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"找不到输入文件: {INPUT_CSV}")
        return

    df = load_and_preprocess_data(INPUT_CSV)

    for label, config in FREQ_CONFIG.items():
        analyze_consensus_periods(df, label, config['freq'])


if __name__ == "__main__":
    main()