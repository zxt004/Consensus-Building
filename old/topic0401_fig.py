"""
阶段 2：预定义主题演化趋势图（河流图+情感指数）
新增功能：导出每月统计数据 CSV
"""
from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==================== 配置区 ====================
INPUT_CSV = "/root/pmd/processed_data/ai_summarized_with_themes.csv"
OUTPUT_STREAM_CHART = "./processed_data/topic_evolution_streamgraph.svg"
OUTPUT_STATS_CSV = "./processed_data/monthly_topic_statistics.csv"

# 预定义的九大主题 (已更新为最新表述)
THEMES_MAPPING = {
    "个体消费行为倾向": ["面条", "没吃过", "不吃", "不去西贝", "反正", "去过", "以后", "好吃"],
    "食品安全与营养认知": ["校园", "食品", "冷冻", "食品安全", "营养", "食堂", "学生", "新鲜"],
    "消费者与企业家立场差异": ["罗永浩", "消费者", "贾国龙", "老板", "顾客"],
    "技术风险认知": ["鱼刺", "辐射", "残留", "检测", "物质", "能量", "鱼肉", "去除"],
    "社会发展隐喻讨论": ["快乐教育", "美国", "孩子", "大学", "教学", "学习", "预制菜"],
    "国际品牌参照与消费预期": ["萨莉亚", "香港", "照片", "老周", "360", "罗永浩", "医生", "没有萨莉亚"],
    "产品供应链透明度关注": ["西贝货", "假货", "西贝", "蜀海", "厨房"],
    "语言表达与身份标签": ["网络黑社会", "老罗", "真正·黑社会", "罗永浩", "龙哥", "内蒙"],
    "历史经验与系统信任": ["奶粉", "牛奶", "三鹿", "狗屎", "三聚氰胺", "内蒙", "标准", "现挤", "国标"]
}
VALID_THEMES = list(THEMES_MAPPING.keys())


def get_topic_color(idx: int, total_topics: int) -> str:
    """生成具备高区分度的主题颜色"""
    distinct_palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
        "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
        "#800080", "#0000ff"
    ]
    if idx < len(distinct_palette):
        return distinct_palette[idx]

    golden_ratio_conjugate = 0.618033988749895
    hue = (idx * golden_ratio_conjugate) % 1.0
    saturation = 0.6 + (idx % 3) * 0.15
    value = 0.75 + (idx % 2) * 0.15
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    print(f"\n[1/3] 加载数据与预处理：{csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "ai_summary" in df.columns:
        initial_len = len(df)

        # 去空值：剔除 ai_summary 为空的行
        df = df.dropna(subset=["ai_summary"])
        # 去重：剔除完全重复的行
        df = df.drop_duplicates(subset=["ai_summary"])
        # 过滤无效模式：剔除包含无意义表述的行
        invalid_patterns = "未提及|未涉及|无法提炼|无法概括|无法得出|没有涉及|不涉及"
        df = df[~df["ai_summary"].astype(str).str.contains(invalid_patterns, regex=True, na=False)]

        filtered_len = initial_len - len(df)
        print(f"  -> [清理] 已基于 'ai_summary' 完成去重、去空和无效内容过滤，共剔除 {filtered_len} 行数据。")
    else:
        print("  -> [警告] 数据中未检测到 'ai_summary' 列，已跳过初步清理。")

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

    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0
    else:
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0)

    if "匹配的主题标签" not in df.columns:
        raise ValueError("CSV中未找到 '匹配的主题标签' 列。")

    df = df[df["匹配的主题标签"].notna()]
    df = df[df["匹配的主题标签"].astype(str).str.strip() != "无匹配"]

    old_to_new_mapping = {
        "个体消费抗拒行为": "个体消费行为倾向",
        "食品安全与营养争议": "食品安全与营养认知",
        "消费者与企业家的对立立场": "消费者与企业家立场差异",
        "技术风险感知": "技术风险认知",
        "社会发展隐喻争议": "社会发展隐喻讨论",
        "国际品牌参照下的预期管理": "国际品牌参照与消费预期",
        "产品供应链真实性质疑": "产品供应链透明度关注",
        "语言暴力与身份污名化": "语言表达与身份标签",
        "历史创伤与信任危机": "历史经验与系统信任"
    }

    def extract_valid_themes(theme_str: str) -> list[str]:
        cleaned_str = re.sub(r'\([0-9.]+\)', '', str(theme_str))
        extracted_themes = [t.strip() for t in cleaned_str.split('|') if t.strip()]

        valid_extracted = []
        for t in extracted_themes:
            # 如果是旧标签，就映射成新的；如果是已经是新的，就保持不变
            mapped_theme = old_to_new_mapping.get(t, t)
            if mapped_theme in VALID_THEMES:
                valid_extracted.append(mapped_theme)

        return valid_extracted

    df["themes_list"] = df["匹配的主题标签"].apply(extract_valid_themes)
    df = df[df["themes_list"].map(len) > 0].copy()
    df['period'] = df['时间'].dt.to_period('M').dt.to_timestamp()
    df = df.dropna(subset=['period'])

    print(f"  ✓ 数据加载完毕。有效数据 {len(df)} 行。")
    return df


def plot_comprehensive_chart(df: pd.DataFrame, output_img: str, output_csv: str):
    print("\n[2/3] 数据聚合与指标计算...")

    # 1. 字体设置
    from matplotlib import font_manager
    font_path = "/root/pmd/SIMSUN.TTC"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 计算情感均值
    sentiment_idx = df.groupby('period')['sentiment_score'].mean()

    # 3. 计算主题频次 (展开多标签)
    df_exploded = df.explode('themes_list').reset_index(drop=True)
    topic_counts = pd.crosstab(df_exploded['period'], df_exploded['themes_list'])

    for t in VALID_THEMES:
        if t not in topic_counts.columns:
            topic_counts[t] = 0
    topic_counts = topic_counts[VALID_THEMES]

    dates = topic_counts.index
    if len(dates) == 0:
        print("  ! 数据为空，无法绘图。")
        return

    # --- 保存统计数据到 CSV ---
    stats_df = topic_counts.copy()
    stats_df['Sentiment_Index'] = sentiment_idx
    # 格式化日期索引
    stats_df.index = stats_df.index.strftime('%Y-%m')
    stats_df.index.name = "月份"

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_csv, encoding="utf-8-sig")
    print(f"  ✓ 每月统计数据已保存至：{output_csv}")
    # ------------------------------

    # 4. 绘图
    print("\n[3/3] 正在生成图表...")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    y = [np.log1p(topic_counts[c].values) for c in topic_counts.columns]
    colors = [get_topic_color(i, len(VALID_THEMES)) for i in range(len(VALID_THEMES))]

    ax1.stackplot(dates, y, labels=VALID_THEMES, baseline='sym', colors=colors, alpha=0.65)
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('单月话题分布对数规模', fontsize=12, color='gray')
    ax1.set_yticklabels([])
    ax1.tick_params(axis='y', length=0)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), frameon=False, ncol=3, fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(dates, sentiment_idx.reindex(dates).values, color='blue', linewidth=2.5, linestyle='--', marker='s', label='情感均值指数')
    ax2.set_ylabel('情感指数', fontsize=12)

    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_2, labels_2, loc='lower right', bbox_to_anchor=(1.0, 1.02), frameon=False, ncol=1, fontsize=10)

    plt.title('舆情演化河流图与情感指数综合趋势', fontsize=18, y=1.16)
    fig.tight_layout()
    plt.savefig(output_img, format='svg')
    plt.close()
    print(f"  ✓ 综合图表已保存至：{output_img}")


def main():
    print("====== 主题演化综合趋势分析 ======")
    df = load_and_preprocess_data(INPUT_CSV)
    plot_comprehensive_chart(df, OUTPUT_STREAM_CHART, OUTPUT_STATS_CSV)
    print("==================================")


if __name__ == "__main__":
    main()