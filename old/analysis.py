from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

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

FREQ_CONFIG = {
    "周": {"freq": "W-MON", "offset": pd.DateOffset(days=6)},
    "月": {"freq": "M", "offset": pd.DateOffset(months=1)},
    "两个月": {"freq": "2M", "offset": pd.DateOffset(months=2)},
    "季度": {"freq": "Q", "offset": pd.DateOffset(months=3)}
}


def setup_chinese_font():
    font_path = "/root/pmd/SIMSUN.TTC"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


# 数据加载与基础预处理
def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    print(f"\n加载数据与预处理")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "ai_summary" in df.columns:
        df = df.dropna(subset=["ai_summary"])
        df = df.drop_duplicates(subset=["ai_summary"])
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
    df = df[df["匹配的主题标签"].notna()]
    df = df[df["匹配的主题标签"].astype(str).str.strip() != "无匹配"]

    def extract_valid_themes(theme_str: str) -> list[str]:
        cleaned = re.sub(r'\([0-9.]+\)', '', str(theme_str))
        extracted = [t.strip() for t in cleaned.split('|') if t.strip()]
        return [t for t in extracted if t in VALID_THEMES]

    df["themes_list"] = df["匹配的主题标签"].apply(extract_valid_themes)
    df = df[df["themes_list"].map(len) > 0]

    return df


# 根据指定频率聚合数据
def aggregate_time_series(df: pd.DataFrame, freq: str):
    try:
        df['period'] = df['时间'].dt.to_period(freq).dt.to_timestamp()
    except ValueError:
        alt_freq = freq.replace('M', 'ME')
        df['period'] = df['时间'].dt.to_period(alt_freq).dt.to_timestamp()

    df_valid = df.dropna(subset=['period'])
    df_ex = df_valid.explode('themes_list').reset_index(drop=True)

    topic_counts = pd.crosstab(df_ex['period'], df_ex['themes_list'])
    for t in VALID_THEMES:
        if t not in topic_counts.columns: topic_counts[t] = 0
    topic_counts = topic_counts[VALID_THEMES].sort_index()

    sentiment_means = df_ex.groupby(['period', 'themes_list'])['sentiment_score'].mean().unstack(fill_value=0)
    for t in VALID_THEMES:
        if t not in sentiment_means.columns: sentiment_means[t] = 0
    sentiment_means = sentiment_means[VALID_THEMES].sort_index()

    sentiment_idx = df_valid.groupby('period')['sentiment_score'].mean()
    text_counts = df_valid.groupby('period').size()

    return topic_counts, sentiment_means, sentiment_idx, text_counts


# 同向/排斥计算与聚类
def analyze_correlation_and_cluster(topic_counts: pd.DataFrame, output_dir: str, n_clusters=2):
    topic_proportions = topic_counts.div(topic_counts.sum(axis=1), axis=0).fillna(0)
    cols = topic_proportions.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                corr_matrix.loc[c1, c2] = 1.0
                pval_matrix.loc[c1, c2] = 0.0
            else:
                try:
                    r, p = pearsonr(topic_proportions[c1], topic_proportions[c2])
                except:
                    r, p = 0.0, 1.0
                corr_matrix.loc[c1, c2] = r
                pval_matrix.loc[c1, c2] = p

    corr_matrix.to_csv(os.path.join(output_dir, "topic_correlation.csv"), encoding="utf-8-sig", float_format="%.5f")
    pval_matrix.to_csv(os.path.join(output_dir, "topic_correlation_pvalue.csv"), encoding="utf-8-sig",
                       float_format="%.5f")

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            r = corr_matrix.loc[c1, c2]
            p = pval_matrix.loc[c1, c2]
            rel_type = "无明显关系"
            if p < 0.05:
                rel_type = "同向关系" if r > 0 else "排斥关系"
            pairs.append(
                {'议题A': c1, '议题B': c2, '相关系数(r)': round(r, 5), '显著性(p)': round(p, 5), '关系类型': rel_type})

    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(os.path.join(output_dir, "topic_relationships_classified.csv"), index=False, encoding="utf-8-sig")

    distance_matrix = (1 - corr_matrix.fillna(0).values) / 2
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, 2)
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = sch.linkage(condensed_dist, method='average')
    cluster_labels = sch.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

    topic_clusters = {topic: int(cluster) for topic, cluster in zip(VALID_THEMES, cluster_labels)}
    clusters_dict = {}
    for topic, c_id in topic_clusters.items():
        clusters_dict.setdefault(c_id, []).append(topic)

    return corr_matrix, topic_clusters, clusters_dict


def generate_cluster_colors(topic_clusters: dict, clusters_dict: dict, theme_order=None):
    if theme_order is None: theme_order = VALID_THEMES
    base_hues = {1: 0.0, 2: 0.6, 3: 0.33, 4: 0.15}
    topic_colors = {}
    for c_id, topics in clusters_dict.items():
        hue = base_hues.get(c_id, np.random.rand())
        for i, topic in enumerate(topics):
            saturation = 0.5 + (i / max(1, len(topics))) * 0.4
            value = 0.9 - (i / max(1, len(topics))) * 0.3
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            topic_colors[topic] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    return [topic_colors[t] for t in theme_order]


# 绘制河流图 & 情感图
def plot_clustered_streamgraph(topic_counts: pd.DataFrame, sentiment_idx: pd.Series,
                               topic_clusters: dict, clusters_dict: dict, consensus_df: pd.DataFrame,
                               output_dir: str, freq_config: dict):
    setup_chinese_font()
    if topic_counts.empty: return

    try:
        full_periods = pd.period_range(start=topic_counts.index.min(), end=topic_counts.index.max(),
                                       freq=freq_config['freq']).to_timestamp()
    except ValueError:
        alt_freq = freq_config['freq'].replace('M', 'ME')
        full_periods = pd.period_range(start=topic_counts.index.min(), end=topic_counts.index.max(),
                                       freq=alt_freq).to_timestamp()

    stream_topic_counts = topic_counts.reindex(full_periods).interpolate(method='time').fillna(0)

    sorted_themes = sorted(VALID_THEMES, key=lambda t: topic_clusters[t], reverse=True)
    colors = generate_cluster_colors(topic_clusters, clusters_dict, theme_order=sorted_themes)

    # 主题河流图
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    transformed_df = np.log1p(stream_topic_counts)
    y = [transformed_df[c].values for c in sorted_themes]
    ax1.stackplot(stream_topic_counts.index, y, labels=sorted_themes, baseline='sym', colors=colors, alpha=0.65)
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('话题热度 (Ln对数分布)', fontsize=12, color='gray')
    ax1.set_yticklabels([])
    ax1.tick_params(axis='y', length=0)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), frameon=False, ncol=3, fontsize=10)

    # 自适应时间轴
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.title('舆情演化河流图', fontsize=18, y=1.16)
    ax1.set_xlim(full_periods.min(), full_periods.max())
    fig1.tight_layout()
    plt.savefig(os.path.join(output_dir, "clustered_streamgraph.svg"), format='svg')
    plt.close(fig1)

    # 情感指数与双阵营演化图
    fig2, ax_hhi = plt.subplots(figsize=(24, 6))
    ax_hhi.set_xlabel('时间', fontsize=12)

    shifted_topic_counts_idx = topic_counts.index
    shifted_sentiment_idx = sentiment_idx.index
    true_total_counts = topic_counts.sum(axis=1)

    if 1 in clusters_dict:
        c1_prop = topic_counts[clusters_dict[1]].sum(axis=1) / true_total_counts
        ax_hhi.plot(shifted_topic_counts_idx, c1_prop.values, color='red', linewidth=2, linestyle='-', marker='^',
                    markersize=5, label='阵营1占比')

    if 2 in clusters_dict:
        c2_prop = topic_counts[clusters_dict[2]].sum(axis=1) / true_total_counts
        ax_hhi.plot(shifted_topic_counts_idx, c2_prop.values, color='green', linewidth=2, linestyle='-', marker='v',
                    markersize=5, label='阵营2占比')

    ax_hhi.set_ylabel('阵营占比', fontsize=12)
    ax_hhi.axhline(y=0.66, color='grey', linestyle=':', alpha=0.8, label='高占比阈值 (0.75)')
    ax_hhi.axhline(y=0.50, color='grey', linestyle=':', alpha=0.4, label='极化中线 (0.50)')
    ax_hhi.set_ylim(0, 1.05)

    ax_sent = ax_hhi.twinx()
    ax_sent.plot(shifted_sentiment_idx, sentiment_idx.values, color='blue', linewidth=2.5, linestyle='--', marker='s',
                 label='情感均值指数')
    ax_sent.set_ylabel('情感均值指数', fontsize=12, color='blue')
    ax_sent.axhline(y=0.625, color='cornflowerblue', linestyle='--', alpha=0.6, label='情感阈值 (±0.625)')
    ax_sent.axhline(y=-0.625, color='cornflowerblue', linestyle='--', alpha=0.6)
    ax_sent.axhline(y=0.40, color='cornflowerblue', linestyle=':', alpha=0.6, label='脆弱情感阈值 (±0.40)')
    ax_sent.axhline(y=-0.40, color='cornflowerblue', linestyle=':', alpha=0.6)

    ax_hhi.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_hhi.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax_hhi.get_xticklabels(), rotation=35, ha="right")

    offset = freq_config['offset']
    if consensus_df is not None and not consensus_df.empty:
        active_points = consensus_df[consensus_df['状态'] != '否']
        for i, (idx, row) in enumerate(active_points.iterrows()):
            d = pd.to_datetime(row['Date'])
            state = row['状态']

            if '强共识' in state:
                span_color, text_color = 'gold', 'darkgoldenrod'
            elif '脆弱共识' in state:
                span_color, text_color = 'lightblue', 'teal'
            elif '极化' in state:
                span_color, text_color = 'lightcoral', 'darkred'
            else:
                span_color, text_color = 'gray', 'black'

            d_end = d + offset
            ax_hhi.axvspan(d, d_end, color=span_color, alpha=0.3)

            ax_hhi.annotate(state, xy=(d + (d_end - d) / 2, 1.03), ha='center', fontsize=9,
                            color=text_color, fontweight='bold', rotation=30,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    lines_hhi, labels_hhi = ax_hhi.get_legend_handles_labels()
    lines_sent, labels_sent = ax_sent.get_legend_handles_labels()
    ax_hhi.legend(lines_hhi + lines_sent, labels_hhi + labels_sent,
                  loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  frameon=False, ncol=4, fontsize=10, handletextpad=0.6, columnspacing=1.8)

    plt.title('情感均值指数与双阵营占比演化趋势', fontsize=16, pad=20)
    ax_hhi.set_xlim(full_periods.min(), full_periods.max())
    fig2.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.25)
    plt.savefig(os.path.join(output_dir, "sentiment_trend.svg"), format='svg', dpi=300)
    plt.close(fig2)


# 各周期议题占比与阵营占比
def export_proportions_and_camps(topic_counts: pd.DataFrame, clusters_dict: dict, output_dir: str):
    total_counts = topic_counts.sum(axis=1)
    topic_proportions = topic_counts.div(total_counts, axis=0).fillna(0)
    camp_proportions = pd.DataFrame(index=topic_proportions.index)
    for camp_id, topics in clusters_dict.items():
        if topics:
            camp_proportions[f'阵营{camp_id}占比'] = topic_proportions[topics].sum(axis=1)
        else:
            camp_proportions[f'阵营{camp_id}占比'] = 0.0

    combined_df = pd.concat([topic_proportions, camp_proportions], axis=1)

    combined_df.index = combined_df.index.strftime('%Y-%m-%d')
    combined_df.index.name = '时间周期起始'

    output_path = os.path.join(output_dir, "topic_and_camp_proportions.csv")
    combined_df.to_csv(output_path, encoding="utf-8-sig", float_format="%.5f")

    return combined_df

# 共识观测点提取
def extract_consensus_points(topic_counts: pd.DataFrame, sentiment_idx: pd.Series,
                             clusters_dict: dict, text_counts: pd.Series, output_dir: str, freq_config: dict):
    total_counts = topic_counts.sum(axis=1)
    proportions = topic_counts.div(total_counts, axis=0).fillna(0)

    cluster_1_topics = clusters_dict.get(1, [])
    cluster_2_topics = clusters_dict.get(2, [])
    c1_prop = proportions[cluster_1_topics].sum(axis=1) if cluster_1_topics else pd.Series(0, index=proportions.index)
    c2_prop = proportions[cluster_2_topics].sum(axis=1) if cluster_2_topics else pd.Series(0, index=proportions.index)

    results = []
    for date in proportions.index:
        sent = sentiment_idx.get(date, 0)
        c1_p, c2_p = c1_prop.get(date, 0), c2_prop.get(date, 0)
        t_count = text_counts.get(date, 0)
        abs_sent = abs(sent)

        state = "否"
        c_id_found, dominant_topics_found, consensus_prop = "", "", ""

        if abs(c1_p - c2_p) <= 0.2 and (c1_p >= 0.35 and c2_p >= 0.35) and abs_sent >= 0.625:
            state = "极化"
        else:
            for c_id, topics in clusters_dict.items():
                cluster_prop = proportions.loc[date, topics].sum() if date in proportions.index else 0
                if cluster_prop >= 0.66:
                    if abs_sent >= 0.625:
                        state = "强共识"
                    elif 0.4 <= abs_sent < 0.625:
                        state = "脆弱共识"
                    if state != "否":
                        c_id_found = c_id
                        dominant_topics_found = " | ".join(topics)
                        consensus_prop = cluster_prop
                    break

        if state != "否" and t_count < 10:
            state += "(数量少)"

        d_start = date.strftime('%Y-%m-%d')
        d_end = (date + freq_config['offset']).strftime('%Y-%m-%d')

        results.append({
            "Date": d_start,
            "Period_Info": f"{d_start} 至 {d_end}",
            "该周期总文本数": t_count,
            "阵营1占比": round(c1_p, 5),
            "阵营2占比": round(c2_p, 5),
            "平均情感指数": round(sent, 5),
            "状态": state,
            "主导阵营ID": c_id_found if "共识" in state else "",
            "主导阵营核心议题": dominant_topics_found if "共识" in state else "",
            "主导阵营占比": round(consensus_prop, 5) if "共识" in state else ""
        })

    consensus_df = pd.DataFrame(results)
    consensus_df.to_csv(os.path.join(output_dir, "consensus_points.csv"), index=False, encoding="utf-8-sig",
                        float_format="%.5f")
    return consensus_df


# 交叉滞后与情感耦合
def cross_lagged_analysis(series_X: pd.Series, series_Y: pd.Series):
    df_lag = pd.DataFrame({'Y_t': series_Y, 'Y_t_1': series_Y.shift(1), 'X_t_1': series_X.shift(1)}).dropna()
    if len(df_lag) < 5: return np.nan, np.nan
    X = sm.add_constant(df_lag[['Y_t_1', 'X_t_1']])
    Y = df_lag['Y_t']
    try:
        model = sm.OLS(Y, X).fit()
        return model.params['X_t_1'], model.pvalues['X_t_1']
    except:
        return np.nan, np.nan


def analyze_causality_and_emotion(topic_counts: pd.DataFrame, sentiment_means: pd.DataFrame, output_dir: str,
                                  label: str):
    print(f"  [DEBUG {label}] 开始进行交叉滞后与情感耦合分析...")
    print(f"  [DEBUG {label}] 当前时间刻度聚合后的总周期数 (点数): {len(topic_counts)}")

    causality_results = []
    topics = topic_counts.columns

    # 抽取一对议题测试 df_lag 的实际长度
    if len(topics) >= 2:
        sample_X = topic_counts[topics[0]]
        sample_Y = topic_counts[topics[1]]
        sample_df_lag = pd.DataFrame({'Y_t': sample_Y, 'Y_t_1': sample_Y.shift(1), 'X_t_1': sample_X.shift(1)}).dropna()
        print(f"  [DEBUG {label}] 平移滞后截断后，CLPM模型实际可用样本点数: {len(sample_df_lag)}")

        if len(sample_df_lag) < 5:
            print(f"  [DEBUG {label}] 实际可用样本点数 < 5！不满足 OLS 回归的最低自由度要求，将返回空结果。")

    # 1. 交叉滞后 (CLPM) 计算
    for X_topic in topics:
        for Y_topic in topics:
            if X_topic == Y_topic: continue
            coef, pval = cross_lagged_analysis(topic_counts[X_topic], topic_counts[Y_topic])
            if pd.notna(pval) and pval < 0.1:
                causality_results.append({
                    "Cause (X)": X_topic, "Effect (Y)": Y_topic, "Coefficient (β2)": round(coef, 5),
                    "P-value": round(pval, 5)
                })

    df_causality = pd.DataFrame(causality_results)
    print(f"  [DEBUG {label}] 找到满足 p < 0.1 的显著因果关系数量: {len(df_causality)}")

    if not df_causality.empty:
        out_path = os.path.join(output_dir, "topic_causality_CLPM.csv")
        df_causality.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"  [DEBUG {label}] 已输出: {out_path}")
    else:
        print(f"  [DEBUG {label}] 未找到显著的因果关系对，不生成 topic_causality_CLPM.csv 文件。")

    # 2. 情感 Granger 因果计算
    emotion_granger_results = []
    for X_topic in topics:
        for Y_topic in topics:
            if X_topic == Y_topic: continue
            data = pd.concat([sentiment_means[Y_topic], sentiment_means[X_topic]], axis=1).dropna()
            if data.var().min() < 1e-5 or len(data) < 5: continue
            try:
                test_result = grangercausalitytests(data, maxlag=1, verbose=False)
                p_value = test_result[1][0]['ssr_ftest'][1]
                if p_value < 0.1:
                    emotion_granger_results.append(
                        {"Source Emotion (X)": X_topic, "Target Emotion (Y)": Y_topic, "P-value": round(p_value, 5)})
            except:
                pass

    df_emotion = pd.DataFrame(emotion_granger_results)
    if not df_emotion.empty:
        df_emotion.to_csv(os.path.join(output_dir, "emotion_coupling_granger.csv"), index=False, encoding='utf-8-sig')


# 执行单刻度流水线
def run_analysis_for_frequency(df: pd.DataFrame, label: str, config: dict):
    print(f"\n{label}维度分析")
    output_dir = os.path.join(BASE_OUTPUT_DIR, label)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    topic_counts, sentiment_means, sentiment_idx, text_counts = aggregate_time_series(df, config["freq"])

    if topic_counts.empty or len(topic_counts) < 2:
        print(f"【{label}】维度数据量不足以进行分析。")
        return

    # 1. 聚类分析
    _, topic_clusters, clusters_dict = analyze_correlation_and_cluster(topic_counts, output_dir, n_clusters=2)
    export_proportions_and_camps(topic_counts, clusters_dict, output_dir)
    # 2. 状态判定
    consensus_df = extract_consensus_points(topic_counts, sentiment_idx, clusters_dict, text_counts, output_dir, config)
    # 3. 绘图
    plot_clustered_streamgraph(topic_counts, sentiment_idx, topic_clusters, clusters_dict, consensus_df, output_dir,config)
    # 4. 因果耦合
    analyze_causality_and_emotion(topic_counts, sentiment_means, output_dir, label)

    print(f"【{label}】维度分析完成，文件已输出至：{output_dir}")


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"找不到输入文件: {INPUT_CSV}")
        return

    df = load_and_preprocess_data(INPUT_CSV)
    df_ex = df.explode('themes_list').reset_index(drop=True)

    total_counts = df_ex['themes_list'].value_counts()
    total_proportions = df_ex['themes_list'].value_counts(normalize=True)
    print("\n各议题全局总占比统计")
    summary_df = pd.DataFrame({'出现次数': total_counts, '总占比': total_proportions.map(lambda x: f"{x:.2%}")})
    print(summary_df)

    for label, config in FREQ_CONFIG.items():
        run_analysis_for_frequency(df, label, config)


if __name__ == "__main__":
    main()