# evolutionary_game_data_generator.py
# 该脚本独立运行，用于生成跨滞后分析中与“演化博弈”相关的扩展数据表。
# 这些表格在原始主程序中被移除，现作为独立功能提供。

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import re
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

# ================= 字体设置（仅用于绘图，此处可保留） =================
font_path = "/root/SIMSUN.TTC"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ================= 配置参数 =================
INPUT_EXCEL = "/root/autodl-tmp/延迟退休/延迟退休output.xlsx"
OUTPUT_DIR = "/root/autodl-tmp/延迟退休/cross_lagged_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PUBLIC_THEMES = [
    "政策执行细则", "政策合理性的公众表态", "养老权益保障", "劳动者体力精力负荷", "就业竞争压力", "社会代际公平",
]
MEDIA_THEMES_MAPPING = {
    1: "改革必要性阐述",
    2: "改革制度设计与执行",
    3: "就业市场影响评估"
}

# 策略映射（与主程序完全一致）
gov_theme_to_strats = {
    "改革必要性阐述": ["渐进式", "推进", "法定", "退休年龄", "应对", "人口老龄化", "调整", "养老金", "人均", "预期",
                       "寿命", "延长", "延迟", "自愿", "弹性", "原则", "改革", "劳动者", "就业", "老龄", "结构", "变化",
                       "可持续性", "压力", "设计", "支付"],
    "改革制度设计与执行": ["养老金", "最低", "缴费", "年限", "法定", "退休年龄", "延迟", "选择", "弹性", "提前退休",
                           "自愿", "养老保险", "渐进式", "月龄", "递增", "方案", "差异化", "职工", "参与率", "查询",
                           "渠道", "窗口", "计算器", "官方", "十五年", "调整", "策略", "个体差异"],
    "就业市场影响评估": ["劳动者", "就业", "参与率", "挤占岗位", "代际竞争", "青年就业空间", "年龄歧视", "35岁门槛",
                         "大龄求职", "适老岗位", "灵活就业", "再就业", "超龄劳动者权益", "工伤保障", "结构性失业"]
}
med_theme_to_strats = {
    "改革必要性阐述": ["渐进式", "推进", "法定", "退休年龄", "应对", "人口老龄化", "调整", "养老金", "人均", "预期",
                       "寿命", "延长", "延迟", "自愿", "弹性", "原则", "改革", "劳动者", "就业", "老龄", "结构", "变化",
                       "可持续性", "压力", "设计", "支付"],
    "改革制度设计与执行": ["养老金", "最低", "缴费", "年限", "法定", "退休年龄", "延迟", "选择", "弹性", "提前退休",
                           "自愿", "养老保险", "渐进式", "月龄", "递增", "方案", "差异化", "职工", "参与率", "查询",
                           "渠道", "窗口", "计算器", "官方", "十五年", "调整", "策略", "个体差异"],
    "就业市场影响评估": ["劳动者", "就业", "参与率", "挤占岗位", "代际竞争", "青年就业空间", "年龄歧视", "35岁门槛",
                         "大龄求职", "适老岗位", "灵活就业", "再就业", "超龄劳动者权益", "工伤保障", "结构性失业"]
}
pub_theme_to_strats = {
    "政策执行细则": ["法定", "退休年龄", "查询", "渠道", "计算器", "官方", "政策", "执行", "渐进式", "改革", "小步",
                     "调整", "路径", "支持", "策略", "方案", "推进", "节奏", "月龄", "递增"],
    "政策合理性的公众表态": ["自愿", "弹性", "原则", "支持", "延迟", "退休", "政策", "意愿", "反对", "强制", "提前退休",
                             "统一", "渐进式", "改革", "延长", "工作", "年限"],
    "养老权益保障": ["养老金", "税收", "个人所得税", "制度", "自负盈亏", "退休费", "缴费", "年限", "拉长", "最低",
                     "工龄", "社保"],
    "劳动者体力精力负荷": ["体力", "劳动者", "脑力劳动者", "健康透支", "职业病", "慢性病", "身体机能", "重体力", "高危",
                           "一线工人", "职业倦怠", "精神压力", "丧失劳动能力", "老年生活质量"],
    "就业竞争压力": ["反对", "延迟", "退休", "退休年龄", "市场", "歧视", "政策", "高龄", "就业", "困难", "劳动者",
                     "面临", "青年", "压力", "优先", "年轻人", "强制", "延长", "工作"],
    "社会代际公平": ["反对", "延迟", "退休", "政策", "退休年龄", "延长", "工作", "年限", "强制", "公平性", "无奈",
                     "生育率", "负面影响", "生育", "意愿", "降低", "年轻人", "增加", "加剧", "收入"]
}

WINDOW_DAYS = 7
STEP_DAYS = 2

# ================= 被删除的辅助函数 =================
def get_sentiment_weight(ts_dict, theme, date_center, strategy, party_type):
    """根据策略类型和情感得分计算权重"""
    try:
        ts_df = ts_dict[theme]
        nearest_idx = abs(ts_df.index - date_center).argmin()
        sent = ts_df.iloc[nearest_idx].get('平均情感得分', 0.0)
        if pd.isna(sent):
            sent = 0.0

        polarity = 0.0
        if party_type == 'Gov':
            if strategy in ['安全稳定', '警示', '切割责任方']:
                polarity = -1.0
            elif strategy in ['发展效率', '安抚', '承诺', '抢先发布']:
                polarity = 1.0
        elif party_type == 'Med':
            if strategy in ['监督追责', '煽情', '悲情', '保持独立']:
                polarity = -1.0
            elif strategy in ['反思分析', '平衡多元', '理性']:
                polarity = 1.0
        elif party_type == 'Pub':
            if strategy in ['愤怒', '悲情', '集体行动', '曝光爆料']:
                polarity = -1.0
            elif strategy in ['幽默', '网络联动', '精英结盟']:
                polarity = 1.0
        return 1.0 + polarity * float(sent) * 1.5
    except Exception:
        return 1.0

def append_significant_window_rows(target_rows, rolling_df, source_name, metric, src_theme, pub_theme):
    """将显著期记录添加到全局列表（原函数）"""
    direction_specs = [
        ('X引导Y_显著', 'X引导Y_P值', f'{source_name}引导公众'),
        ('Y引导X_显著', 'Y引导X_P值', f'公众倒逼{source_name}')
    ]
    for flag_col, p_col, direction in direction_specs:
        for start, end, mean_p, min_p, max_p in extract_significant_periods_dates(
                rolling_df, flag_col, p_col):
            target_rows.append({
                '分析指标': metric,
                '分析视角': source_name,
                '政府/媒体议题_X': src_theme,
                '公众议题_Y': pub_theme,
                '因果方向': direction,
                'VAR显著期开始': start,
                'VAR显著期结束': end,
                'VAR显著期平均P值': mean_p,
                'VAR显著期最小P值': min_p,
                'VAR显著期最大P值': max_p
            })

def build_topic_consistent_overlap_table(significant_rows, consensus_df, consensus_topic_df):
    """时间交叠与议题一致性核验（原函数）"""
    output_rows = []
    if not significant_rows or consensus_df.empty:
        return pd.DataFrame()

    consensus = consensus_df.copy()
    consensus['窗口开始时间_dt'] = pd.to_datetime(consensus['窗口开始时间'])
    consensus['窗口结束时间_dt'] = pd.to_datetime(consensus['窗口结束时间'])

    for sig in significant_rows:
        var_start = pd.Timestamp(sig['VAR显著期开始'])
        var_end = pd.Timestamp(sig['VAR显著期结束'])
        overlaps = consensus[
            (consensus['窗口开始时间_dt'] <= var_end) &
            (consensus['窗口结束时间_dt'] > var_start)
        ]

        for _, consensus_row in overlaps.iterrows():
            window_start = consensus_row['窗口开始时间']
            window_end = consensus_row['窗口结束时间']
            topic_match = consensus_topic_df[
                (consensus_topic_df['窗口开始时间'] == window_start) &
                (consensus_topic_df['窗口结束时间'] == window_end) &
                (consensus_topic_df['公众议题'] == sig['公众议题_Y'])
            ]
            matched = not topic_match.empty
            matched_topic = topic_match.iloc[0] if matched else None

            output_rows.append({
                **sig,
                '共识窗口开始': window_start,
                '共识窗口结束': window_end,
                '共识窗口全部公众议题排序': consensus_row['共识窗口议题排序'],
                '共识窗口议题种类数': consensus_row['公众议题种类数'],
                '时间交叠': '是',
                '公众议题_Y在共识窗口中出现': '是' if matched else '否',
                '议题一致且时间交叠': '是' if matched else '否',
                '公众议题_Y排名': int(matched_topic['议题排名']) if matched else None,
                '公众议题_Y标注数': int(matched_topic['议题标注数']) if matched else 0,
                '公众议题_Y占比': matched_topic['议题占比'] if matched else '0.00%'
            })

    return pd.DataFrame(output_rows)

# ================= 通用数据处理函数（与主程序一致） =================
def process_all_data(filepath):
    print(f"\n[1/4] 正在加载和预处理全局数据... 文件路径: {filepath}")
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"[!] 读取Excel失败: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"  -> 初始数据量: {len(df)} 条")
    if "蒸馏总结" in df.columns:
        df = df.dropna(subset=["蒸馏总结"])
        invalid_patterns = "未提及|未涉及|无法提炼|无法概括|无法得出|没有涉及|不涉及"
        df = df[~df["蒸馏总结"].astype(str).str.contains(invalid_patterns, regex=True, na=False)]

    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期'])

    if '情感得分' in df.columns:
        df['sentiment_score'] = pd.to_numeric(df['情感得分'], errors='coerce').fillna(0)
    else:
        df['sentiment_score'] = 0.0
        print("  [!] 警告: 数据中未发现 '情感得分' 列，已默认置 0。")

    ALL_ALLOWED_THEMES = PUBLIC_THEMES + list(MEDIA_THEMES_MAPPING.values())
    def extract_valid_themes(theme_str: str) -> list[str]:
        if pd.isna(theme_str): return []
        cleaned = re.sub(r'\([0-9.]+\)', '', str(theme_str))
        extracted = [t.strip() for t in cleaned.split('|') if t.strip()]
        return [t for t in extracted if t in ALL_ALLOWED_THEMES]

    if "匹配的主题标签" in df.columns:
        df["themes_list"] = df["匹配的主题标签"].apply(extract_valid_themes)
        df = df.explode('themes_list')
        df = df.rename(columns={'themes_list': '最终主题'})
        df = df.dropna(subset=['最终主题'])
    else:
        print("  [!] 警告: 数据中未发现 '匹配的主题标签' 列！")
        df['最终主题'] = None

    if '账号分类' in df.columns:
        df['账号分类_lower'] = df['账号分类'].astype(str).str.lower().str.strip()
        df_public = df[df['账号分类_lower'].isin(['public', '公众', '普通用户', '群众'])].copy()
        df_media = df[df['账号分类_lower'].isin(['media', '媒体', '新闻媒体', '自媒体'])].copy()
        df_gov = df[df['账号分类_lower'].isin(['government', 'gov', '政府', '政务', '政务机构'])].copy()
    else:
        print("  [!] 警告: 数据中未发现 '账号分类' 列！")
        df_public, df_media, df_gov = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"  -> 公众有效数据量: {len(df_public)} 条")
    print(f"  -> 媒体有效数据量: {len(df_media)} 条")
    print(f"  -> 政府有效数据量: {len(df_gov)} 条")
    return df_public, df_media, df_gov

def get_global_time_bounds(dfs: list) -> tuple:
    valid_dates = []
    for d in dfs:
        if not d.empty and '日期' in d.columns:
            valid_dates.append(d['日期'].dropna())
    if not valid_dates:
        return pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01')
    all_dates = pd.concat(valid_dates)
    return all_dates.min().floor('D'), all_dates.max().ceil('D')

def aggregate_timeseries_sliding(df, theme_col, global_start, global_end, window_days=14, step_days=2,
                                 expected_themes=None):
    timeseries_dict = {}
    themes = expected_themes if expected_themes else (df[theme_col].unique() if not df.empty else [])
    for theme in themes:
        records = []
        current_date = global_start
        if not df.empty and theme_col in df.columns and theme in df[theme_col].values:
            theme_df = df[df[theme_col] == theme]
        else:
            theme_df = pd.DataFrame()

        while current_date + pd.Timedelta(days=window_days) <= global_end + pd.Timedelta(days=step_days):
            window_end = current_date + pd.Timedelta(days=window_days)
            center_date = current_date + pd.Timedelta(days=window_days / 2)

            if not theme_df.empty:
                mask = (theme_df['日期'] >= current_date) & (theme_df['日期'] < window_end)
                window_df = theme_df.loc[mask]
                count = len(window_df)
                avg_score = window_df['sentiment_score'].mean() if count > 0 else 0.0
                pos_df = window_df[window_df['sentiment_score'] > 0]
                neg_df = window_df[window_df['sentiment_score'] < 0]
                pos_count = len(pos_df)
                neg_count = len(neg_df)
                pos_score = pos_df['sentiment_score'].mean() if pos_count > 0 else 0.0
                neg_score = neg_df['sentiment_score'].mean() if neg_count > 0 else 0.0
            else:
                count = 0
                avg_score, pos_score, neg_score = 0.0, 0.0, 0.0

            records.append({
                '日期': center_date,
                '议题': count,
                '平均情感得分': avg_score,
                '正面情感': pos_score,
                '负面情感': neg_score
            })
            current_date += pd.Timedelta(days=step_days)

        if records:
            ts_df = pd.DataFrame(records).set_index('日期')
        else:
            ts_df = pd.DataFrame(columns=['议题', '平均情感得分', '正面情感', '负面情感'])
        timeseries_dict[theme] = ts_df
    return timeseries_dict

def granger_causality_p_value(df_pair, cause_col, effect_col, lag):
    ordered_pair = df_pair[[effect_col, cause_col]]
    result = grangercausalitytests(ordered_pair, maxlag=[lag], verbose=False)
    return result[lag][0]['ssr_ftest'][1]

def rolling_granger_causality(df_pair, window_size=15, maxlag=1, alpha=0.05):
    rolling_results = []
    df_pair = df_pair.sort_index()
    for i in range(window_size, len(df_pair)):
        df_window = df_pair.iloc[i - window_size: i]
        start_date = df_window.index[0]
        end_date = df_window.index[-1]
        if df_window.iloc[:, 0].nunique() <= 1 or df_window.iloc[:, 1].nunique() <= 1:
            continue
        try:
            x_col, y_col = df_window.columns[:2]
            p_xy = granger_causality_p_value(df_window, cause_col=x_col, effect_col=y_col, lag=maxlag)
            p_yx = granger_causality_p_value(df_window, cause_col=y_col, effect_col=x_col, lag=maxlag)
            irf_x2y, irf_y2x = 0.0, 0.0
            try:
                loc_model = VAR(df_window)
                loc_res = loc_model.fit(maxlags=1)
                loc_irf = loc_res.irf(2)
                irf_x2y = loc_irf.orth_irfs[1, 1, 0]
                irf_y2x = loc_irf.orth_irfs[1, 0, 1]
            except:
                pass
            rolling_results.append({
                '窗口期起': start_date,
                '窗口期止': end_date,
                'X引导Y_P值': p_xy,
                'Y引导X_P值': p_yx,
                'X引导Y_显著': p_xy < alpha,
                'Y引导X_显著': p_yx < alpha,
                'X引导Y_IRF': irf_x2y,
                'Y引导X_IRF': irf_y2x
            })
        except Exception:
            continue
    return pd.DataFrame(rolling_results)

def extract_significant_periods_dates(rolling_df, col_flag, col_p):
    if rolling_df.empty:
        return []
    periods = []
    in_period = False
    start_time = None
    p_values = []
    for i in range(len(rolling_df)):
        row = rolling_df.iloc[i]
        if row[col_flag]:
            if not in_period:
                in_period = True
                start_time = row['窗口期起']
                p_values = [row[col_p]]
            else:
                p_values.append(row[col_p])
        else:
            if in_period:
                in_period = False
                end_time = rolling_df.iloc[i - 1]['窗口期止']
                mean_p = sum(p_values) / len(p_values)
                min_p = min(p_values)
                max_p = max(p_values)
                periods.append((start_time, end_time, mean_p, min_p, max_p))
                p_values = []
    if in_period:
        end_time = rolling_df.iloc[-1]['窗口期止']
        mean_p = sum(p_values) / len(p_values)
        min_p = min(p_values)
        max_p = max(p_values)
        periods.append((start_time, end_time, mean_p, min_p, max_p))
    return periods

def check_stationarity(ts_data):
    res = adfuller(ts_data.dropna())
    is_stationary = res[1] < 0.05
    if not is_stationary:
        ts_data = ts_data.diff().dropna()
    return ts_data, is_stationary

def build_var_model(df_pair, maxlags=4):
    model = VAR(df_pair)
    try:
        order_results = model.select_order(maxlags=maxlags)
        best_lag = order_results.aic
        if best_lag == 0: best_lag = 1
    except:
        best_lag = 1
    var_res = model.fit(best_lag)
    try:
        x_col, y_col = df_pair.columns[:2]
        p_val_xy = granger_causality_p_value(df_pair, cause_col=x_col, effect_col=y_col, lag=best_lag)
        p_val_yx = granger_causality_p_value(df_pair, cause_col=y_col, effect_col=x_col, lag=best_lag)
    except:
        p_val_xy, p_val_yx = 1.0, 1.0
    return var_res, best_lag, p_val_xy, p_val_yx

def process_single_var(src_ts, pub_ts, source_name, metric, src_theme, pub_theme):
    df_pair = pd.concat([src_ts, pub_ts], axis=1, keys=[f'{source_name}_X', 'Public_Y']).dropna()
    if len(df_pair) < 15 or df_pair.iloc[:, 0].nunique() <= 1 or df_pair.iloc[:, 1].nunique() <= 1:
        return None, pd.DataFrame()
    x_stat, _ = check_stationarity(df_pair[f'{source_name}_X'])
    y_stat, _ = check_stationarity(df_pair['Public_Y'])
    df_stat = pd.concat([x_stat, y_stat], axis=1).dropna()
    try:
        var_model, best_lag, p_src_to_pub, p_pub_to_src = build_var_model(df_stat, maxlags=2)
        rolling_df = rolling_granger_causality(df_stat, window_size=12, maxlag=best_lag)

        src_to_pub_strs, pub_to_src_strs = [], []
        for s, e, mean_p, min_p, max_p in extract_significant_periods_dates(rolling_df, 'X引导Y_显著', 'X引导Y_P值'):
            src_to_pub_strs.append(f"{s.strftime('%Y/%m/%d')}-{e.strftime('%Y/%m/%d')}(均值p={mean_p:.4f})")
        for s, e, mean_p, min_p, max_p in extract_significant_periods_dates(rolling_df, 'Y引导X_显著', 'Y引导X_P值'):
            pub_to_src_strs.append(f"{s.strftime('%Y/%m/%d')}-{e.strftime('%Y/%m/%d')}(均值p={mean_p:.4f})")

        fevd = var_model.fevd(10)
        src_contrib_to_public = fevd.decomp[1][10-1][0]
        pub_contrib_to_src = fevd.decomp[0][10-1][1]
        irf = var_model.irf(10)
        cum_irf_src_to_pub = irf.orth_cum_effects[10, 1, 0]
        cum_irf_pub_to_src = irf.orth_cum_effects[10, 0, 1]

        causality_conclusion = "无显著因果"
        if p_src_to_pub < 0.05 and p_pub_to_src < 0.05:
            causality_conclusion = "双向强化"
        elif p_src_to_pub < 0.05:
            causality_conclusion = f"{source_name}主导"
        elif p_pub_to_src < 0.05:
            causality_conclusion = "公众倒逼"

        res_dict = {
            "分析指标": metric,
            "分析视角": source_name,
            "政府/媒体议题_X": src_theme,
            "公众议题_Y": pub_theme,
            "全局因果关系": causality_conclusion,
            f"政府/媒体引导的具体时间段": " | ".join(src_to_pub_strs) if src_to_pub_strs else "无",
            f"公众倒逼的具体时间段": " | ".join(pub_to_src_strs) if pub_to_src_strs else "无",
            "全局VAR_最优滞后阶数": best_lag,
            f"全局P值(政府/媒体引导)": round(p_src_to_pub, 4) if p_src_to_pub else None,
            f"全局P值(公众倒逼)": round(p_pub_to_src, 4) if p_pub_to_src else None,
            f"第10期方差分解(政府/媒体贡献度)": f"{src_contrib_to_public:.2%}" if pd.notna(src_contrib_to_public) else "N/A",
            f"第10期方差分解(公众贡献度)": f"{pub_contrib_to_src:.2%}" if pd.notna(pub_contrib_to_src) else "N/A",
            f"第10期累积脉冲响应(政府/媒体冲击公众)": round(cum_irf_src_to_pub, 4) if pd.notna(cum_irf_src_to_pub) else "N/A",
            f"第10期累积脉冲响应(公众冲击政府/媒体)": round(cum_irf_pub_to_src, 4) if pd.notna(cum_irf_pub_to_src) else "N/A"
        }
        return res_dict, rolling_df
    except Exception:
        return None, pd.DataFrame()

def run_combined_var_pipeline(media_ts_dict, gov_ts_dict, public_ts_dict, metric='议题',
                              window_coef_rows=None, significant_window_rows=None):
    results = []
    total_pairs = len(PUBLIC_THEMES) * len(MEDIA_THEMES_MAPPING)
    current_pair = 1

    for pub_theme in PUBLIC_THEMES:
        for src_theme in MEDIA_THEMES_MAPPING.values():
            print(f"\n[{current_pair}/{total_pairs}] 联合因果分析 [{metric}]: 政府/媒体 [{src_theme}] <---> [公众] {pub_theme}")
            current_pair += 1

            if pub_theme not in public_ts_dict: continue
            pub_ts = public_ts_dict[pub_theme][metric]

            # 政府分析
            rolling_gov = pd.DataFrame()
            if src_theme in gov_ts_dict:
                gov_ts = gov_ts_dict[src_theme][metric]
                res_gov, rolling_gov = process_single_var(gov_ts, pub_ts, "Government", metric, src_theme, pub_theme)
                if res_gov: results.append(res_gov)
                if significant_window_rows is not None and not rolling_gov.empty:
                    append_significant_window_rows(significant_window_rows, rolling_gov, "Government", metric, src_theme, pub_theme)

                if window_coef_rows is not None and not rolling_gov.empty:
                    for _, row in rolling_gov.iterrows():
                        s_str, e_str = row['窗口期起'].strftime('%Y-%m-%d'), row['窗口期止'].strftime('%Y-%m-%d')
                        center_dt = row['窗口期起'] + (row['窗口期止'] - row['窗口期起']) / 2
                        v_s2p, v_p2s = row.get('X引导Y_IRF', 0.0), row.get('Y引导X_IRF', 0.0)
                        gov_strats = gov_theme_to_strats.get(src_theme, [])
                        pub_strats = pub_theme_to_strats.get(pub_theme, [])
                        for g_s in gov_strats:
                            g_w = get_sentiment_weight(gov_ts_dict, src_theme, center_dt, g_s, 'Gov')
                            for p_s in pub_strats:
                                p_w = get_sentiment_weight(public_ts_dict, pub_theme, center_dt, p_s, 'Pub')
                                if pd.notna(v_s2p) and v_s2p != 0:
                                    window_coef_rows.append({
                                        'window_start': s_str, 'window_end': e_str, 'metric': metric,
                                        'cause_party': 'Gov', 'cause_strategy': g_s, 'effect_party': 'Pub',
                                        'effect_strategy': p_s, 'coef': v_s2p * g_w * p_w
                                    })
                                if pd.notna(v_p2s) and v_p2s != 0:
                                    window_coef_rows.append({
                                        'window_start': s_str, 'window_end': e_str, 'metric': metric,
                                        'cause_party': 'Pub', 'cause_strategy': p_s, 'effect_party': 'Gov',
                                        'effect_strategy': g_s, 'coef': v_p2s * p_w * g_w
                                    })

            # 媒体分析
            rolling_media = pd.DataFrame()
            if src_theme in media_ts_dict:
                media_ts = media_ts_dict[src_theme][metric]
                res_media, rolling_media = process_single_var(media_ts, pub_ts, "Media", metric, src_theme, pub_theme)
                if res_media: results.append(res_media)
                if significant_window_rows is not None and not rolling_media.empty:
                    append_significant_window_rows(significant_window_rows, rolling_media, "Media", metric, src_theme, pub_theme)

                if window_coef_rows is not None and not rolling_media.empty:
                    for _, row in rolling_media.iterrows():
                        s_str, e_str = row['窗口期起'].strftime('%Y-%m-%d'), row['窗口期止'].strftime('%Y-%m-%d')
                        center_dt = row['窗口期起'] + (row['窗口期止'] - row['窗口期起']) / 2
                        v_s2p, v_p2s = row.get('X引导Y_IRF', 0.0), row.get('Y引导X_IRF', 0.0)
                        med_strats = med_theme_to_strats.get(src_theme, [])
                        pub_strats = pub_theme_to_strats.get(pub_theme, [])
                        for m_s in med_strats:
                            m_w = get_sentiment_weight(media_ts_dict, src_theme, center_dt, m_s, 'Med')
                            for p_s in pub_strats:
                                p_w = get_sentiment_weight(public_ts_dict, pub_theme, center_dt, p_s, 'Pub')
                                if pd.notna(v_s2p) and v_s2p != 0:
                                    window_coef_rows.append({
                                        'window_start': s_str, 'window_end': e_str, 'metric': metric,
                                        'cause_party': 'Med', 'cause_strategy': m_s, 'effect_party': 'Pub',
                                        'effect_strategy': p_s, 'coef': v_s2p * m_w * p_w
                                    })
                                if pd.notna(v_p2s) and v_p2s != 0:
                                    window_coef_rows.append({
                                        'window_start': s_str, 'window_end': e_str, 'metric': metric,
                                        'cause_party': 'Pub', 'cause_strategy': p_s, 'effect_party': 'Med',
                                        'effect_strategy': m_s, 'coef': v_p2s * p_w * m_w
                                    })

    return results

# ================= 主程序 =================
def main():
    print("\n========== 演化博弈扩展数据生成器 ==========")
    print("本程序将重新执行数据预处理、聚合和VAR分析，生成以下额外表格：")
    print("  - cross_lagged_coefficients.csv")
    print("  - cross_lagged_window_coefficients.csv")
    print("  - 博弈策略频数演化汇总数据.csv")
    print("  - 共识窗口_全部公众议题明细.csv")
    print("  - VAR显著期_共识窗口议题核验明细.csv")
    print("  - VAR显著期_议题一致共识窗口.csv")

    # 第一阶段：数据加载
    df_public, df_media, df_gov = process_all_data(INPUT_EXCEL)

    # 第二阶段：时间序列聚合
    global_start, global_end = get_global_time_bounds([df_media, df_gov, df_public])
    media_ts_dict = aggregate_timeseries_sliding(df_media, '最终主题', global_start, global_end,
                                                 WINDOW_DAYS, STEP_DAYS,
                                                 expected_themes=list(MEDIA_THEMES_MAPPING.values()))
    gov_ts_dict = aggregate_timeseries_sliding(df_gov, '最终主题', global_start, global_end,
                                               WINDOW_DAYS, STEP_DAYS,
                                               expected_themes=list(MEDIA_THEMES_MAPPING.values()))
    public_ts_dict = aggregate_timeseries_sliding(df_public, '最终主题', global_start, global_end,
                                                  WINDOW_DAYS, STEP_DAYS,
                                                  expected_themes=PUBLIC_THEMES)

    # 对数变换
    for ts_dict in [media_ts_dict, gov_ts_dict, public_ts_dict]:
        for theme, df in ts_dict.items():
            if not df.empty and '议题' in df.columns:
                df['议题'] = np.log1p(df['议题'].astype(float))

    # 准备收集列表
    window_coef_rows = []
    significant_window_rows = []

    # 第三阶段：运行VAR（收集所有指标）
    metrics_to_analyze = ['议题', '平均情感得分', '正面情感', '负面情感']
    for metric in metrics_to_analyze:
        print(f"\n========== 运行VAR分析 - {metric} ==========")
        _ = run_combined_var_pipeline(media_ts_dict, gov_ts_dict, public_ts_dict,
                                      metric=metric,
                                      window_coef_rows=window_coef_rows,
                                      significant_window_rows=significant_window_rows)

    # 第四阶段：生成额外数据表格
    print("\n========== 生成演化博弈相关表格 ==========")

    # 1. 全局静态系数表
    STRATEGIES = {
        'Gov': ['安全稳定', '发展效率', '公平正义', '安抚', '警示', '承诺', '联合权威', '切割责任方', '媒体协同',
                '及时公开', '分批披露', '抢先发布'],
        'Med': ['监督追责', '反思分析', '平衡多元', '煽情', '理性', '悲情', '倾向官方', '保持独立', '倾向公众',
                '深度调查', '滚动快讯', '观点转载'],
        'Pub': ['权益维护', '公共利益', '生存生计', '愤怒', '悲情', '幽默', '集体行动', '网络联动', '精英结盟',
                '曝光爆料', '证据收集', '话题刷屏']
    }

    print("\n[*] 重新运行VAR（仅议题和情感）以提取全局因果系数...")
    all_results_global = []
    for metric in ['议题', '平均情感得分']:
        res = run_combined_var_pipeline(media_ts_dict, gov_ts_dict, public_ts_dict,
                                        metric=metric,
                                        window_coef_rows=None,
                                        significant_window_rows=None)
        all_results_global.extend(res)

    eg_coef_rows = []
    for res in all_results_global:
        src_party = "Gov" if res.get('分析视角') == "Government" else "Med"
        src_theme = res.get('政府/媒体议题_X')
        pub_theme = res.get('公众议题_Y')
        coef_src_to_pub = res.get("第10期累积脉冲响应(政府/媒体冲击公众)")
        coef_pub_to_src = res.get("第10期累积脉冲响应(公众冲击政府/媒体)")

        src_strats = gov_theme_to_strats.get(src_theme, []) if src_party == "Gov" else med_theme_to_strats.get(src_theme, [])
        pub_strats = pub_theme_to_strats.get(pub_theme, [])

        if coef_src_to_pub != "N/A" and coef_src_to_pub is not None:
            try:
                val = float(coef_src_to_pub)
                for s_s in src_strats:
                    for p_s in pub_strats:
                        eg_coef_rows.append({
                            'cause_party': src_party, 'cause_strategy': s_s,
                            'effect_party': 'Pub', 'effect_strategy': p_s, 'coef': val
                        })
            except: pass
        if coef_pub_to_src != "N/A" and coef_pub_to_src is not None:
            try:
                val = float(coef_pub_to_src)
                for p_s in pub_strats:
                    for s_s in src_strats:
                        eg_coef_rows.append({
                            'cause_party': 'Pub', 'cause_strategy': p_s,
                            'effect_party': src_party, 'effect_strategy': s_s, 'coef': val
                        })
            except: pass

    if eg_coef_rows:
        df_eg_coef = pd.DataFrame(eg_coef_rows)
        df_eg_coef = df_eg_coef.groupby(['cause_party', 'cause_strategy', 'effect_party', 'effect_strategy'],
                                        as_index=False).mean()
    else:
        df_eg_coef = pd.DataFrame(columns=['cause_party', 'cause_strategy', 'effect_party', 'effect_strategy', 'coef'])
    coef_out_path = os.path.join(OUTPUT_DIR, "cross_lagged_coefficients.csv")
    df_eg_coef.to_csv(coef_out_path, index=False, encoding='utf-8-sig')
    print(f"  -> 静态全局系数表已导出: {coef_out_path}")

    # 2. 窗口动态系数表
    if window_coef_rows:
        df_win_coef = pd.DataFrame(window_coef_rows)
        df_win_coef = df_win_coef[df_win_coef['metric'].isin(['议题', '平均情感得分'])]
        df_win_coef = df_win_coef.groupby(
            ['window_start', 'window_end', 'cause_party', 'cause_strategy', 'effect_party', 'effect_strategy'],
            as_index=False)['coef'].mean()
        win_out_path = os.path.join(OUTPUT_DIR, "cross_lagged_window_coefficients.csv")
        df_win_coef.to_csv(win_out_path, index=False, encoding='utf-8-sig')
        print(f"  -> 窗口动态系数表已导出: {win_out_path}")

    # 3. 共识窗口数据（博弈策略频数演化汇总数据.csv 和 全部公众议题明细）
    windows_eg = []
    consensus_topic_rows = []
    current_date = global_start
    while current_date + pd.Timedelta(days=WINDOW_DAYS) <= global_end + pd.Timedelta(days=STEP_DAYS):
        window_end = current_date + pd.Timedelta(days=WINDOW_DAYS)

        mask_m = (df_media['日期'] >= current_date) & (df_media['日期'] < window_end)
        mask_g = (df_gov['日期'] >= current_date) & (df_gov['日期'] < window_end)
        mask_p = (df_public['日期'] >= current_date) & (df_public['日期'] < window_end)

        w_m = df_media[mask_m]
        w_g = df_gov[mask_g]
        w_p = df_public[mask_p]

        counts = {}
        for party in STRATEGIES:
            for s in STRATEGIES[party]:
                counts[s] = 1.0

        for _, row in w_g.iterrows():
            theme = row.get('最终主题')
            sent = row.get('sentiment_score', 0)
            for s in gov_theme_to_strats.get(theme, []):
                if s in ['安全稳定', '警示', '切割责任方']:
                    counts[s] += 2.0 * (1.0 - sent)
                elif s in ['发展效率', '安抚', '承诺', '抢先发布']:
                    counts[s] += 2.0 * (1.0 + sent)
                else:
                    counts[s] += 2.0
            if sent > 0: counts['安抚'] += 1.5
            if sent < 0: counts['警示'] += 1.5

        for _, row in w_m.iterrows():
            theme = row.get('最终主题')
            sent = row.get('sentiment_score', 0)
            for s in med_theme_to_strats.get(theme, []):
                if s in ['监督追责', '煽情', '悲情', '保持独立', '倾向公众']:
                    counts[s] += 2.0 * (1.0 - sent)
                elif s in ['反思分析', '平衡多元', '理性', '倾向官方', '观点转载']:
                    counts[s] += 2.0 * (1.0 + sent)
                else:
                    counts[s] += 2.0
            if sent < 0: counts['监督追责'] += 1.5

        for _, row in w_p.iterrows():
            theme = row.get('最终主题')
            sent = row.get('sentiment_score', 0)
            for s in pub_theme_to_strats.get(theme, []):
                if s in ['愤怒', '悲情', '集体行动', '曝光爆料', '话题刷屏']:
                    counts[s] += 2.0 * (1.0 - sent)
                elif s in ['幽默', '网络联动', '精英结盟', '证据收集']:
                    counts[s] += 2.0 * (1.0 + sent)
                else:
                    counts[s] += 2.0
            if sent < 0: counts['愤怒'] += 1.5

        if not w_p.empty and '最终主题' in w_p.columns:
            public_topic_counts = w_p['最终主题'].dropna().value_counts()
        else:
            public_topic_counts = pd.Series(dtype='int64')

        topic_assignment_total = int(public_topic_counts.sum())
        dominant_topic = public_topic_counts.index[0] if not public_topic_counts.empty else "综合政策议题"
        dominant_topic_pct = (public_topic_counts.iloc[0] / topic_assignment_total if topic_assignment_total > 0 else 0.0)
        topic_ranking = "；".join(
            f"{rank}.{theme}({int(count)}, {count / topic_assignment_total:.2%})"
            for rank, (theme, count) in enumerate(public_topic_counts.items(), start=1)
        ) or "无公众议题"
        topic_set = " | ".join(public_topic_counts.index.astype(str)) or "无"

        window_start_str = current_date.strftime('%Y-%m-%d')
        window_end_str = window_end.strftime('%Y-%m-%d')
        for rank, (theme, count) in enumerate(public_topic_counts.items(), start=1):
            consensus_topic_rows.append({
                '窗口开始时间': window_start_str,
                '窗口结束时间': window_end_str,
                '公众议题': theme,
                '议题排名': rank,
                '议题标注数': int(count),
                '议题占比值': count / topic_assignment_total,
                '议题占比': f"{count / topic_assignment_total:.2%}"
            })

        row_dict = {
            '窗口开始时间': window_start_str,
            '窗口结束时间': window_end_str,
            '主导话题': dominant_topic,
            '主导话题占比': f"{dominant_topic_pct:.2%}",
            '公众议题种类数': len(public_topic_counts),
            '公众议题标注总数': topic_assignment_total,
            '公众议题集合': topic_set,
            '共识窗口议题排序': topic_ranking
        }
        for theme in PUBLIC_THEMES:
            theme_count = int(public_topic_counts.get(theme, 0))
            row_dict[f'公众议题计数_{theme}'] = theme_count
            row_dict[f'公众议题占比_{theme}'] = (f"{theme_count / topic_assignment_total:.2%}" if topic_assignment_total > 0 else "0.00%")
        for s in counts: row_dict[s] = counts[s]
        windows_eg.append(row_dict)
        current_date += pd.Timedelta(days=STEP_DAYS)

    # 导出共识窗口
    df_eg_consensus = pd.DataFrame(windows_eg)
    advanced_dir = "/root/autodl-tmp/延迟退休/advanced_analysis/"
    os.makedirs(advanced_dir, exist_ok=True)
    consensus_out_path = os.path.join(advanced_dir, "博弈策略频数演化汇总数据.csv")
    df_eg_consensus.to_csv(consensus_out_path, index=False, encoding='utf-8-sig')
    print(f"  -> 共识窗口策略频数数据已导出: {consensus_out_path}")

    # 导出共识窗口全部公众议题明细
    consensus_topic_columns = [
        '窗口开始时间', '窗口结束时间', '公众议题', '议题排名',
        '议题标注数', '议题占比值', '议题占比'
    ]
    df_consensus_topics = pd.DataFrame(consensus_topic_rows, columns=consensus_topic_columns)
    consensus_topic_path = os.path.join(advanced_dir, "共识窗口_全部公众议题明细.csv")
    df_consensus_topics.to_csv(consensus_topic_path, index=False, encoding='utf-8-sig')
    print(f"  -> 共识窗口公众议题明细已导出: {consensus_topic_path}")

    # 4. VAR显著期与共识窗口议题核验表
    df_overlap_check = build_topic_consistent_overlap_table(
        significant_window_rows, df_eg_consensus, df_consensus_topics)
    overlap_check_path = os.path.join(advanced_dir, "VAR显著期_共识窗口议题核验明细.csv")
    df_overlap_check.to_csv(overlap_check_path, index=False, encoding='utf-8-sig')
    print(f"  -> VAR显著期与共识窗口核验表已导出: {overlap_check_path}")

    if not df_overlap_check.empty:
        df_topic_consistent = df_overlap_check[df_overlap_check['议题一致且时间交叠'] == '是'].copy()
    else:
        df_topic_consistent = df_overlap_check.copy()
    topic_consistent_path = os.path.join(advanced_dir, "VAR显著期_议题一致共识窗口.csv")
    df_topic_consistent.to_csv(topic_consistent_path, index=False, encoding='utf-8-sig')
    print(f"  -> 议题一致共识窗口已导出: {topic_consistent_path}")

    print("\n========== 所有演化博弈相关数据已生成完毕 ==========")

if __name__ == "__main__":
    main()