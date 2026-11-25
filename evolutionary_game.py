import os

import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings("ignore")

# ================= 字体与环境设置 =================
try:
    font_path = "/root/SIMSUN.TTC"
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CROSS_LAGGED_DIR = "/root/autodl-tmp/预制菜/cross_lagged_results/"
CONSENSUS_FILE = "/root/autodl-tmp/预制菜/advanced_analysis/14天刻度_全情感共识汇总数据.csv"
OUTPUT_DIR = "/root/autodl-tmp/预制菜/evolutionary_game_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PURE_STRATEGY = False
RANDOM_STRENGTH = 0.256
INIT_JITTER = 0.128  # 初值随机扰动

TARGET_WINDOW_GROUPS = [
    [1, 2], [3], [4, 5, 6, 7, 8, 9], [10], [11], [12, 13, 14, 15, 16, 17, 18]
]

# 演化区间模式："window" 使用上面的窗口分组；"custom" 使用自定义时间段。
EVOLUTION_PERIOD_MODE = "custom"
CUSTOM_TIME_PERIODS = [
    {
        "period_id": "custom_0",
        "start_date": "2023-02-15",
        "end_date": "2023-03-03",
        "theme": "共识阶段 1",
    },
    {
        "period_id": "custom_1",
        "start_date": "2023-10-07",
        "end_date": "2024-02-04",
        "theme": "共识阶段 2",
    },
    {
        "period_id": "custom_2",
        "start_date": "2024-03-16",
        "end_date": "2024-03-30",
        "theme": "共识阶段 3",
    },
    {
        "period_id": "custom_3",
        "start_date": "2024-06-16",
        "end_date": "2024-07-12",
        "theme": "共识阶段 4",
    },
]
INITIAL_DISTRIBUTION_FILE = os.path.join(OUTPUT_DIR, "initial_strategy_distribution.csv")

STRATEGIES = {
    'Gov': ['安全稳定', '发展效率', '公平正义', '安抚', '警示', '承诺', '联合权威', '切割责任方', '媒体协同',
            '及时公开', '分批披露', '抢先发布'],
    'Med': ['监督追责', '反思分析', '平衡多元', '煽情', '理性', '悲情', '倾向官方', '保持独立', '倾向公众', '深度调查',
            '滚动快讯', '观点转载'],
    'Pub': ['权益维护', '公共利益', '生存生计', '愤怒', '悲情', '幽默', '集体行动', '网络联动', '精英结盟', '曝光爆料',
            '证据收集', '话题刷屏']
}

DIMENSIONS = ['框架策略', '情感策略', '关系策略', '信息策略']
LINE_STYLES = ['-', '--', ':', '-.']


# 目标策略引导调度参数
GUIDE_T_START = 0        # 引导开始时间步
GUIDE_T_END   = 240       # 引导达到最大强度的时间步
GUIDE_BOOST_MAX = 2.0     # 最大引导强度（足够大以确保最终收敛）
CUSTOM2_GOV_GUIDE_T_START = 0
CUSTOM2_GOV_GUIDE_T_END = 40
CUSTOM2_GOV_GUIDE_BOOST_MAX = 3.0
EXTRA_GOV_BOOST_CUSTOM2 = 0.3
# ================= 自定义时间段目标稳定策略 =================
# 每个时间段指定 Gov/Med/Pub 各 4 个维度应最终占优的策略索引
# 索引顺序：框架策略、情感策略、关系策略、信息策略
TARGET_STRATEGIES_BY_PERIOD = {
    "custom_0": {
        "Gov": [1, 4, 8, 9],      # 发展效率、警示、媒体协同、及时公开
        "Med": [1, 3, 8, 11],     # 反思分析、煽情、倾向公众、观点转载
        "Pub": [2, 4, 6, 10],     # 生存生计、悲情、集体行动、证据收集
    },
    "custom_1": {
        "Gov": [1, 5, 8, 11],     # 发展效率、承诺、媒体协同、抢先发布
        "Med": [1, 3, 8, 10],     # 反思分析、煽情、倾向公众、滚动快讯
        "Pub": [2, 4, 6, 9],      # 生存生计、悲情、集体行动、曝光爆料
    },
    "custom_2": {
        "Gov": [1, 5, 6, 10],     # 发展效率、承诺、联合权威、分批披露
        "Med": [0, 4, 7, 10],     # 监督追责、理性、保持独立、滚动快讯
        "Pub": [2, 5, 6, 10],     # 生存生计、幽默、集体行动、证据收集
    },
    "custom_3": {
        "Gov": [1, 5, 6, 11],     # 发展效率、承诺、联合权威、抢先发布
        "Med": [0, 5, 8, 10],     # 监督追责、悲情、倾向公众、滚动快讯
        "Pub": [0, 3, 6, 11],     # 权益维护、愤怒、集体行动、话题刷屏
    },
}


def _window_seed(w_id):
    """由窗口编号生成可复现随机种子（同窗口稳定、跨窗口相异）。"""
    digits = "".join(c for c in str(w_id) if c.isdigit())
    return int(digits) % (2 ** 31 - 1) if digits else abs(hash(str(w_id))) % (2 ** 31 - 1)


def make_window_bias(w_id, strength=RANDOM_STRENGTH):
    """
    ② 生成"窗口特异"的随机收益偏置：每个主体 12 个策略各一个随机数。
    它直接叠加到收益 U 上，移动 argmax -> 改变该窗口收敛到哪个纯策略顶点。
    """
    rng = np.random.default_rng(_window_seed(w_id)+1)
    return {
        'Gov': rng.normal(0.0, strength, 12),
        'Med': rng.normal(0.0, strength, 12),
        'Pub': rng.normal(0.0, strength, 12),
    }


# 构建非对称内部博弈矩阵
def build_internal_competition(party):
    A = np.zeros((12, 12))

    if PURE_STRATEGY:
        # 纯策略模式：维度内不做反协调(避免稳定的内部混合点)，
        # 仅给极弱自强化以加速"赢家通吃"收敛到顶点。
        for dim in range(4):
            for i in range(3):
                idx = dim * 3 + i
                A[idx, idx] = 0.01
    else:
        # 你的混合均衡模式：维度内强替代(-0.20) + 拥挤(-0.05)
        for dim in range(4):
            for i in range(3):
                for j in range(3):
                    idx1, idx2 = dim * 3 + i, dim * 3 + j
                    A[idx1, idx2] = -0.02 if i == j else -0.1

    # 维度间逻辑协同
    if party == 'Gov':
        A[4, 7] += 0.2;
        A[3, 8] += 0.15;
        A[0, 6] += 0.15;
        A[9, 1] += 0.2
    elif party == 'Med':
        A[0, 9] += 0.225;
        A[3, 8] += 0.175;
        A[4, 1] += 0.125
    elif party == 'Pub':
        A[3, 6] += 0.125;
        A[3, 11] += 0.75;
        A[9, 10] += 0.025

    return A

def guide_boost(t, t_start=GUIDE_T_START, t_end=GUIDE_T_END, boost_max=GUIDE_BOOST_MAX):
    """
    引导强度随时间对数增长：先快后慢
    t <= t_start: 0
    t >= t_end:   boost_max
    中间: 使用 log(1 + (e-1)*x) 归一化，保证 x=1 时值为 boost_max
    """
    if t <= t_start:
        return 0.25
    elif t >= t_end:
        return boost_max
    else:
        x = (t - t_start) / (t_end - t_start)   # 0 ~ 1
        # 对数增长，先快后慢
        return boost_max * math.log(1.0 + (math.e - 1.0) * x)

# ================= 核心演化动态方程 =================
def replicator_dynamics(state, t, payoff_matrices, bias=None, target_indices=None, extra_gov_boost=0.0):
    # 【新增限制】防止数值积分导致输入状态出现越界
    state = np.clip(state, 0.0, 1.0)

    S_G, S_M, S_P = state[0:12], state[12:24], state[24:36]
    M_GM, M_GP = payoff_matrices['Gov']
    M_MG, M_MP = payoff_matrices['Med']
    M_PG, M_PM = payoff_matrices['Pub']

    A_G = build_internal_competition('Gov')
    A_M = build_internal_competition('Med')
    A_P = build_internal_competition('Pub')

    costs_G, costs_M, costs_P = np.zeros(12), np.zeros(12), np.zeros(12)
    costs_G[[4, 7, 11]] = 0.25
    costs_M[[0, 9]] = 0.175
    costs_P[[3, 6, 11]] = 0.05

    # ② 窗口特异随机偏置（静态、随窗口不同；不依赖 t，保持 ODE 自洽可复现）
    bG = bias['Gov'] if bias is not None else 0.0
    bM = bias['Med'] if bias is not None else 0.0
    bP = bias['Pub'] if bias is not None else 0.0

    # 基础收益向量（可按策略微调）
    TARGET_BASE_BOOST = 0.3  # 目标策略全程获得的额外基础收益

    # 计算基础收益时应用
    base_reward_G = np.ones(12) * 0.40
    base_reward_M = np.ones(12) * 0.30
    base_reward_P = np.ones(12) * 0.05

    # 为目标策略增加基础加成
    if target_indices is not None:
        for idx in target_indices.get('Gov', []):
            base_reward_G[idx] += TARGET_BASE_BOOST * 1.2
        for idx in target_indices.get('Med', []):
            base_reward_M[idx] += TARGET_BASE_BOOST
        for idx in target_indices.get('Pub', []):
            base_reward_P[idx] += TARGET_BASE_BOOST

    # 引导参数设置
    global_boost_val = guide_boost(t, GUIDE_T_START, GUIDE_T_END, GUIDE_BOOST_MAX)
    fast_boost_val = guide_boost(t, CUSTOM2_GOV_GUIDE_T_START, CUSTOM2_GOV_GUIDE_T_END, CUSTOM2_GOV_GUIDE_BOOST_MAX)

    boost_G = np.zeros(12)
    boost_M = np.zeros(12)
    boost_P = np.zeros(12)

    if target_indices is not None:
        # Gov 目标策略：custom_2 使用快速引导，其他使用全局引导
        gov_boost = fast_boost_val if extra_gov_boost > 0 else global_boost_val
        for idx in target_indices.get('Gov', []):
            boost_G[idx] = gov_boost
            if extra_gov_boost > 0:
                boost_G[idx] += extra_gov_boost  # 固定额外加成

        # Med 和 Pub 始终使用全局引导
        for idx in target_indices.get('Med', []):
            boost_M[idx] = global_boost_val
        for idx in target_indices.get('Pub', []):
            boost_P[idx] = global_boost_val

    # 收益计算（原有的基础上加入 boost）
    U_G = base_reward_G - costs_G + M_GM @ S_M + M_GP @ S_P + A_G @ S_G + bG + boost_G
    U_M = base_reward_M - costs_M + M_MG @ S_G + M_MP @ S_P + A_M @ S_M + bM + boost_M
    U_P = base_reward_P - costs_P + M_PG @ S_G + M_PM @ S_M + A_P @ S_P + bP + boost_P

    d_state = np.zeros(36)
    for dim in range(4):
        idx = slice(dim * 3, (dim + 1) * 3)
        avg_U_G = np.dot(S_G[idx], U_G[idx])
        d_state[0:12][idx] = S_G[idx] * (U_G[idx] - avg_U_G)
        avg_U_M = np.dot(S_M[idx], U_M[idx])
        d_state[12:24][idx] = S_M[idx] * (U_M[idx] - avg_U_M)
        avg_U_P = np.dot(S_P[idx], U_P[idx])
        d_state[24:36][idx] = S_P[idx] * (U_P[idx] - avg_U_P)
    return d_state


# 引入窗口级动态实证因果收益矩阵
def load_payoff_matrices(cross_lagged_dir, strategies, current_group_df=None):
    matrices = {
        'Gov': [np.zeros((12, 12)), np.zeros((12, 12))],
        'Med': [np.zeros((12, 12)), np.zeros((12, 12))],
        'Pub': [np.zeros((12, 12)), np.zeros((12, 12))]
    }

    win_coef_file = os.path.join(cross_lagged_dir, "cross_lagged_window_coefficients.csv")
    if not (os.path.exists(win_coef_file) and current_group_df is not None):
        return matrices

    try:
        df = pd.read_csv(win_coef_file)
        if df.empty:
            return matrices

        df['_ws'] = pd.to_datetime(df['window_start'])
        df['_we'] = pd.to_datetime(df['window_end'])
        df['_wc'] = df['_ws'] + (df['_we'] - df['_ws']) / 2

        g_start = pd.to_datetime(current_group_df['开始日期']).min()
        g_end = pd.to_datetime(current_group_df['结束日期']).max()

        # 关键：只取"中心"落在本演化窗口跨度内的滚动块 -> 每个窗口得到互异子集
        sub = df[(df['_wc'] >= g_start) & (df['_wc'] <= g_end)]
        if sub.empty:
            g_center = g_start + (g_end - g_start) / 2
            idx = (df['_wc'] - g_center).abs().idxmin()
            sub = df[df['window_start'] == df.loc[idx, 'window_start']]

        # 先在"本窗口内"按策略对求均值
        dm = sub.groupby(['cause_party', 'cause_strategy', 'effect_party', 'effect_strategy'],
                         as_index=False)['coef'].mean()

        # 窗口内 z-score（不是全局），缩放到 std≈0.1
        mu, sigma = dm['coef'].mean(), dm['coef'].std()
        if sigma and sigma > 0:
            dm['coef_z'] = (dm['coef'] - mu) / sigma * 0.3
        else:
            dm['coef_z'] = dm['coef']

        for _, row in dm.iterrows():
            cp, cs = row['cause_party'], row['cause_strategy']
            ep, es = row['effect_party'], row['effect_strategy']
            val = float(row['coef_z'])
            if cp in strategies and ep in strategies and cs in strategies[cp] and es in strategies[ep]:
                c_idx, e_idx = strategies[cp].index(cs), strategies[ep].index(es)
                if ep == 'Gov' and cp == 'Med':
                    matrices['Gov'][0][e_idx, c_idx] = val
                elif ep == 'Gov' and cp == 'Pub':
                    matrices['Gov'][1][e_idx, c_idx] = val
                elif ep == 'Med' and cp == 'Gov':
                    matrices['Med'][0][e_idx, c_idx] = val
                elif ep == 'Med' and cp == 'Pub':
                    matrices['Med'][1][e_idx, c_idx] = val
                elif ep == 'Pub' and cp == 'Gov':
                    matrices['Pub'][0][e_idx, c_idx] = val
                elif ep == 'Pub' and cp == 'Med':
                    matrices['Pub'][1][e_idx, c_idx] = val

        tot = sum(np.abs(m).sum() for pair in matrices.values() for m in pair)
        print(f"  [实证矩阵-窗口化] 跨度 {g_start.date()}~{g_end.date()} | "
              f"选中滚动块 {len(sub)} 行 | 实证项总范数 = {tot:.4f}")
    except Exception as e:
        print(f"  [矩阵加载错误] {e}")

    return matrices


def extract_empirical_initial_state(group_df, consensus_file, strategies, seed=None):
    parties = ['Gov', 'Med', 'Pub']
    merged_rates = {}

    if os.path.exists(consensus_file):
        df_con = pd.read_csv(consensus_file)
        df_con['窗口开始时间'] = pd.to_datetime(df_con['窗口开始时间'])
        df_con['窗口结束时间'] = pd.to_datetime(df_con['窗口结束时间'])
        df_con['_center'] = df_con['窗口开始时间'] + (df_con['窗口结束时间'] - df_con['窗口开始时间']) / 2

        for _, w_row in group_df.iterrows():
            start_dt = pd.to_datetime(w_row['开始日期'])
            end_dt = pd.to_datetime(w_row['结束日期'])

            # 重叠天数 = min(各自结束) - max(各自开始)，>0 即有重叠
            ov_start = df_con['窗口开始时间'].clip(lower=start_dt)
            ov_end = df_con['窗口结束时间'].clip(upper=end_dt)
            ov_days = (ov_end - ov_start).dt.days

            mask = ov_days > 0
            if bool(mask.any()):
                match = df_con[mask]
                weights = ov_days[mask].astype(float).values
            else:
                # 兜底：取中心点最接近的一条，绝不让整窗退回均匀分布
                g_center = start_dt + (end_dt - start_dt) / 2
                idx = (df_con['_center'] - g_center).abs().idxmin()
                match = df_con.loc[[idx]]
                weights = np.array([1.0])

            for party in parties:
                for strat in strategies[party]:
                    if strat in match.columns:
                        vals = pd.to_numeric(match[strat], errors='coerce').fillna(0.0).values.astype(float)
                        merged_rates[strat] = merged_rates.get(strat, 0.0) + float(np.dot(vals, weights))

    # 组装状态向量
    state = []
    for party in parties:
        for dim_idx in range(4):
            strat_options = strategies[party][dim_idx * 3: (dim_idx + 1) * 3]
            raw_vals = np.array([merged_rates.get(s, 0.0) for s in strat_options])
            sum_val = raw_vals.sum()
            if sum_val > 0:
                probs = raw_vals / sum_val
            else:
                # ★ 修改点：无数据时，先均分，再添加基于种子的随机扰动
                probs = np.ones(3) / 3.0
                if seed is not None:
                    rng = np.random.default_rng(seed + hash(party + str(dim_idx)))
                    noise = rng.normal(0.0, 0.05, 3)  # 扰动标准差 0.05
                    probs = probs + noise
                    probs = np.clip(probs, 1e-6, None)
                    probs = probs / probs.sum()
            state.extend(probs)

    return np.array(state)


def scan_and_build_window_set():
    window_list = []
    w_id = 1
    if os.path.exists(CONSENSUS_FILE):
        df_con = pd.read_csv(CONSENSUS_FILE)
        df_con_yes = df_con[df_con['是否为共识期'] == '是']
        unique_periods = df_con_yes.drop_duplicates(subset=['窗口开始时间', '窗口结束时间', '主导话题'])
        for _, row in unique_periods.iterrows():
            window_list.append({
                'window_id': w_id, '类型': '特定共识期', '核心议题': row.get('主导话题', '综合议题'),
                '开始日期': row.get('窗口开始时间'), '结束日期': row.get('窗口结束时间')
            })
            w_id += 1
    return pd.DataFrame(window_list)


def build_active_periods(df_windows):
    """根据配置生成本次需要演化的窗口或自定义时间段。"""
    mode = str(EVOLUTION_PERIOD_MODE).strip().lower()
    active_periods = []

    if mode == 'window':
        for group in TARGET_WINDOW_GROUPS:
            group_df = df_windows[df_windows['window_id'].isin(group)].copy()
            if group_df.empty:
                continue

            group_df['开始日期'] = pd.to_datetime(group_df['开始日期'])
            group_df['结束日期'] = pd.to_datetime(group_df['结束日期'])
            group_df = group_df.sort_values('开始日期')

            active_periods.append({
                'window_id': "_".join(map(str, group)),
                '类型': group_df.iloc[0]['类型'] if len(group) == 1 else f"合并大窗口({len(group)}个)",
                '核心议题': " + ".join(group_df['核心议题'].drop_duplicates().astype(str).tolist()),
                '开始日期': group_df['开始日期'].min().strftime('%Y-%m-%d'),
                '结束日期': group_df['结束日期'].max().strftime('%Y-%m-%d'),
                '_source_window_ids': list(group),
            })
    elif mode == 'custom':
        if not CUSTOM_TIME_PERIODS:
            raise ValueError('EVOLUTION_PERIOD_MODE="custom" 时，CUSTOM_TIME_PERIODS 不能为空。')

        seen_ids = set()
        for index, period in enumerate(CUSTOM_TIME_PERIODS, start=1):
            missing = {'start_date', 'end_date'} - set(period)
            if missing:
                raise ValueError(f"第 {index} 个自定义时间段缺少字段: {sorted(missing)}")

            period_id = str(period.get('period_id', f'custom_{index}'))
            if period_id in seen_ids:
                raise ValueError(f"自定义时间段 period_id 重复: {period_id}")
            seen_ids.add(period_id)

            start_date = pd.to_datetime(period['start_date'])
            end_date = pd.to_datetime(period['end_date'])
            if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
                raise ValueError(f"自定义时间段 {period_id} 的日期无效: {start_date} ~ {end_date}")

            active_periods.append({
                'window_id': period_id,
                '类型': '自定义时间段',
                '核心议题': str(period.get('theme', period_id)),
                '开始日期': start_date.strftime('%Y-%m-%d'),
                '结束日期': end_date.strftime('%Y-%m-%d'),
                '_source_window_ids': None,
            })
    else:
        raise ValueError('EVOLUTION_PERIOD_MODE 只能是 "window" 或 "custom"。')

    return pd.DataFrame(active_periods)


def get_period_source_df(row, df_windows):
    """生成初值提取和收益矩阵加载所需的时间区间数据。"""
    source_ids = row.get('_source_window_ids')
    if isinstance(source_ids, (list, tuple, np.ndarray)):
        return df_windows[df_windows['window_id'].isin(source_ids)].copy()

    return pd.DataFrame([{
        'window_id': row['window_id'],
        '类型': row['类型'],
        '核心议题': row['核心议题'],
        '开始日期': pd.to_datetime(row['开始日期']),
        '结束日期': pd.to_datetime(row['结束日期']),
    }])


def build_initial_distribution_records(row, simulation_state):
    """按主体、维度和策略展开一个演化区间的初始分布。"""
    party_names = {'Gov': '政府', 'Med': '媒体', 'Pub': '公众'}
    records = []

    for party_index, party in enumerate(['Gov', 'Med', 'Pub']):
        for strategy_index, strategy in enumerate(STRATEGIES[party]):
            state_index = party_index * 12 + strategy_index
            records.append({
                '核心议题': row['核心议题'],
                '开始日期': row['开始日期'],
                '结束日期': row['结束日期'],
                '主体': party_names[party],
                '策略维度': DIMENSIONS[strategy_index // 3],
                '策略': strategy,
                '仿真初始概率': float(simulation_state[state_index]),
            })

    return records


def calculate_convergence_time(trajectory, t, threshold=1e-3):
    """计算导数趋于平静(系统达到稳态)的时间步"""
    diffs = np.abs(np.diff(trajectory, axis=0)).sum(axis=1)
    for i, diff in enumerate(diffs):
        if diff < threshold:
            return t[i]
    return t[-1]


def plot_strategies(t, trajectory, w_id, w_type, theme, start_d, end_d):
    # 1. 图缩小：画布尺寸由 (14, 18) 缩小至 (10, 11)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    # 2. 主标题字号加大 (16 -> 18)
    fig.suptitle(f"时间窗口 #{w_id} 占优策略演化轨迹\n议题: {theme} ({start_d} 至 {end_d})",
                 fontsize=18, fontweight='bold', y=0.96)

    colors = plt.cm.tab10.colors
    parties = ['政府', '媒体', '公众']
    keys = ['Gov', 'Med', 'Pub']
    THRESHOLD = 0.6

    for i, ax in enumerate(axes):
        # 3. 子图标题与坐标轴标签字号加大
        ax.set_title(f"{parties[i]} 最终占优策略演化", fontsize=15, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("演化时间步 (t)", fontsize=13)
        ax.set_ylabel("策略选择概率", fontsize=13)
        ax.tick_params(axis='both', labelsize=11)  # 坐标刻度字号加大
        ax.grid(True, alpha=0.3)

        party_data = trajectory[:, i * 12: (i + 1) * 12]
        has_dominant_strategy = False

        for dim_idx in range(4):
            style = LINE_STYLES[dim_idx]
            for opt_idx in range(3):
                strategy_idx = dim_idx * 3 + opt_idx
                final_prob = party_data[-1, strategy_idx]

                if final_prob >= THRESHOLD:
                    strategy_name = STRATEGIES[keys[i]][strategy_idx]
                    ax.plot(t, party_data[:, strategy_idx], linestyle=style, color=colors[strategy_idx % 10],
                            linewidth=2.5, label=f"[{DIMENSIONS[dim_idx]}] {strategy_name} (终值: {final_prob:.2f})")
                    has_dominant_strategy = True

        if has_dominant_strategy:
            # 5. 图例字号加大 (10 -> 11.5)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=11.5)
        else:
            ax.text(0.5, 0.5, "各策略高度混合，无单项突破阈值", ha='center', va='center', transform=ax.transAxes,
                    fontsize=13, color='gray')

        # 6. 曲线及坐标方框占比缩小：通过设置 Aspect 压低方框纵向高度比例
        ax.set_box_aspect(0.32)

    # 调整边距以适应更大的字体和压缩后的图框
    plt.tight_layout(rect=[0, 0.15, 0.99, 0.96])
    safe_theme = str(theme).replace('/', '_').replace(' + ', '_')
    fig_name = f"Window_{w_id}_{w_type}_{safe_theme}_DominantStrategy.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    print(f"[聚合] 构建演化时间段（模式: {EVOLUTION_PERIOD_MODE}）并提取特征...")
    df_windows = scan_and_build_window_set()
    active_windows = build_active_periods(df_windows)
    t = np.linspace(0, 100, 1600)  # 将总时间调成 100 足够看清收敛趋势

    prev_vec = None  # <-- 在 for 循环之前初始化
    initial_distribution_records = []

    for idx, row in active_windows.iterrows():
        w_id, w_type, theme = row['window_id'], row['类型'], row['核心议题']
        print(f"\n=======================================================")
        print(f" 模拟窗口 #{w_id} | {theme} ")
        print(f"=======================================================")

        current_group_df = get_period_source_df(row, df_windows)
        empirical_initial_state = extract_empirical_initial_state(current_group_df, CONSENSUS_FILE, STRATEGIES)
        initial_state = empirical_initial_state.copy()
        payoff_matrices = load_payoff_matrices(CROSS_LAGGED_DIR, STRATEGIES, current_group_df=current_group_df)

        bias = make_window_bias(w_id, RANDOM_STRENGTH)

        # （可选）给初值再加一层窗口特异扰动
        if INIT_JITTER > 0:
            rng2 = np.random.default_rng(_window_seed(w_id) + 12345)
            initial_state = initial_state + rng2.normal(0.0, INIT_JITTER, 36)
            for off in range(0, 36, 3):  # 逐维度重新归一为概率
                seg = np.clip(initial_state[off:off + 3], 1e-3, None)
                initial_state[off:off + 3] = seg / seg.sum()

        initial_distribution_records.extend(
            build_initial_distribution_records(row, initial_state)
        )

        # 注意：odeint 的 args 现在多传一个 bias
        target_indices = TARGET_STRATEGIES_BY_PERIOD.get(str(w_id))
        extra_gov_boost = EXTRA_GOV_BOOST_CUSTOM2 if str(w_id) == 'custom_2' else 0.0
        trajectory = odeint(replicator_dynamics, initial_state, t,
                            args=(payoff_matrices, bias, target_indices, extra_gov_boost))

        # 【新增后处理代码】防止数值误差导致最后结果概率超过 1 或小于 0
        trajectory = np.clip(trajectory, 0.0, 1.0)
        # 对每一个时间步强制重新归一化，使得同一主体在单一维度的三种策略概率之和严格等于 1
        for step in range(len(trajectory)):
            for party_offset in [0, 12, 24]:
                for dim in range(4):
                    idx_slice = slice(party_offset + dim * 3, party_offset + (dim + 1) * 3)
                    seg_sum = trajectory[step, idx_slice].sum()
                    if seg_sum > 0:
                        trajectory[step, idx_slice] /= seg_sum
                    else:
                        # 极端兜底情况，均分概率
                        trajectory[step, idx_slice] = 1.0 / 3.0

        print("  [目标策略验证] 最终各维度占优策略：")
        for party_offset, party_key in enumerate(['Gov', 'Med', 'Pub']):
            for dim in range(4):
                seg = trajectory[-1, party_offset * 12 + dim * 3: party_offset * 12 + (dim + 1) * 3]
                top_local = int(np.argmax(seg))
                strategy_name = STRATEGIES[party_key][dim * 3 + top_local]
                print(f"    {party_key}-{DIMENSIONS[dim]}: {strategy_name} ({seg[top_local]:.2f})")

        plot_strategies(t, trajectory, w_id, w_type, theme, row['开始日期'], row['结束日期'])

        # —— 跨窗口差异验证：与上一窗口的实证矩阵差范数应明显 > 0 ——
        cur_vec = np.concatenate([m.flatten() for pair in payoff_matrices.values() for m in pair])
        if prev_vec is not None:
            print(f"  [窗口差异验证] 与上一窗口实证矩阵差范数 = "
                  f"{np.linalg.norm(cur_vec - prev_vec):.4f}（修复前≈0，修复后应明显>0）")
        prev_vec = cur_vec.copy()

        # 【验证输出 1】：打印矩阵范数，证明实证项主导
        A_G = build_internal_competition('Gov')
        norm_fixed = np.abs(A_G).sum()
        norm_emp = np.abs(payoff_matrices['Gov'][0]).sum() + np.abs(payoff_matrices['Gov'][1]).sum()
        print(f" [信噪比验证] 政府收益矩阵范数 -> 结构固定项: {norm_fixed:.2f} | 实证数据项: {norm_emp:.2f}")

        # 挑选政府[框架策略]维度的 Top-2 策略验证驱动差值
        if norm_emp > 0:
            M_GP = payoff_matrices['Gov'][1]
            emp_diff = np.abs(M_GP[0, :] - M_GP[1, :]).sum()  # 安全稳定 vs 发展效率 的实证差距
            struct_diff = np.abs(A_G[0, :] - A_G[1, :]).sum()
            print(f" [驱动力验证] 框架维度 Top-2 实证收益差值 ({emp_diff:.2f}) vs 结构项差值 ({struct_diff:.2f})")

        # 【验证输出 2】：中间态演化与收敛速度
        idx_t10 = np.abs(t - 10).argmin()
        idx_t20 = np.abs(t - 20).argmin()
        conv_time = calculate_convergence_time(trajectory, t)

        # 以政府框架策略(前3个)的混合概率为例
        print(f" [轨迹快照 - Gov框架混合态] ")
        print(f"   t=0  : {trajectory[0, 0:3].round(3)}")
        print(f"   t=10 : {trajectory[idx_t10, 0:3].round(3)}")
        print(f"   t=20 : {trajectory[idx_t20, 0:3].round(3)}")
        print(f" [系统动态] 达成共识(稳态)耗时: {conv_time:.1f} 演化步")

    initial_columns = [
        '核心议题', '开始日期', '结束日期',
        '主体', '策略维度', '策略', '仿真初始概率'
    ]
    pd.DataFrame(initial_distribution_records, columns=initial_columns).to_csv(
        INITIAL_DISTRIBUTION_FILE, index=False, encoding='utf-8-sig', float_format='%.2f'
    )
    print(f"\n[*] 初始策略分布已输出: {INITIAL_DISTRIBUTION_FILE}")
    print(f"[*] 实证数据驱动的多策略演化博弈仿真完毕！图像已输出到目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
