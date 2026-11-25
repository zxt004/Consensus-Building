from __future__ import annotations
from matplotlib.transforms import blended_transform_factory
import os
import re
import colorsys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "/root/pmd/SIMSUN.TTC"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

data_path = '/root/autodl-tmp/预制菜/预制菜output.xlsx'
output_path = '/root/autodl-tmp/预制菜/预制菜output.xlsx'
river_output_path = '/root/autodl-tmp/预制菜/预制菜_主题河流图.svg'

# 1. 采用“主题名称+关键词+定义性描述”三个维度的数据结构
public_themes = {
    "食品安全与营养认知": {
        "description": "公众围绕预制菜的食品安全标准、营养流失及新鲜度展开讨论，尤其是面向学生等群体的食堂应用场景",
        "keywords": ["校园", "食品", "冷冻", "食品安全", "营养", "食堂", "学生", "新鲜"],
    },
    "消费者与企业家立场差异": {
        "description": "公众在事件中形成“普通消费者”与“知名企业家”之间的立场对立，将讨论从产品层面上升至阶层、话语权和利益关系的博弈",
        "keywords": ["罗永浩", "消费者", "贾国龙", "老板", "顾客"],
    },
    "产品供应链透明度关注": {
        "description": "公众对预制菜企业是否诚信告知产品来源、制作方式及供应链透明度提出质疑",
        "keywords": ["西贝货", "假货", "西贝", "蜀海", "厨房"],
    },
    "个体消费行为倾向": {
        "description": "公众基于对预制菜品牌及品类的评价，以个人消费选择作为立场的直接体现",
        "keywords": ["面条", "没吃过", "不吃", "不去西贝", "反正", "去过", "以后", "好吃"],
    },
    "社会发展隐喻讨论": {
        "description": "公众将预制菜的普及视为教育功利化、社会过度追求效率而牺牲品质的隐喻",
        "keywords": ["快乐教育", "美国", "孩子", "大学", "教学", "学习", "预制菜"],
    },
    "国际品牌参照与消费预期": {
        "description": "公众以国际或成熟市场的同类产品为参照坐标，表达对本土品牌在品质、价格及消费体验上的预期落差感受",
        "keywords": ["萨莉亚", "香港", "照片", "老周", "360", "罗永浩", "医生", "没有萨莉亚"],
    },
    "历史经验与系统信任": {
        "description": "公众将预制菜事件与三聚氰胺奶粉等历史食品安全事件进行联结，当下的不信任被视为过往创伤经验累积而成的系统性心态",
        "keywords": ["奶粉", "牛奶", "三鹿", "狗屎", "三聚氰胺", "内蒙", "标准", "现挤", "国标"],
    },
    "语言表达与身份标签": {
        "description": "公众围绕特定企业家及发言者的言论展开讨论，涉及标签化的话语争议，反映舆论场中情绪化的身份对立已超出事实讨论的范畴",
        "keywords": ["网络黑社会", "老罗", "真正·黑社会", "罗永浩", "龙哥", "内蒙"],
    },
    "技术风险认知": {
        "description": "公众对预制菜加工过程中可能使用的辐照、添加剂等技术手段表达不安全感，形成一种难以精准定位但普遍存在的技术风险疑虑",
        "keywords": ["鱼刺", "辐射", "残留", "检测", "物质", "能量", "鱼肉", "去除"]
    }
}

other_themes = {
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

# 2. 加载 BERT 模型
model_name = 'shibing624/text2vec-base-chinese'
print(f"正在加载模型 {model_name}...")
model = SentenceTransformer(model_name)


def validate_and_load_excel(filepath):
    print(f"\n[文件完整性检查] 正在验证: {filepath}")

    # 1. 检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ 文件不存在: {filepath}")

    # 2. 检查文件大小是否为 0
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError(f"❌ 文件大小为 0 字节，文件可能已损坏: {filepath}")
    print(f"   ✓ 文件存在，大小: {file_size} 字节")

    # 3. 尝试读取 Excel，捕获损坏/格式错误异常
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        raise IOError(f"❌ 读取 Excel 文件失败 (文件损坏或格式不兼容): {e}")

    print(f"   ✓ 成功读取，总行数: {len(df)}，总列数: {len(df.columns)}")

    # 4. 检查是否为空表
    if df.empty:
        raise ValueError("❌ 数据框为空，文件中没有数据行")

    # 5. 检查必需的列是否存在
    required_cols = ['情感得分', '蒸馏总结', '账号分类']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ 缺少必需的列: {missing_cols}，当前列名: {list(df.columns)}")
    print(f"   ✓ 必需列检查通过: {required_cols}")

    # 6. 检查关键列是否包含有效数据（非空、非纯空格）
    # 情感得分
    temp_sent = df['情感得分'].replace(r'^\s*$', np.nan, regex=True).dropna()
    if len(temp_sent) == 0:
        print("   ⚠️ 警告: '情感得分' 列全部为空或无效，后续分析可能无意义")
    else:
        print(f"   ✓ '情感得分' 有效非空值数量: {len(temp_sent)}/{len(df)}")

    # 蒸馏总结
    temp_text = df['蒸馏总结'].replace(r'^\s*$', np.nan, regex=True).dropna()
    if len(temp_text) == 0:
        print("   ⚠️ 警告: '蒸馏总结' 列全部为空或无效，后续分类可能无结果")
    else:
        print(f"   ✓ '蒸馏总结' 有效非空值数量: {len(temp_text)}/{len(df)}")

    # 账号分类
    temp_acc = df['账号分类'].replace(r'^\s*$', np.nan, regex=True).dropna()
    if len(temp_acc) == 0:
        print("   ⚠️ 警告: '账号分类' 列全部为空")
    else:
        print(f"   ✓ '账号分类' 非空数量: {len(temp_acc)}/{len(df)}")
        # 顺便检查是否包含预期的分类关键词
        sample_vals = temp_acc.astype(str).str.lower().unique()[:5]
        print(f"   ℹ️  账号分类样例: {list(sample_vals)}")

    print("[文件完整性检查] ✅ 验证通过\n")
    return df


def get_theme_vectors(themes_dict):
    theme_names = []
    theme_super_texts = []

    # 将名称、定义、关键词拼接成一段高信息密度的超级文本
    for name, content in themes_dict.items():
        theme_names.append(name)
        desc = content["description"]
        kw_str = "、".join(content["keywords"])

        super_text = f"议题名称：{name}。核心定义：{desc} 高频关键词包含：{kw_str}。"
        theme_super_texts.append(super_text)
        print(f"  - 生成特征锚点: {super_text}")

    theme_vectors = model.encode(theme_super_texts)
    return theme_names, np.array(theme_vectors)


print("正在构建并计算各个分类主题的三维特征向量...")
public_names, public_vectors = get_theme_vectors(public_themes)
other_names, other_vectors = get_theme_vectors(other_themes)

# 3. 加载包含评论数据的 Excel 文件
print(f"\n正在加载数据: {data_path} ...")
df = validate_and_load_excel(data_path)

print("正在清理“情感得分”为空的数据...")
initial_len = len(df)
df['情感得分'] = df['情感得分'].replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(subset=['情感得分'])
print(f"清理完毕！删除了 {initial_len - len(df)} 条数据，剩余数据量: {len(df)}")

text_column = '蒸馏总结'
df[text_column] = df[text_column].fillna("").astype(str)

if '账号分类' not in df.columns:
    raise ValueError("错误：数据源中不存在 '账号分类' 列，请检查文件内容！")

is_public_mask = df['账号分类'].astype(str).str.lower().str.contains('public')
df_public = df[is_public_mask].copy()
df_other = df[~is_public_mask].copy()

THRESHOLD = 0.75


def process_classification(sub_df, t_names, t_vectors):
    if sub_df.empty:
        sub_df['匹配的主题标签'] = []
        sub_df['各议题相似度(概率)'] = []
        return sub_df

    print(f"--> 正在转化 {len(sub_df)} 条文本并计算相似度...")
    comment_embeddings = model.encode(sub_df[text_column].tolist(), batch_size=64, show_progress_bar=True)

    # 直接计算余弦相似度，不再减去 mean_embedding
    similarities = cosine_similarity(comment_embeddings, t_vectors)

    labels = []
    probs_combined = []

    for sim_array in similarities:
        # 【新增逻辑】1. 先找出所有过阈值的 (索引, 得分)
        valid_matches = [(i, sim_array[i]) for i in range(len(t_names)) if sim_array[i] > THRESHOLD]

        # 【新增逻辑】2. 按照得分从高到低排序
        valid_matches.sort(key=lambda x: x[1], reverse=True)

        # 【新增逻辑】3. 最多截取前 3 个
        top_3_matches = valid_matches[:3]

        # 4. 格式化输出
        matched = [f"{t_names[i]}({score:.2f})" for i, score in top_3_matches]

        # 兜底策略：如果都低于阈值，挑选最高的一个
        if not matched:
            max_idx = np.argmax(sim_array)
            if sim_array[max_idx] > (THRESHOLD - 0.1):
                matched = [f"{t_names[max_idx]}({sim_array[max_idx]:.2f})"]

        labels.append(" | ".join(matched) if matched else "无匹配")

        # 记录所有概率，依然保持原序输出以供调试/查看
        prob_str = ", ".join([f"{t_names[i]}: {sim_array[i]:.2f}" for i in range(len(t_names))])
        probs_combined.append(prob_str)

    sub_df['匹配的主题标签'] = labels
    sub_df['各议题相似度(概率)'] = probs_combined
    return sub_df


print("\n【开始处理 public 数据集】")
df_public = process_classification(df_public, public_names, public_vectors)

print("\n【开始处理 其他 数据集】")
df_other = process_classification(df_other, other_names, other_vectors)

print("\n正在合并数据并保存...")
df_final = pd.concat([df_public, df_other]).sort_index()

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_final.to_excel(output_path, index=False)
print(f"处理完成！处理后的合并数据已保存至：{output_path}")

# ================= 绘图步骤保留不变 =================
print("\n【开始生成公众舆情演化河流图】")


def get_topic_color(idx: int, total_topics: int) -> str:
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


def plot_theme_river(df_plot: pd.DataFrame, output_img: str):
    is_public = df_plot["账号分类"].astype(str).str.lower().str.contains("public")
    df_plot = df_plot[is_public].copy()

    if df_plot.empty:
        print(" -> [警告] 过滤后的 public 数据集为空，跳过河流图绘制。")
        return

    # 自动寻找时间列并解析日期
    time_col_found = False
    for tcand in ("日期", "时间", "创建时间", "发布时间", "parsed_time", "评论时间", "date"):
        if tcand in df_plot.columns:
            clean_time = (
                df_plot[tcand]
                .astype(str)
                .str.replace(r"(?:发布于|编辑于)\s*", "", regex=True)
                .str.strip()
            )
            time_strs = clean_time.str.extract(
                r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}(?:\s+\d{1,2}:\d{1,2})?)"
            )[0]
            time_strs = (
                time_strs.str.replace("年", "-", regex=False)
                .str.replace("月", "-", regex=False)
                .str.replace("日", "", regex=False)
            )
            df_plot["解析后时间"] = pd.to_datetime(time_strs, errors="coerce")
            # 保留原脚本的月份修正逻辑
            # df_plot["解析后时间"] = df_plot["解析后时间"] + pd.DateOffset(months=1)
            time_col_found = True
            break

    if not time_col_found or df_plot["解析后时间"].isna().all():
        print(" -> [警告] 数据中未找到有效的时间列(如'发布时间'等)，跳过河流图绘制。")
        return

    df_plot = df_plot.dropna(subset=["解析后时间"]).copy()
    df_plot["period"] = df_plot["解析后时间"].dt.to_period("D").dt.to_timestamp()

    def extract_themes(label_str):
        if pd.isna(label_str) or label_str == "无匹配":
            return []
        cleaned = re.sub(r"\([0-9.]+\)", "", str(label_str))
        return [t.strip() for t in cleaned.split("|") if t.strip()]

    df_plot["themes_list"] = df_plot["匹配的主题标签"].apply(extract_themes)
    df_plot = df_plot[df_plot["themes_list"].map(len) > 0]
    if df_plot.empty:
        print(" -> [警告] 未提取到任何有效主题，无法绘图。")
        return

    daily_post_counts = df_plot.groupby("period").size()
    df_exploded = df_plot.explode("themes_list").reset_index(drop=True)
    topic_counts = pd.crosstab(df_exploded["period"], df_exploded["themes_list"])

    all_themes = list(public_themes.keys())
    for theme in all_themes:
        if theme not in topic_counts.columns:
            topic_counts[theme] = 0
    topic_counts = topic_counts[all_themes]

    # 过滤掉帖子数过少的日期
    daily_post_counts = daily_post_counts.reindex(topic_counts.index, fill_value=0)
    topic_counts = topic_counts[daily_post_counts >= 5]
    if topic_counts.empty:
        print(" -> [警告] 过滤掉数据量小于5的日期后，无有效数据，无法绘图。")
        return

    dates = pd.DatetimeIndex(topic_counts.index)

    # 根据日期连续性拆分数据段
    segments: list[tuple[pd.DatetimeIndex, np.ndarray]] = []
    current_dates: list[pd.Timestamp] = []
    current_indices: list[int] = []
    for i, date in enumerate(dates):
        current_dates.append(date)
        current_indices.append(i)
        if i < len(dates) - 1 and (dates[i + 1] - dates[i]).days > 1:
            segments.append((pd.DatetimeIndex(current_dates), np.asarray(current_indices, dtype=int)))
            current_dates, current_indices = [], []
    if current_dates:
        segments.append((pd.DatetimeIndex(current_dates), np.asarray(current_indices, dtype=int)))

    # 将各段映射到压缩后的横轴坐标。
    # 段内仍按天排列，段间只保留固定宽度的断轴间隔，不再保留数月空白。
    gap_width = 0.5
    x_segments: list[np.ndarray] = []
    break_positions: list[float] = []
    cursor = 0.0
    for seg_idx, (seg_dates, _) in enumerate(segments):
        if seg_idx > 0:
            break_positions.append(cursor - 0.5 * gap_width)
            cursor += gap_width
        xs = cursor + np.arange(len(seg_dates), dtype=float)
        x_segments.append(xs)
        cursor = float(xs[-1] + 1.0)

    y = [np.log1p(topic_counts[theme].to_numpy()) for theme in all_themes]
    colors = [get_topic_color(i, len(all_themes)) for i in range(len(all_themes))]

    plt.rcParams["axes.unicode_minus"] = False
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # 分段绘制河流图，使用压缩后的 x 坐标
    for seg_idx, (_, seg_indices) in enumerate(segments):
        seg_y = [values[seg_indices] for values in y]
        if seg_idx == 0:
            ax1.stackplot(
                x_segments[seg_idx],
                seg_y,
                labels=all_themes,
                baseline="sym",
                colors=colors,
                alpha=0.65,
            )
        else:
            ax1.stackplot(
                x_segments[seg_idx],
                seg_y,
                baseline="sym",
                colors=colors,
                alpha=0.65,
            )

    ax1.set_xlabel("时间", fontsize=12)
    ax1.set_ylabel("议题分布规模趋势 (对数平滑平铺)", fontsize=12, color="gray")
    ax1.set_yticklabels([])
    ax1.tick_params(axis="y", length=0)

    # 为每个数据段选择少量真实日期作为刻度，避免压缩后标签拥挤
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for seg_dates, xs in zip((item[0] for item in segments), x_segments):
        n_ticks = min(4, len(seg_dates))
        tick_idx = np.unique(np.linspace(0, len(seg_dates) - 1, n_ticks).round().astype(int))
        tick_positions.extend(xs[tick_idx].tolist())
        tick_labels.extend([seg_dates[i].strftime("%Y-%m-%d") for i in tick_idx])
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    # 用虚线和 // 标出被省略的时间区间
    # 用横轴附近的虚线和 // 标出被省略的时间区间
    for break_x in break_positions:
        ax1.axvline(
            break_x,
            color="gray",
            linewidth=1.2,
            linestyle=(0, (2, 3)),
            alpha=0.9,
            ymin=0.0,
            ymax=0.06,
            zorder=0,
        )
        break_transform = blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.text(
            break_x,
            -0.015,
            "//",
            transform=break_transform,
            ha="center",
            va="top",
            fontsize=14,
            color="gray",
            clip_on=False,
        )

    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=3,
        fontsize=10,
    )

    # 情感得分折线同样使用压缩后的横轴，并在各段之间断开
    if "情感得分" in df_plot.columns:
        df_plot["情感得分"] = pd.to_numeric(df_plot["情感得分"], errors="coerce")
        sentiment_idx = df_plot.groupby("period")["情感得分"].mean()
        for seg_idx, (_, seg_indices) in enumerate(segments):
            seg_dates = segments[seg_idx][0]
            seg_sent = sentiment_idx.reindex(seg_dates).to_numpy()
            if seg_idx == 0:
                ax2 = ax1.twinx()
                ax2.plot(
                    x_segments[seg_idx],
                    seg_sent,
                    color="blue",
                    linewidth=2.5,
                    linestyle="--",
                    marker="s",
                    label="公众情感得分均值",
                )
            else:
                ax2.plot(
                    x_segments[seg_idx],
                    seg_sent,
                    color="blue",
                    linewidth=2.5,
                    linestyle="--",
                    marker="s",
                )
        ax2.set_ylabel("情感得分", fontsize=12)
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines_2,
            labels_2,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            frameon=False,
            ncol=1,
            fontsize=10,
        )

    plt.title("公众舆情演化河流图与情感指数综合趋势", fontsize=18, y=1.20)
    fig.tight_layout()
    fig.savefig(output_img, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f" -> ✓ 绘图成功！综合图表已保存至：{output_img}")


plot_theme_river(df_final, output_img=river_output_path)
# 导出政府、媒体、公众的议题分布与情感倾向占比
print("\n【开始导出政府、媒体、公众的议题分布与情感倾向占比】")

# 从 df_final 中筛选三类账号
def normalize_account(val):
    if isinstance(val, str):
        val = val.strip().lower()
        if '公众' in val or 'public' in val:
            return '公众'
        elif '政府' in val or 'government' in val:
            return '政府'
        elif '媒体' in val or 'media' in val:
            return '媒体'
    return None


df_final['账号分类归一'] = df_final['账号分类'].apply(normalize_account)
df_three = df_final[df_final['账号分类归一'].notna()].copy()

if df_three.empty:
    print(" -> 警告：未找到公众、政府、媒体分类的数据，跳过统计导出。")
else:
    # 提取第一个议题（去除括号分数）
    def extract_first_theme(label_str):
        if pd.isna(label_str) or label_str == "无匹配":
            return None
        cleaned = re.sub(r'\([0-9.]+\)', '', str(label_str))
        parts = [p.strip() for p in cleaned.split('|') if p.strip()]
        return parts[0] if parts else None


    df_three['议题'] = df_three['匹配的主题标签'].apply(extract_first_theme)
    df_three = df_three[df_three['议题'].notna()].copy()

    if df_three.empty:
        print(" -> 警告：没有任何数据包含有效议题，无法统计。")
    else:
        # 1. 主题分布 —— 计数 + 占比
        pivot_count = pd.crosstab(df_three['议题'], df_three['账号分类归一'])
        for col in ['公众', '政府', '媒体']:
            if col not in pivot_count.columns:
                pivot_count[col] = 0
        pivot_count = pivot_count[['公众', '政府', '媒体']]  # 固定顺序

        # 计算列总计（每个分类的总议题数）
        col_totals = pivot_count.sum(axis=0)

        # 计算列百分比（避免除零）
        pivot_pct = pivot_count.div(col_totals, axis=1) * 100
        pivot_pct = pivot_pct.round(2)

        # 构建多级列索引的 DataFrame
        multi_cols = pd.MultiIndex.from_product([['公众', '政府', '媒体'], ['计数', '占比%']])
        theme_stat = pd.DataFrame(index=pivot_count.index, columns=multi_cols)

        for cat in ['公众', '政府', '媒体']:
            theme_stat[(cat, '计数')] = pivot_count[cat]
            theme_stat[(cat, '占比%')] = pivot_pct[cat]

        # 添加总计行
        total_row = pd.Series(index=multi_cols, dtype=object)
        for cat in ['公众', '政府', '媒体']:
            total_row[(cat, '计数')] = col_totals[cat]
            total_row[(cat, '占比%')] = 100.0
        theme_stat.loc['总计'] = total_row

        # 2. 情感倾向占比（保持原有逻辑，确认使用“情感得分”列）
        sentiment_vals = pd.to_numeric(df_three['情感得分'], errors='coerce')
        valid_sent = sentiment_vals.dropna()
        if len(valid_sent) == 0:
            print(" -> 警告：情感得分全为空，无法计算情感占比。")
            sentiment_ratio = pd.DataFrame(columns=['分类', '正面占比%', '负面占比%', '中性占比%', '平均情感得分'])
        else:
            def sentiment_label(val):
                if val > 0.0:
                    return '正面'
                elif val < -0.0:
                    return '负面'
                else:
                    return '中性'


            df_three['情感标签'] = df_three['情感得分'].apply(sentiment_label)
            grouped = df_three.groupby('账号分类归一')['情感标签'].value_counts(normalize=True).unstack(
                fill_value=0) * 100
            avg_sent = df_three.groupby('账号分类归一')['情感得分'].mean()
            grouped['平均情感得分'] = avg_sent
            for col in ['正面', '负面', '中性']:
                if col not in grouped.columns:
                    grouped[col] = 0
            grouped = grouped[['正面', '负面', '中性', '平均情感得分']]
            grouped.reset_index(inplace=True)
            grouped.rename(columns={'账号分类归一': '分类'}, inplace=True)
            sentiment_ratio = grouped

        # 导出到 Excel
        output_stat = os.path.join(os.path.dirname(output_path), '主题分布与情感统计.xlsx')
        with pd.ExcelWriter(output_stat, engine='openpyxl') as writer:
            theme_stat.to_excel(writer, sheet_name='主题分布')
            sentiment_ratio.to_excel(writer, sheet_name='情感占比', index=False)

        print(f" -> ✓ 统计导出成功！文件保存至：{output_stat}")
        print("    - 工作表「主题分布」：各账号分类下的议题计数及占比（列百分比）")
        print("    - 工作表「情感占比」：各账号分类下的正面/负面/中性占比及平均情感得分")