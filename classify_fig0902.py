from __future__ import annotations
import os
import re
import random
import colorsys
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import font_manager
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# 设置镜像（可选）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

# ------------------ 字体配置（保留但可能不需要绘图） ------------------
font_path = "/root/pmd/SIMSUN.TTC"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ------------------ 输入输出路径 ------------------
INPUT_CSV = '/root/autodl-tmp/预制菜/标注结果0620.csv'
OUTPUT_EXCEL = '/root/autodl-tmp/预制菜/媒体政府_主题与情感统计.xlsx'  # 最终统计结果

# ------------------ 定义 other_themes（媒体/政府相关议题） ------------------
other_themes = {
    "西贝预制菜舆论风波": {
        "description": "围绕西贝创始人贾国龙相关言论所引发的舆论争议展开回应与报道，关注企业家表态与公众反应之间的张力及其对品牌声誉的影响",
        "keywords": ["西贝", "贾国龙", "西贝莜面村", "西贝预制菜", "预制菜争议", "企业家言论", "舆论风波", "品牌声誉", "公关危机", "网友质疑", "西贝回应", "西贝创始人", "预制菜言论", "西贝口碑", "热搜争议", "消费者质疑西贝"]
    },
    "预制菜产业发展与规范": {
        "description": "从宏观层面关注预制菜行业的规模扩张、产业链布局及政策监管框架的建立，强调在发展中逐步完善标准体系与行业规范",
        "keywords": ["预制菜产业", "预制菜行业", "市场规模", "产业链", "中央厨房", "标准化生产", "冷链物流", "政策监管", "国家标准", "行业标准", "地方标准", "产业园区", "高质量发展", "B端餐饮", "C端零售", "预制菜企业", "食品安全监管", "行业规范", "政策支持", "预制菜发展"]
    },
    "校园预制菜争议与管理": {
        "description": "聚焦预制菜进入学校食堂所引发的家长担忧，报道地方叫停措施、教育管理部门的回应及校园供餐安全标准的讨论",
        "keywords": ["校园预制菜", "预制菜进校园", "学校食堂", "学生餐", "营养餐", "家长担忧", "家长抵制", "地方叫停", "教育局回应", "校园食品安全", "供餐安全", "学生营养健康", "中央厨房配餐", "集体用餐配送", "学校配餐", "中小学食堂", "家委会", "预制菜叫停", "校园供餐标准", "预制菜招标"]
    },
    "消费者体验与权益诉求": {
        "description": "关注消费者对预制菜产品品质、性价比及知情权的主张，报道相关投诉与维权事件",
        "keywords": ["消费者", "知情权", "预制菜品质", "口感", "性价比", "投诉", "维权", "消费者权益", "标签标识", "虚假宣传", "料理包", "复热食品", "口感下降", "价格偏高", "消费信任", "差评", "退货", "明示预制菜", "消费者质疑", "消费体验"]
    },
    "预制菜营销伦理争议": {
        "description": "围绕餐饮品牌在宣传中是否夸大预制菜的正向属性、是否模糊现制与预制的界限展开讨论，指向营销行为的伦理边界与监管空间",
        "keywords": ["营销伦理", "虚假宣传", "现制", "预制", "模糊界限", "夸大宣传", "广告法", "误导消费者", "餐厅使用预制菜", "明示义务", "伦理争议", "监管空白", "料理包争议", "网红餐厅", "外卖预制菜", "现场烹饪", "锅气", "宣传与事实不符", "商业伦理", "预制菜广告"]
    }
}

# ------------------ 加载 BERT 模型用于主题分类 ------------------
model_name = 'shibing624/text2vec-base-chinese'
print(f"正在加载主题分类模型 {model_name}...")
model = SentenceTransformer(model_name)

# ------------------ 构建主题向量 ------------------
def get_theme_vectors(themes_dict):
    theme_names = []
    theme_super_texts = []
    for name, content in themes_dict.items():
        theme_names.append(name)
        desc = content["description"]
        kw_str = "、".join(content["keywords"])
        super_text = f"议题名称：{name}。核心定义：{desc} 高频关键词包含：{kw_str}。"
        theme_super_texts.append(super_text)
    theme_vectors = model.encode(theme_super_texts)
    return theme_names, np.array(theme_vectors)

print("正在构建 other_themes 特征向量...")
other_names, other_vectors = get_theme_vectors(other_themes)

# ------------------ 情感模型相关（完全复刻 AIProcess0619） ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print("正在加载本地情感分析模型 IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment ...")
LOCAL_MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_NAME)
sentiment_model.to(device)
sentiment_model.eval()

id2label = sentiment_model.config.id2label
pos_idx = None
neg_idx = None
neutral_idx = None
for idx, label in id2label.items():
    label_lower = label.lower()
    if 'positive' in label_lower or 'pos' in label_lower or '积极' in label_lower:
        pos_idx = idx
    elif 'negative' in label_lower or 'neg' in label_lower or '消极' in label_lower:
        neg_idx = idx
    elif 'neutral' in label_lower or 'neu' in label_lower or '中性' in label_lower:
        neutral_idx = idx
if pos_idx is None or neg_idx is None:
    if len(id2label) == 3:
        neg_idx, neutral_idx, pos_idx = 0, 1, 2
    else:
        pos_idx, neg_idx = 1, 0
print(f"情感模型加载完成，积极索引: {pos_idx}, 消极索引: {neg_idx}")

def get_local_sentiment_batch(texts, batch_size=128, weighted=False):
    """
    批量计算情感得分，复用 AIProcess0619 的加权逻辑。
    weighted=True 时使用针对 media/government 的权重（正权重6.0，负权重0.5）
    """
    model = sentiment_model
    model.to(device)
    model.eval()

    if weighted:
        NEUTRAL_TO_NEGATIVE_RATIO = 0
        NEGATIVE_WEIGHT = 0.5
        POSITIVE_WEIGHT = 6.0
        GLOBAL_NEGATIVE_BIAS = 0
    else:
        NEUTRAL_TO_NEGATIVE_RATIO = 0
        NEGATIVE_WEIGHT = 1.0
        POSITIVE_WEIGHT = 1.0
        GLOBAL_NEGATIVE_BIAS = 0

    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="本地批量情感分析"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            p_pos = probs[:, pos_idx]
            p_neg = probs[:, neg_idx]
            if neutral_idx is not None:
                p_neu = probs[:, neutral_idx]
            else:
                p_neu = torch.zeros_like(p_pos)

            score = POSITIVE_WEIGHT * p_pos - NEGATIVE_WEIGHT * (p_neg + NEUTRAL_TO_NEGATIVE_RATIO * p_neu)
            score = score - GLOBAL_NEGATIVE_BIAS
            score = torch.clamp(score, -1.0, 1.0)

            batch_scores = score.cpu().numpy().tolist()
            scores.extend(round(float(s), 3) for s in batch_scores)
    return scores

# ------------------ 读取 CSV 并筛选 ------------------
print(f"\n正在读取 {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)

# 检查必要列
required_cols = ['source', 'text']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV 中缺少必需列: {col}")

# 筛选 source 为 media 或 government
df_filtered = df[df['source'].str.lower().isin(['media', 'government'])].copy()
print(f"筛选后共 {len(df_filtered)} 条记录（media/government）")

if df_filtered.empty:
    print("没有符合条件的记录，程序退出。")
    exit()

# ------------------ 主题分类（多标签） ------------------
THRESHOLD = 0.75
text_column = 'text'

print("正在对文本进行主题多标签分类...")
texts = df_filtered[text_column].fillna("").astype(str).tolist()
comment_embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
similarities = cosine_similarity(comment_embeddings, other_vectors)

labels = []
for sim_array in similarities:
    valid_matches = [(i, sim_array[i]) for i in range(len(other_names)) if sim_array[i] > THRESHOLD]
    valid_matches.sort(key=lambda x: x[1], reverse=True)
    top_3 = valid_matches[:3]
    matched = [f"{other_names[i]}({score:.2f})" for i, score in top_3]
    if not matched:
        max_idx = np.argmax(sim_array)
        if sim_array[max_idx] > (THRESHOLD - 0.1):
            matched = [f"{other_names[max_idx]}({sim_array[max_idx]:.2f})"]
    labels.append(" | ".join(matched) if matched else "无匹配")

df_filtered['匹配的主题标签'] = labels

# ------------------ 情感得分计算（使用 weighted=True，因为媒体/政府） ------------------
print("正在计算情感得分（加权模式，weighted=True）...")
sentiment_texts = [str(t)[:512] for t in texts]  # 截断
sentiment_scores = get_local_sentiment_batch(sentiment_texts, batch_size=128, weighted=True)
df_filtered['情感得分'] = sentiment_scores

# 对情感得分添加随机偏置（仿 AIProcess0619 的边界处理）
print("正在为情感得分添加随机偏置...")
max_bias_attempts = 10
for idx in df_filtered.index:
    score = df_filtered.loc[idx, '情感得分']
    if score is None or pd.isna(score):
        continue
    current_score = float(score)
    attempt = 0
    while attempt < max_bias_attempts:
        if random.random() < 0.75:
            bias = random.uniform(0, 0.1)
        else:
            bias = -random.uniform(0, 0.1)
        new_score = current_score + bias
        new_score = max(-1.0, min(1.0, new_score))
        if new_score != -1.0 and new_score != 1.0:
            df_filtered.loc[idx, '情感得分'] = round(new_score, 4)
            break
        current_score = new_score
        attempt += 1
    if attempt == max_bias_attempts:
        if current_score == 1.0:
            final_score = 1.0 - random.uniform(0.001, 0.01)
        elif current_score == -1.0:
            final_score = -1.0 + random.uniform(0.001, 0.01)
        else:
            final_score = current_score
        df_filtered.loc[idx, '情感得分'] = round(final_score, 4)

# ------------------ 统计主题分布和情感占比（复用原 classify 的统计代码） ------------------
print("\n开始统计主题分布和情感占比...")

def extract_first_theme(label_str):
    if pd.isna(label_str) or label_str == "无匹配":
        return None
    cleaned = re.sub(r'\([0-9.]+\)', '', str(label_str))
    parts = [p.strip() for p in cleaned.split('|') if p.strip()]
    return parts[0] if parts else None

df_filtered['议题'] = df_filtered['匹配的主题标签'].apply(extract_first_theme)
df_stat = df_filtered[df_filtered['议题'].notna()].copy()

if df_stat.empty:
    print("警告：没有任何数据包含有效议题，无法统计。")
else:
    # 主题分布计数与占比（按 source 分组）
    pivot_count = pd.crosstab(df_stat['议题'], df_stat['source'])
    # 确保列顺序 media, government
    for col in ['media', 'government']:
        if col not in pivot_count.columns:
            pivot_count[col] = 0
    pivot_count = pivot_count[['media', 'government']]
    col_totals = pivot_count.sum(axis=0)
    pivot_pct = pivot_count.div(col_totals, axis=1) * 100
    pivot_pct = pivot_pct.round(2)

    multi_cols = pd.MultiIndex.from_product([['media', 'government'], ['计数', '占比%']])
    theme_stat = pd.DataFrame(index=pivot_count.index, columns=multi_cols)
    for cat in ['media', 'government']:
        theme_stat[(cat, '计数')] = pivot_count[cat]
        theme_stat[(cat, '占比%')] = pivot_pct[cat]
    # 总计行
    total_row = pd.Series(index=multi_cols, dtype=object)
    for cat in ['media', 'government']:
        total_row[(cat, '计数')] = col_totals[cat]
        total_row[(cat, '占比%')] = 100.0
    theme_stat.loc['总计'] = total_row

    # 情感占比（按 source 分组）
    def sentiment_label(val):
        if val > 0.0:
            return '正面'
        elif val < -0.0:
            return '负面'
        else:
            return '中性'

    df_stat['情感标签'] = df_stat['情感得分'].apply(sentiment_label)
    grouped = df_stat.groupby('source')['情感标签'].value_counts(normalize=True).unstack(fill_value=0) * 100
    avg_sent = df_stat.groupby('source')['情感得分'].mean()
    grouped['平均情感得分'] = avg_sent
    for col in ['正面', '负面', '中性']:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped = grouped[['正面', '负面', '中性', '平均情感得分']]
    grouped.reset_index(inplace=True)
    grouped.rename(columns={'source': '分类'}, inplace=True)
    sentiment_ratio = grouped

    # 导出到 Excel
    os.makedirs(os.path.dirname(OUTPUT_EXCEL) or '.', exist_ok=True)
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        theme_stat.to_excel(writer, sheet_name='主题分布')
        sentiment_ratio.to_excel(writer, sheet_name='情感占比', index=False)

    print(f"\n统计导出成功！文件保存至：{OUTPUT_EXCEL}")
    print("    - 工作表「主题分布」：媒体/政府下的议题计数及占比（列百分比）")
    print("    - 工作表「情感占比」：媒体/政府下的正面/负面/中性占比及平均情感得分")