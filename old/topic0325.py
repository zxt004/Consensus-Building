"""
阶段 2：BERTopic 主题聚类 + 综合演化趋势图（河流图+HHI+情感指数）
"""
from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

import torch
from transformers import pipeline
import transformers

transformers.logging.set_verbosity_error()
from bertopic.representation import TextGeneration
import re
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import jieba
import colorsys
import networkx as nx
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import umap
import hdbscan
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==================== 配置区 ====================
INPUT_CSV = "/root/pmd/processed_data/ai_summarized.csv"
OUTPUT_TOPIC_TABLE = "./processed_data/topic_keyword_table.csv"
STOPWORDS_FILE = "/root/pmd2/stopwords_full.txt"

OUTPUT_GEXF_KW = "./processed_data/keywords_subtopic_graph.gexf"
OUTPUT_STREAM_CHART = "./processed_data/topic_evolution_streamgraph.png"

MAX_KW_PER_TOPIC = 12
MIN_TOPIC_SIZE = 15
NR_TOPICS = 10
COOC_MIN_WEIGHT = 2
SYNONYM_THRESHOLD = 0.75

EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"

USE_PRECOMPUTED_EMBEDDINGS = False
EMBEDDINGS_FILE = "/root/autodl-tmp/opinion_distill.npy"
SENTIMENT_MODEL = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"

UMAP_N_NEIGHBORS = 20
UMAP_N_COMPONENTS = 8
UMAP_MIN_DIST = 0.0

HDBSCAN_MIN_CLUSTER_SIZE = 25
HDBSCAN_MIN_SAMPLES = 6
HDBSCAN_METRIC = "euclidean"

EVENT_LABELS: dict[str, list[str]] = {
    "2022-07": ["罗敏直播预制菜遭质疑"],
    "2022-12": ["格力5000万元成立预制菜装备公司"],
    "2023-02": ["预制菜写入中央一号文件"],
    "2023-09": ["预制菜进校园引热议"],
    "2023-11": ["马云成立新公司「马家厨房」布局预制菜"],
    "2024-03": ["六部门明确预制菜不允许添加防腐剂"],
    "2025-01": ["预制菜年夜饭热销"],
    "2025-09": ["罗永浩西贝预制菜之争"],
    "2026-01": [],
    "2024-03-15": ["315梅菜扣肉预制菜事件"],
    "2025-09-21": ["预制菜要有国标了"],
    "2026-01-16": ["贾国龙和罗永浩的微博账号均被封禁"],
}

_STOPWORDS_RAW = """
的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 情感 提及 存在 口吻 强烈 缺乏 使用 辅助 公众 对比 成像 只能 出对 剪开  
好 着 没有 看 这 那 天 年 月 日 吗 啊 哦 嗯 呢 吧 啦 哈 嘛 而且 但是 误解 餐厅 行为 核心 即食 提取 评论 宣传 排烟 餐饮 便利店 言论 老板 公关
所以 因为 如果 虽然 然后 还是 就是 这个 那个 什么 怎么 为什么 可以 无法 提炼 内容 观点 有效 进行 相关 倾向 体现 原理 光机 加工 利用光 技术  
应该 觉得 感觉 其实 已经 现在 时候 知道 自己 大家 可能 还有 以及 评论者 表达 认为 可能 指出 暗示 担忧 态度 抹黑 语气 调侃 信息 强调 说明  
推广 争议 问题 涉及 无关 破圈 回应 事件 转变 提问者 调侃 并视 本土化 客观 直播 技术难题 看法 扩张 对现 隐含 主题 博主 定义 讨论 事实陈述 分类   
货应 网红带 辛巴为 信息性 狮子狗 行业 认知 经过 简单 属于 用户 一种 养老 公务员 是否 理应 阶层 技能 导致 岗位 困惑 声称 承包商 利用 群体 陈述
本身 评测 干预 用户 广泛 相对 通过 职业 发展 事实 游戏
"""
STOPWORDS: set[str] = set(_STOPWORDS_RAW.split())

if os.path.exists(STOPWORDS_FILE):
    try:
        with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
            extra_stops = {line.strip() for line in f if line.strip()}
            STOPWORDS.update(extra_stops)
        print(f"[*] 成功加载外部停用词表 '{STOPWORDS_FILE}'")
    except Exception as e:
        print(f"[!] 加载停用词表失败: {e}")


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


def jieba_tokenize(text: str) -> list[str]:
    tokens = jieba.lcut(str(text))
    return [
        t for t in tokens
        if len(t) >= 2
           and t not in STOPWORDS
           and re.match(r"^[\u4e00-\u9fff]+$", t)
    ]


def _parse_event_ts(key: str) -> pd.Timestamp:
    if len(key) == 7:
        return pd.Timestamp(key + "-01")
    return pd.Timestamp(key)


_SORTED_EVENTS: list[tuple[pd.Timestamp, str]] = sorted(
    [(_parse_event_ts(k), labels[0] if labels else k) for k, labels in EVENT_LABELS.items()],
    key=lambda x: x[0],
)


def get_event_segment(date) -> str:
    if pd.isna(date):
        return "未知"
    ts = pd.Timestamp(date)
    result = "早期"
    for evt_ts, evt_label in _SORTED_EVENTS:
        if ts >= evt_ts:
            result = evt_label
        else:
            break
    return result


def load_data(csv_path: str) -> pd.DataFrame:
    print("\n[1/9] 加载数据与预计算的Embedding…")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if USE_PRECOMPUTED_EMBEDDINGS and os.path.exists(EMBEDDINGS_FILE):
        try:
            embs = np.load(EMBEDDINGS_FILE)
            if len(embs) == len(df):
                df["_pre_emb"] = list(embs)
                print(f"  ✓ 成功加载预计算词向量: {EMBEDDINGS_FILE} (共 {len(embs)} 条)")
            else:
                print(f"  ! 警告: 预计算词向量长度({len(embs)})与CSV行数({len(df)})不匹配，将回退到运行时计算。")
        except Exception as e:
            print(f"  ! 警告: 加载词向量失败: {e}，将回退到运行时计算。")

    for cand in ("ai_summary", "评论", "comment", "text", "内容"):
        if cand in df.columns:
            df.rename(columns={cand: "ai_summary"}, inplace=True) if cand != "ai_summary" else None
            break
    assert "ai_summary" in df.columns, "未找到摘要列，请检查 CSV 列名"

    # 清洗时间列，过滤“发布于 ”、“编辑于 ”等多余字符，并支持提取到具体时间(HH:MM)
    for tcand in ("时间", "创建时间", "发布时间"):
        if tcand in df.columns:
            # 去除前缀和首尾空格
            clean_time = df[tcand].astype(str).str.replace(r'(?:发布于|编辑于)\s*', '', regex=True).str.strip()
            # 提取时间格式，兼容包含小时和分钟的情况
            time_strs = clean_time.str.extract(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}(?:\s+\d{1,2}:\d{1,2})?)')[0]
            time_strs = time_strs.str.replace("年", "-").str.replace("月", "-").str.replace("日", "")
            df["时间"] = pd.to_datetime(time_strs, errors="coerce")
            break

    if "时间" not in df.columns:
        df["时间"] = pd.NaT

    # 1. 定义无效模式
    invalid_patterns = r"未提及预制菜|未提及|未涉及|无法提炼|无法概括|观点提炼|没有涉及|包含.*预制|并未.*预制|文本.*预制|不涉及|未在.*提及|无法.*提炼|没有.*相关内容"
    mask_invalid = df["ai_summary"].str.contains(invalid_patterns, na=False, regex=True)
    df.loc[mask_invalid, ["评论", "ai_summary"]] = ""

    df["ai_summary"] = df["ai_summary"].fillna("").astype(str)
    mask_not_empty = df["ai_summary"].str.strip().ne("")
    df = df[mask_not_empty].copy()

    df = df.drop_duplicates(subset=["ai_summary"]).copy()

    # 7. 重置索引
    df = df.reset_index(drop=True)

    return df


def preprocess(df: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame, np.ndarray | None]:
    print("\n[2/9] jieba 分词…")
    toks = [" ".join(jieba_tokenize(d)) for d in tqdm(df["ai_summary"], unit="doc")]

    mask = np.array([bool(t.strip()) for t in toks])
    df_v = df.iloc[mask].reset_index(drop=True)
    raw_v = [d for d, m in zip(df["ai_summary"].tolist(), mask) if m]
    tok_v = [t for t, m in zip(toks, mask) if m]

    embs_v = None
    if "_pre_emb" in df_v.columns:
        embs_v = np.stack(df_v["_pre_emb"].values)
        df_v.drop(columns=["_pre_emb"], inplace=True)

    return raw_v, tok_v, df_v, embs_v


def calculate_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """在原有评论/提炼文本上计算情感得分"""
    # 增加的判断：如果 CSV 中已经包含情感得分，直接跳过计算
    if 'sentiment_score' in df.columns:
        print("\n[3/9] 检测到数据中已存在 'sentiment_score' 列，直接跳过情感计算")
        # 顺便检查一下极性列是否存在，防止后续有依赖
        if 'sentiment_polarity' not in df.columns:
            df['sentiment_polarity'] = "未知"
        return df
    print(f"\n[3/9] 情感得分计算 (模型: {SENTIMENT_MODEL})...")
    device = 0 if torch.cuda.is_available() else -1

    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=device)
    except Exception as e:
        print(f"  ! 情感模型加载失败，请检查模型名称或网络连接: {e}")
        print("  ! 将使用默认值 0 填充。")
        df['sentiment_score'] = 0.0
        df['sentiment_polarity'] = "未知"
        return df

    texts = df["ai_summary"].fillna("").astype(str).tolist()
    texts = [t[:500] if len(t) > 0 else "中立" for t in texts]

    print(f"  → 正在批量推理 (共 {len(texts)} 条文本)...")
    results = sentiment_pipeline(texts, batch_size=32, truncation=True, max_length=512)

    sentiments = []
    polarities = []

    for res in results:
        label = str(res['label']).upper()
        score = res['score']

        if label in ['POSITIVE', '正向', 'LABEL_1', '1', 'STAR 4', 'STAR 5']:
            sentiments.append(score)
            polarities.append("正向")
        else:
            sentiments.append(-score)
            polarities.append("负向")

    df['sentiment_score'] = sentiments
    df['sentiment_polarity'] = polarities
    print("  ✓ 情感得分计算完成！")
    return df


def run_bertopic(docs_tok: list[str], docs_raw: list[str], embeddings: np.ndarray = None):
    print("\n[4/9] 构建 BERTopic 模型（自动聚类）…")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        prediction_data=True
    )

    llm_path = "/root/autodl-tmp/Qwen3.5-9B"
    try:
        generator = pipeline("text-generation", model=llm_path, tokenizer=llm_path, device_map="auto", dtype=torch.float16)

        # [修改] 更新 Prompt 限制字数到15字以内
        prompt = """<|im_start|>system
你是一位擅长新闻与舆情领域的专家。<|im_end|>
<|im_start|>user
以下是某个话题簇的代表性文本：
[DOCUMENTS]
该话题的核心关键词有：[KEYWORDS]
请根据以上信息，用极其简练的中文（不超过15个字）高度概括这个话题的核心内容。直接输出最终的文本，不要任何解释。<|im_end|>
<|im_start|>assistant
主题总结："""

        llm_representation = TextGeneration(
            generator,
            prompt=prompt,
            nr_docs=15,
            doc_length=200,
            tokenizer="char",
            # [修改] 缩减 max_new_tokens
            pipeline_kwargs={"max_new_tokens": 20, "truncation": True, "return_full_text": False})

        representation_models = {
            "Main": KeyBERTInspired(top_n_words=MAX_KW_PER_TOPIC, nr_repr_docs=10),
            "topic_name": llm_representation,
        }
    except Exception as e:
        print(f"  ! LLM加载失败，回退到普通 KeyBERT: {e}")
        representation_models = {"Main": KeyBERTInspired(top_n_words=MAX_KW_PER_TOPIC, nr_repr_docs=10)}


    vectorizer = CountVectorizer(tokenizer=jieba_tokenize, min_df=1, max_df=0.95, ngram_range=(1, 2))

    topic_model = BERTopic(
        embedding_model=emb_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_models,
        min_topic_size=HDBSCAN_MIN_CLUSTER_SIZE,
        nr_topics=NR_TOPICS,
        language="multilingual",
        calculate_probabilities=False,
        verbose=False,
        top_n_words=MAX_KW_PER_TOPIC,
    )

    print(f"  → 正在执行 fit_transform…")
    if embeddings is not None:
        print("  → 检测到预加载词向量，跳过文本向量化耗时步骤！")
        topics, _ = topic_model.fit_transform(docs_raw, embeddings=embeddings)
    else:
        print("  → 未使用预存词向量，启动实时嵌入计算...")
        topics, _ = topic_model.fit_transform(docs_raw)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"  → 发现主题数：{n_topics}，噪音文档：{topics.count(-1)}")
    return topic_model, topics, emb_model


def extract_representations(topic_model: BERTopic, topics: list[int]) -> tuple[
    dict[int, str], dict[int, list[tuple[str, float]]]]:
    print("\n[5/9] 提取主题名称和关键词…")
    valid_tids = sorted(t for t in set(topics) if t >= 0)
    topic_names: dict[int, str] = {}
    aspect_store = getattr(topic_model, "topic_aspects_", {})

    for tid in valid_tids:
        name = f"Topic_{tid}"
        if "topic_name" in aspect_store:
            llm_output = aspect_store["topic_name"].get(tid, [])
            if llm_output:
                raw_text = llm_output[0][0] if isinstance(llm_output[0], tuple) else str(llm_output[0])
                name = raw_text.split("主题名称：")[-1].strip().split('\n')[0] if "主题名称：" in raw_text else \
                    raw_text.strip().split('\n')[0]

        name = name.replace("摘要：", "").replace('"', '').replace("'", "").strip() if name else f"Topic_{tid}"
        # 强制截断至 15 字以内，彻底规避过长标题
        name = name[:15]
        topic_names[tid] = name

    topic_kw_raw: dict[int, list[tuple[str, float]]] = {}
    for tid in valid_tids:
        topic_kw_raw[tid] = aspect_store.get("keywords", {}).get(tid, []) or (topic_model.get_topic(tid) or [])
    return topic_names, topic_kw_raw


def merge_synonyms_per_topic(topic_kw_raw: dict[int, list[tuple[str, float]]], emb_model: SentenceTransformer,
                             sim_threshold: float = 0.85) -> dict[int, list[tuple[str, float]]]:
    print(f"\n[6/9] 在主题内部合并近义词…")
    merged_topic_kw = {}
    for tid, kw_list in tqdm(topic_kw_raw.items(), desc="合并同义词"):
        if not kw_list:
            merged_topic_kw[tid] = []
            continue
        kw_list = sorted(kw_list, key=lambda x: x[1], reverse=True)
        words, scores = [kw for kw, _ in kw_list], [score for _, score in kw_list]
        embeddings = emb_model.encode(words, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)

        to_keep_idx = set(range(len(words)))
        for i in range(len(words)):
            if i not in to_keep_idx: continue
            for j in range(i + 1, len(words)):
                if j in to_keep_idx and sim_matrix[i, j] >= sim_threshold:
                    to_keep_idx.remove(j)
        merged_topic_kw[tid] = [(words[i], scores[i]) for i in sorted(list(to_keep_idx))]
    return merged_topic_kw


def deduplicate_keywords(topic_kw_raw: dict[int, list[tuple[str, float]]]) -> tuple[
    dict[int, list[str]], dict[str, str]]:
    print("\n[7/9] 全局关键词去重…")
    kw_best: dict[str, tuple[int, float, str]] = {}
    for tid, kw_list in topic_kw_raw.items():
        for kw, score in kw_list:
            kw_clean = kw.replace(" ", "").strip()
            if kw_clean and (kw_clean not in kw_best or score > kw_best[kw_clean][1]):
                kw_best[kw_clean] = (tid, score, kw)

    topic_final: dict[int, list[str]] = defaultdict(list)
    kw_to_orig: dict[str, str] = {}
    for kw_clean, (tid, _score, kw_orig) in sorted(kw_best.items(), key=lambda x: -x[1][1]):
        if len(topic_final[tid]) < MAX_KW_PER_TOPIC:
            topic_final[tid].append(kw_clean)
            kw_to_orig[kw_clean] = kw_orig
    return dict(topic_final), kw_to_orig


def compute_keyword_metadata(docs_tok: list[str], df_valid: pd.DataFrame, topic_final: dict[int, list[str]],
                             kw_to_orig: dict[str, str]) -> tuple[dict[str, pd.Timestamp | None], dict[str, set[int]]]:
    print("\n[8/9] 扫描关键词数据与共现统计…")
    all_kws: set[str] = {kw for kws in topic_final.values() for kw in kws}
    kw_first_time: dict[str, pd.Timestamp | None] = {kw: None for kw in all_kws}
    kw_doc_indices: dict[str, set[int]] = defaultdict(set)
    times = df_valid["时间"].tolist()

    for idx, doc in enumerate(tqdm(docs_tok, unit="doc")):
        doc_date = times[idx] if idx < len(times) else pd.NaT
        valid_date = not pd.isna(doc_date)
        padded_doc = f" {doc} "

        for kw_clean in all_kws:
            padded_kw = f" {kw_to_orig[kw_clean]} "
            if padded_kw in padded_doc:
                kw_doc_indices[kw_clean].add(idx)
                if valid_date:
                    cur = kw_first_time[kw_clean]
                    if cur is None or doc_date < cur:
                        kw_first_time[kw_clean] = doc_date
    return kw_first_time, kw_doc_indices


def build_cooccurrence(kw_doc_indices: dict[str, set[int]]) -> dict[tuple[str, str], int]:
    doc_to_kws: dict[int, list[str]] = defaultdict(list)
    for kw, doc_ids in kw_doc_indices.items():
        for did in doc_ids:
            doc_to_kws[did].append(kw)

    cooc: dict[tuple[str, str], int] = defaultdict(int)
    for kws_in_doc in doc_to_kws.values():
        if len(kws_in_doc) < 2: continue
        for kw1, kw2 in combinations(sorted(kws_in_doc), 2):
            cooc[(kw1, kw2)] += 1

    return {pair: w for pair, w in cooc.items() if w >= COOC_MIN_WEIGHT}


def export_keyword_graph(
        topic_final: dict[int, list[str]], topic_names: dict[int, str],
        kw_first_time: dict[str, pd.Timestamp | None], cooc: dict[tuple[str, str], int], output_gexf: str
) -> nx.Graph:
    kw_to_tid = {kw: tid for tid, kws in topic_final.items() for kw in kws}
    valid_tids = list(topic_final.keys())
    topic_colors = {tid: get_topic_color(i, len(valid_tids)) for i, tid in enumerate(valid_tids)}

    G = nx.Graph()
    for kw, tid in kw_to_tid.items():
        first_ts = kw_first_time.get(kw)
        G.add_node(
            kw, label=kw, topic_id=int(tid), topic_name=topic_names.get(tid, f"Topic_{tid}"),
            first_appearance=first_ts.strftime("%Y-%m-%d") if first_ts else "未知",
            event_segment=get_event_segment(first_ts), color=topic_colors[tid], viz_color=topic_colors[tid],
        )

    for (kw1, kw2), weight in cooc.items():
        if kw1 in G and kw2 in G:
            G.add_edge(kw1, kw2, weight=float(weight))

    Path(output_gexf).parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, output_gexf)
    print(f"  ✓ 关键词 GEXF 已保存：{output_gexf}")
    return G


def save_topic_table(topic_final: dict[int, list[str]], topic_names: dict[int, str],
                     topics_per_doc: list[int], output_csv: str):
    rows = []
    for tid, kws in topic_final.items():
        doc_count = sum(1 for t in topics_per_doc if t == tid)
        rows.append({
            "topic_id": tid,
            "topic_name": topic_names.get(tid, f"Topic_{tid}"),
            "keywords": "、".join(kws),
            "keyword_cnt": len(kws),
            "doc_count": doc_count,
        })
    tbl = pd.DataFrame(rows).sort_values("doc_count", ascending=False)
    tbl.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"  ✓ 主题-关键词表：{output_csv}")


def plot_comprehensive_chart(df: pd.DataFrame, topics: list[int], topic_names: dict[int, str], output_img: str):
    print("\n[9/9] 绘制河流图、HHI指数与情感指数综合图（在同一画框内）…")

    from matplotlib import font_manager
    font_path = "/root/pmd/SIMSUN.TTC"

    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        custom_font = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = custom_font.get_name()
    else:
        print(f"  ! 未找到指定的字体文件 {font_path}，将回退到默认备选字体。")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'sans-serif']

    plt.rcParams['axes.unicode_minus'] = False

    df = df.copy()
    df['topic'] = topics

    if 'sentiment_score' not in df.columns:
        print("  ! 未检测到 'sentiment_score' 列，将情感值设为 0。")
        df['sentiment_score'] = 0.0

    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0)

    df['period'] = df['时间'].dt.to_period('M').dt.to_timestamp()
    df_plot = df.dropna(subset=['period', 'topic'])

    topic_counts = pd.crosstab(df_plot['period'], df_plot['topic'])
    if -1 in topic_counts.columns:
        topic_counts.drop(columns=[-1], inplace=True)

    dates = topic_counts.index
    if len(dates) == 0:
        print("  ! 数据时间轴为空，无法绘制图表。")
        return

    # [修改] 使用对数缩放 (np.log1p) 替代开根号，进一步压缩极端峰值并提振早期低谷数据
    y = [np.log1p(topic_counts[c].values) for c in topic_counts.columns]
    colors = [get_topic_color(int(c), len(topic_counts.columns)) for c in topic_counts.columns]

    props = topic_counts.div(topic_counts.sum(axis=1), axis=0).fillna(0)
    hhi = (props ** 2).sum(axis=1)

    sentiment_idx = df_plot.groupby('period')['sentiment_score'].mean()

    # [修改] 稍微加高图表画幅，给顶部留空间
    fig, ax1 = plt.subplots(figsize=(14, 7))

    topic_labels = [topic_names.get(c, f"Topic_{c}") for c in topic_counts.columns]
    ax1.stackplot(dates, y, labels=topic_labels, baseline='sym', colors=colors, alpha=0.65)

    ax1.set_xlabel('时间', fontsize=12)
    # [修改] 提示文案变更为对数缩放
    ax1.set_ylabel('单月话题规模分布', fontsize=12, color='gray')
    ax1.set_yticklabels([])
    ax1.tick_params(axis='y', which='both', length=0)
    ax1.tick_params(axis='y', labelcolor='gray')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    # [修改] 将图例锚点设置在坐标轴外正上方（y=1.02），避免侵占绘图区
    ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), frameon=False, ncol=4, fontsize=10)

    ax2 = ax1.twinx()

    line1 = ax2.plot(dates, hhi.reindex(dates).values, color='black', linewidth=2.5, marker='o', label='HHI 集中度指数')
    line2 = ax2.plot(dates, sentiment_idx.reindex(dates).values, color='red', linewidth=2.5, linestyle='--', marker='s', label='情感均值指数')

    ax2.set_ylabel('HHI 指数 / 情感指数', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # [修改] 右侧图例也同样放置到坐标系外部上方
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_2, labels_2, loc='lower right', bbox_to_anchor=(1.0, 1.02), frameon=False, ncol=2, fontsize=10)

    # [修改] 强行抬高标题（使用 y=1.2）防止与外部图例重叠
    plt.title('舆情演化河流图与 HHI、情感指数综合趋势', fontsize=18, y=1.2)

    # [修改] 预留整体画布顶部15%作为绝对保护区，防止裁减标题
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    Path(output_img).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img, dpi=300)
    print(f"  ✓ 综合图表已保存：{output_img}")
    plt.close()


def main():
    print("单路 BERTopic + 综合演化河流趋势导出")

    df = load_data(INPUT_CSV)
    raw_v, tok_v, df_valid, embs_v = preprocess(df)

    df_valid = calculate_sentiment_scores(df_valid)

    topic_model, topics, emb_model = run_bertopic(tok_v, raw_v, embeddings=embs_v)
    topic_names, topic_kw_raw = extract_representations(topic_model, topics)

    topic_kw_raw = merge_synonyms_per_topic(topic_kw_raw, emb_model, sim_threshold=SYNONYM_THRESHOLD)
    topic_final, kw_to_orig = deduplicate_keywords(topic_kw_raw)

    print("\n[*] 开始执行主题过滤...")
    forbidden_words = ["预制话题", "预制逻辑"]
    tids_to_remove = set()

    for tid, kws in topic_final.items():
        for kw in kws:
            # 只要该主题的任意一个关键词包含这些词，就标记为需要删除
            if any(fw in kw for fw in forbidden_words):
                tids_to_remove.add(tid)
                break

    if tids_to_remove:
        print(f"  → 发现并过滤掉 {len(tids_to_remove)} 个包含违禁词的主题: {tids_to_remove}")
        for tid in tids_to_remove:
            topic_final.pop(tid, None)
            topic_names.pop(tid, None)

        # 将被过滤掉的主题对应的文档标签改为 -1 (归为噪音，不在河流图中显示)
        topics = [-1 if t in tids_to_remove else t for t in topics]
    else:
        print("  → 未发现需要过滤的主题。")

    # custom_names = {
    #     0: "预制菜定价与接受度争议",
    #     1: "西贝预制菜价格争议",
    #     2: "对预制菜的全方面反对",
    #     3: "预制菜食品安全隐患讨论",
    #     5: "光去刺技术引发辐射担忧",
    #     6: "隋坡探店预制菜讨论",
    # }
    # topic_names.update(custom_names)


    kw_first_time, kw_doc_indices = compute_keyword_metadata(tok_v, df_valid, topic_final, kw_to_orig)
    cooc = build_cooccurrence(kw_doc_indices)

    print("\n" + "=" * 50)
    print("【第一层：话题 (Topics) 详情】")
    for tid, name in topic_names.items():
        kws_with_events = []
        for kw in topic_final.get(tid, [])[:15]:
            first_ts = kw_first_time.get(kw)
            event_seg = get_event_segment(first_ts)
            kws_with_events.append(f"{kw}({event_seg})")
        kws_str = "、".join(kws_with_events)
        print(f"  - [Topic {tid}] {name} | 关键词: {kws_str}")
    print("=" * 50 + "\n")

    export_keyword_graph(topic_final, topic_names, kw_first_time, cooc, OUTPUT_GEXF_KW)
    save_topic_table(topic_final, topic_names, topics, OUTPUT_TOPIC_TABLE)

    plot_comprehensive_chart(df_valid, topics, topic_names, OUTPUT_STREAM_CHART)


if __name__ == "__main__":
    main()