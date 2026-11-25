import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import numpy as np
import pandas as pd
from collections import Counter
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from bertopic import BERTopic
from openai import OpenAI
import umap
import jieba
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation import KeyBERTInspired

# ================= 1. 参数设置区 =================
INPUT_FILE = "/root/autodl-tmp/新余火灾/新余火灾output.xlsx"
OUTPUT_FILE = "/root/autodl-tmp/新余火灾/新余火灾_聚类及命名结果.xlsx"
TARGET_COLUMN = "蒸馏总结"
CATEGORY_COLUMN = "账号分类"

DEEPSEEK_API_KEY = "sk-70411e5d7f9e4c3da4517797dc730c97"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"

MAX_KW_PER_TOPIC = 15
NR_TOPICS = 12

UMAP_N_NEIGHBORS = 8
UMAP_N_COMPONENTS = 12

HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5

STOPWORDS_FILE = "/root/pmd2/stopwords_full.txt"
_STOPWORDS_RAW = """
的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 情感 提及 存在 口吻 强烈 缺乏 使用 辅助 公众 对比 成像 只能 出对 剪开  
好 着 没有 看 这 那 天 年 月 日 吗 啊 哦 嗯 呢 吧 啦 哈 嘛 而且 但是 误解 餐厅 行为 核心 即食 提取 评论 宣传 排烟 餐饮 便利店 言论 老板 公关
所以 因为 如果 虽然 然后 还是 就是 这个 那个 什么 怎么 为什么 可以 无法 提炼 内容 观点 有效 进行 相关 倾向 体现 原理 光机 加工 利用光 技术  
应该 觉得 感觉 其实 已经 现在 时候 知道 自己 大家 可能 还有 以及 评论者 表达 认为 可能 指出 暗示 担忧 态度 抹黑 语气 调侃 信息 强调 说明  
推广 争议 问题 涉及 无关 破圈 回应 事件 转变 提问者 调侃 并视 本土化 客观 直播 技术难题 看法 扩张 对现 隐含 主题 博主 定义 讨论 事实陈述 分类 讽刺
货应 网红带 辛巴为 信息性 狮子狗 行业 认知 经过 简单 属于 用户 一种 养老 公务员 是否 理应 阶层 技能 导致 岗位 困惑 声称 承包商 利用 群体 陈述 揭露
本身 评测 干预 用户 广泛 相对 通过 职业 发展 事实 游戏 交织 愤怒 讽刺 诉求 情绪 质疑 回应 释放 信号 隐性 重塑 威严 日常 缺位 要求 形式主义 滞后 
选择性 强烈 表达 交织 优先于 逐利 威慑 惩罚性 启动 问责 专项 信号 重塑 威严 根基
"""
STOPWORDS: set[str] = set(_STOPWORDS_RAW.split())

try:
    with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
        extra_stops = {line.strip() for line in f if line.strip()}
        STOPWORDS.update(extra_stops)
except FileNotFoundError:
    print(f"警告：未找到 {STOPWORDS_FILE}，将仅使用内置停用词。")


def jieba_tokenize(text: str) -> list[str]:
    tokens = jieba.lcut(str(text))
    return [
        t for t in tokens
        if len(t) >= 2
           and t not in STOPWORDS
           and re.match(r"^[\u4e00-\u9fff]+$", t)
    ]


# ================= 核心处理逻辑封装 =================
def process_category(df_subset, category_name, emb_model, client):
    """
    对特定的数据子集（public 或 其他）进行清洗、去重、聚类和命名
    """
    print(f"\n{'=' * 20} 开始处理分类: 【{category_name}】 {'=' * 20}")

    # --- 1. 清洗：过滤无观点数据 ---
    initial_count = len(df_subset)
    df_subset = df_subset.dropna(subset=[TARGET_COLUMN]).copy()
    df_subset[TARGET_COLUMN] = df_subset[TARGET_COLUMN].astype(str)

    invalid_patterns = "未提及|未涉及|无法提炼|无法概括|无法得出|没有涉及|不涉及"
    mask = ~df_subset[TARGET_COLUMN].str.contains(invalid_patterns, na=False, regex=True).values
    df_filtered = df_subset[mask].reset_index(drop=True)

    filtered_count = len(df_filtered)
    print(f"[{category_name}] 初步清洗：剔除了 {initial_count - filtered_count} 条无效文本。")

    if filtered_count == 0:
        print(f"[{category_name}] 有效文本数量为 0，跳过该分类。")
        return None

    # --- 2. 相似度去重 ---
    texts = df_filtered[TARGET_COLUMN].tolist()
    tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    keep_indices = []
    drop_indices = set()
    n_docs = len(texts)

    for i in range(n_docs):
        if i in drop_indices:
            continue
        keep_indices.append(i)
        if i < n_docs - 1:
            sims = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i + 1:]).flatten()
            duplicates = np.where(sims > 1.0)[0] + i + 1
            drop_indices.update(duplicates.tolist())

    df_filtered = df_filtered.iloc[keep_indices].copy().reset_index(drop=True)
    docs = df_filtered[TARGET_COLUMN].tolist()
    final_count = len(docs)

    print(f"[{category_name}] 去重完成：共剔除 {len(drop_indices)} 条高度重复文本，保留 {final_count} 条。")

    # 防止数据量过少导致 HDBSCAN 报错
    if final_count < HDBSCAN_MIN_CLUSTER_SIZE:
        print(
            f"[{category_name}] 警告：有效文本数量 ({final_count}) 小于 HDBSCAN 最小聚类要求 ({HDBSCAN_MIN_CLUSTER_SIZE})，跳过该分类。")
        return None

    # --- 3. 计算文本向量 ---
    print(f"[{category_name}] 计算文本向量 (Embeddings)...")
    embeddings = emb_model.encode(docs, show_progress_bar=True)

    # --- 4. 模型初始化与聚类 ---
    umap_model = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS, min_dist=0.0, metric="cosine",
                           random_state=42)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
                                    metric="euclidean", cluster_selection_epsilon=0.2, prediction_data=True)
    vectorizer = CountVectorizer(tokenizer=jieba_tokenize, min_df=2, ngram_range=(1, 6))
    representation_models = {
        "Main":
            [
                KeyBERTInspired(top_n_words=MAX_KW_PER_TOPIC, nr_repr_docs=10),
                MaximalMarginalRelevance(diversity=0.5)
            ]
    }
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
    print(f"[{category_name}] 进行 BERTopic 聚类...")
    topic_model = BERTopic(
        embedding_model=emb_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_models,
        min_topic_size=HDBSCAN_MIN_CLUSTER_SIZE,
        ctfidf_model=ctfidf_model,
        nr_topics=NR_TOPICS,
        calculate_probabilities=False,
        verbose=False,  # 避免太多冗余日志
        top_n_words=MAX_KW_PER_TOPIC,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # 消除 -1 (噪声) 主题
    print(f"[{category_name}] 检查并消除 -1 噪声主题...")
    if -1 in topics:
        topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
        topic_model.update_topics(
            docs,
            topics=topics,
            vectorizer_model=vectorizer,
            representation_model=representation_models
        )
        print(f"  [提示] 成功消除 -1 噪声点并重新分配。")
    else:
        print(f"  [提示] 该分类下未发现 -1 噪声文本，跳过噪声消除步骤。")

    # 巨型主题拆分
    print(f"[{category_name}] 检查巨型主题并拆分...")
    total_docs = len(docs)

    try:
        # 获取每个文档属于每个主题的概率分布 (n_docs, n_topics)
        topic_distr, _ = topic_model.approximate_distribution(docs, window=5)
        # 取概率最大的主题索引作为该文档的“概率最大主题”
        prob_topics = np.argmax(topic_distr, axis=1)
        topic_counts = Counter(prob_topics)
        print(f"[{category_name}] 基于概率最大分配计算主题占比。")
    except Exception as e:
        # 如果概率计算失败（如模型不支持），则回退到硬分配
        print(f"[{category_name}] 概率计算失败({e})，回退到使用硬分配主题。")
        topic_counts = Counter(topics)

    current_max_topic = max(topics)
    needs_update = False

    for t, count in topic_counts.items():
        if (count / total_docs) > 0.4:
            needs_update = True
            ratio = count / total_docs
            print(f"  -> 发现巨型主题 {t} (占比 {ratio:.1%})，执行硬拆分...")
            n_splits = max(2, int(np.ceil(ratio / 0.4)))
            idx_list = [i for i, topic_id in enumerate(topics) if topic_id == t]
            sub_embeddings = embeddings[idx_list]

            kmeans = KMeans(n_clusters=n_splits, random_state=42)
            sub_labels = kmeans.fit_predict(sub_embeddings)

            for i, doc_idx in enumerate(idx_list):
                if sub_labels[i] != 0:
                    topics[doc_idx] = current_max_topic + sub_labels[i]
            current_max_topic += (n_splits - 1)

    if needs_update:
        if needs_update:
            topic_model.update_topics(
                docs,
                topics=topics,
                vectorizer_model=vectorizer,
                representation_model=representation_models
            )

    topic_info = topic_model.get_topic_info()
    print(f"[{category_name}] 聚类完毕！共确立 {len(topic_info)} 个主题。")

    # --- 5. DeepSeek API 命名 ---
    print(f"[{category_name}] 调用 DeepSeek API 命名...")
    topic_names = {}
    for index, row in topic_info.iterrows():
        topic_id = row['Topic']
        keywords_list = topic_model.get_topic(topic_id)
        keywords = ", ".join([word for word, score in keywords_list])

        rep_docs = topic_model.get_representative_docs(topic_id)
        if rep_docs and len(rep_docs) > 0:
            rep_docs_text = "\n".join([f"- {doc}" for doc in rep_docs[:15]])
        else:
            manual_docs = [docs[i] for i, tid in enumerate(topics) if tid == topic_id]
            rep_docs_text = "\n".join([f"- {doc}" for doc in manual_docs[:15]])

        prompt = f"""
        你是一个专业的数据分析师。以下是某个特定文本聚类簇的【关键词】和【代表性文本】。
        请为该主题总结一个精准的名称（10~30个字）。

        【要求】：
        1. 忽略通用的情绪描述（如“愤怒与讽刺交织”），必须提取出该主题最核心的具体事件、诉求或事实对象。
        2. 名称必须是一句流畅的话，中间不要有冒号。
        3. 突出这个主题独有的特征，不要使用宽泛的套话。

        【关键词】：{keywords}
        【代表性文本】：
        {rep_docs_text}
        """

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个高度精炼的文本分类命名助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=15,
                temperature=0.3
            )
            topic_name = response.choices[0].message.content.strip()
            topic_names[topic_id] = topic_name
            print(f"  [成功] ID {topic_id} -> {topic_name}")
        except Exception as e:
            print(f"  [失败] ID {topic_id} 报错: {e}")
            topic_names[topic_id] = f"未命名主题_{topic_id}"

    # --- 6. 整理该分类的结果表 ---
    topic_info["DeepSeek命名"] = topic_info["Topic"].map(topic_names)
    topic_info["主题关键词"] = topic_info["Topic"].apply(
        lambda tid: ", ".join([w for w, s in topic_model.get_topic(tid)])
    )

    # 增加账号分类标识列
    topic_info.insert(0, '账号分类', category_name)

    output_cols = ['账号分类', 'Topic', 'Count', 'DeepSeek命名', '主题关键词', 'Name', 'Representation']
    final_info = topic_info[output_cols].rename(columns={
        "Topic": "主题ID",
        "Count": "文本数量",
        "Name": "BERTopic原始命名",
        "Representation": "特征词列表"
    })

    return final_info


def main():
    print(f"正在读取文件: {INPUT_FILE} ...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    if TARGET_COLUMN not in df.columns:
        print(f"列名 '{TARGET_COLUMN}' 不存在于文件中，请检查！")
        return

    if CATEGORY_COLUMN not in df.columns:
        print(f"列名 '{CATEGORY_COLUMN}' 不存在于文件中，请检查！")
        return

    # 全局初始化一次模型与客户端，传递给子函数以节省资源
    print("正在加载公共组件 (Embedding 模型与 DeepSeek 客户端)...")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # ================= 数据拆分 =================
    df_public = df[df[CATEGORY_COLUMN] == 'public'].copy()
    df_other = df[df[CATEGORY_COLUMN] != 'public'].copy()

    print(f"\n数据拆分完毕: \n - public 类别共有 {len(df_public)} 条\n - 其他类别共有 {len(df_other)} 条")

    results = []

    # ================= 处理 public 类 =================
    if not df_public.empty:
        res_public = process_category(df_public, "public", emb_model, client)
        if res_public is not None:
            results.append(res_public)

    # ================= 处理 其他 类 =================
    if not df_other.empty:
        res_other = process_category(df_other, "其他", emb_model, client)
        if res_other is not None:
            results.append(res_other)

    # ================= 合并与保存 =================
    if results:
        final_df = pd.concat(results, ignore_index=True)
        print("\n最终核心主题概览 (部分展示)：")
        print(final_df[['账号分类', '主题ID', '文本数量', 'DeepSeek命名', '主题关键词']].head(10))

        final_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\n全部处理完毕！分治聚类后的合并结果已保存至：{OUTPUT_FILE}")
    else:
        print("\n所有类别均无有效结果产生，未生成输出文件。")


if __name__ == "__main__":
    main()