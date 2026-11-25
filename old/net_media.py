import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import pandas as pd
import numpy as np
import re
import jieba
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from umap import UMAP
import hdbscan
from bertopic import BERTopic
from bertopic.representation import TextGeneration, KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

# ================= 配置参数 =================
INPUT_CSV = "/root/pmd2/最终合并数据.csv"
OUTPUT_DOC_TOPICS_CSV = "/root/pmd/document_topics.csv"  # 修改了输出路径
OUTPUT_TOPIC_INFO_CSV = "topic_info.csv"

LLM_MODEL_PATH = "/root/autodl-tmp/Qwen3.5-9B"
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"

TOP_N_TOPICS = 1

print("正在读取数据...")
df = pd.read_csv(INPUT_CSV)

df = df[df['source'].isin(['media', 'government'])].copy()

# 确保评论、时间和source都不为空
df.dropna(subset=['评论', '时间', 'source'], inplace=True)
docs = df['评论'].astype(str).tolist()
timestamps = df['时间'].astype(str).tolist()
sources = df['source'].astype(str).tolist()  # 新增：提取 source 字段

print(f"成功提取 {len(docs)} 条有效文本进行聚类。")


# 将你的 seed_topic_list 转化为文字描述，作为 Zero-Shot 的主题列表
zeroshot_topic_list = [
    "产业标准制度建设：中央一号文件, 六部门通知, 定义范围, 不添加防腐剂, 术语分类, 食品安全国家标准, 监管机制, 生产许可, 冷链运输, 地方标准, 团体标准, 生产许可审查, 长三角一体化, 行业准入门槛, 质量安全标准",
    "食品安全质量监管：槽头肉, 315曝光, 添加剂, 保质期, 淋巴肉, 抽检, 处罚, 梅菜扣肉, 食品安全法, 卫生问题, 劣质原料, 过期食材, 非法添加, 监督检查, 风险监测, 食品添加剂使用, 劣质槽头肉, 原料溯源, 变质腐败, 异物, 生产不规范, 卫生不达标",
    "产业发展市场前景：市场规模, 万亿蓝海, 资本融资, 产业园区, 龙头企业, 乡村振兴, 三产融合, 预制菜装备, 冷链物流, 上市公司, 增长预测, 投资, 风口, 产业链, 中央厨房, 产业集群",
    "消费者权益保障：知情权, 选择权, 明示标注, 餐厅告知, 虚假宣传, 菜单标注, 消费者权益, 欺骗, 隐瞒, 价格不匹配, 明码标价, 强制标注, 消费欺诈, 主动公示, 信息透明, 退一赔三",
    "校园预制菜争议与管理：预制菜进校园, 家长担忧, 教育部, 审慎态度, 不宜推广, 校长反对, 学生健康, 辞职送饭, 监管缺失, 未成年人保护, 学校食堂, 营养健康, 中央厨房配餐, 家长委员会, 校园食品安全",
    "预制菜C端消费市场：年夜饭, 预制菜礼盒, 年轻人, 一二线城市, 销量增长, 293元8个菜, 佛跳墙, 半成品, 厨房小白, 年货节, 家庭消费, 电商平台, 消费主力, 自热食品, 盒马, 叮咚买菜, 高端预制菜",
    "西贝预制菜舆论风波：西贝事件, 罗永浩吐槽, 口水战, 公关灾难, 致歉信, 网络黑嘴, 硬刚, 坦诚, 信任危机, 误导消费者, 贾国龙, 开放后厨, 起诉, 集体诉讼, 于东来, 关店事件, 门店关闭"
]

seed_topic_list = [
    ["中央一号文件", "六部门通知", "定义范围", "不添加防腐剂", "术语分类", "食品安全国家标准", "监管机制", "生产许可", "冷链运输", "地方标准",
     "团体标准", "生产许可审查", "长三角一体化", "行业准入门槛", "质量安全标准"],
    ["槽头肉", "315曝光", "添加剂", "保质期", "淋巴肉", "抽检", "处罚", "梅菜扣肉", "食品安全法", "卫生问题", "劣质原料", "过期食材", "非法添加",
     "监督检查", "风险监测", "食品添加剂使用", "劣质槽头肉", "原料溯源", "变质腐败", "异物", "生产不规范", "卫生不达标"],
    ["市场规模", "万亿蓝海", "资本融资", "产业园区", "龙头企业", "乡村振兴", "三产融合", "预制菜装备", "冷链物流", "上市公司", "增长预测", "投资",
     "风口", "产业链", "中央厨房", "产业集群"],
    [ "知情权", "选择权", "明示标注", "餐厅告知", "虚假宣传", "菜单标注", "消费者权益", "欺骗", "隐瞒", "价格不匹配", "明码标价", "强制标注",
      "消费欺诈", "主动公示", "信息透明", "退一赔三"],
    ["预制菜进校园", "家长担忧", "教育部", "审慎态度", "不宜推广", "校长反对", "学生健康", "辞职送饭", "监管缺失", "未成年人保护", "学校食堂",
     "营养健康", "中央厨房配餐", "家长委员会", "校园食品安全"],
    ["年夜饭", "预制菜礼盒", "年轻人", "一二线城市", "销量增长", "293元8个菜", "佛跳墙", "半成品", "厨房小白", "年货节", "家庭消费", "电商平台",
     "消费主力", "自热食品", "盒马", "叮咚买菜", "高端预制菜"],
    ["西贝事件", "罗永浩吐槽", "口水战", "公关灾难", "致歉信", "网络黑嘴", "硬刚", "坦诚", "信任危机", "误导消费者", "贾国龙", "开放后厨",
     "预制菜争议", "起诉", "集体诉讼", "于东来", "关店事件", "门店关闭"]
]

flat_seed_words = [word for topic_list in seed_topic_list for word in topic_list]

for word in flat_seed_words:
    jieba.add_word(word)

def load_stopwords(filepath):
    stopwords = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        print(f"成功加载停用词表，共计 {len(stopwords)} 个停用词。")
    except Exception as e:
        print(f"读取停用词表失败: {e}")
    return stopwords

# 指定你的文件路径
STOP_WORDS = load_stopwords("/root/pmd2/stopwords_full.txt")

def tokenize_zh(text):
    words = jieba.lcut(text)
    result = []
    for w in words:
        w = w.strip()
        if w and re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', w) and w not in STOP_WORDS:
            result.append(w)
    return result


vectorizer_model = CountVectorizer(
    tokenizer=tokenize_zh,
    ngram_range=(1, 4),
    token_pattern=None
)

print("正在加载本地 Qwen 模型...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16
)

model.generation_config.enable_thinking = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=86016,
    return_full_text=False,
    device=0 if torch.cuda.is_available() else -1
)

prompt = """你是一个文本主题分析专家。请根据以下关键词，总结出一个高度概括的主题名称（不超过10个字）。
提示：这些关键词可能与产业标准制度建设、食品安全质量监管、产业发展市场前景、消费者权益保障、校园预制菜争议与管理、预制菜C端消费市场、西贝预制菜舆论风波有关。
关键词：[KEYWORDS]
主题名称："""


# 初始化 MMR 模型，diversity 越高，关键词的词义越分散
mmr_representation = MaximalMarginalRelevance(diversity=0.6)
llm_representation = TextGeneration(text_generator, prompt=prompt)

representation_models = {
    "Main": mmr_representation,
    "LLM": llm_representation
}

ctfidf_model = ClassTfidfTransformer(
    reduce_frequent_words=True,
    seed_words=flat_seed_words,
    seed_multiplier=50.0
)

umap_model = UMAP(
    n_neighbors=10,
    n_components=10,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

n_docs = len(docs)
if n_docs < 5:
    raise ValueError(f"有效文档数量仅为 {n_docs} 条，低于 HDBSCAN 稳定运行阈值。请检查数据过滤条件或扩充数据集。")

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=max(2, min(15, n_docs)),
    min_samples=1,
    metric="euclidean",
    prediction_data=True
)

print("正在进行初始 BERTopic 聚类...")
topic_model = BERTopic(
    embedding_model=EMBEDDING_MODEL,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_models,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=0.95,
    seed_topic_list=seed_topic_list,
    top_n_words=25,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs)

print("正在将离群点分散到距离最近的正常主题中...")
topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")
topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer_model, representation_model=representation_models)

print("正在检查是否有占比超过60%的主题...")
topic_info = topic_model.get_topic_info().copy()
total_docs = len(docs)
topics_arr = np.array(topics)

large_topics = topic_info[topic_info['Count'] / total_docs > 0.6]['Topic'].tolist()

final_topic_info_list = []
current_max_id = max(topic_info['Topic']) if not topic_info.empty else 0


def clean_llm_output(llm_data):
    if isinstance(llm_data, list):
        text = ", ".join(str(x) for x in llm_data)
    else:
        text = str(llm_data)

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'^主题名称[：:]\s*', '', text.strip())
    text = re.sub(r'^[,，。、\s]+', '', text)
    text = re.sub(r'[,，。、\s]+$', '', text)

    return text.strip()


def format_keywords(rep_list):
    if isinstance(rep_list, list):
        cleaned_words = []
        seen = set()
        for word in rep_list:
            word_cleaned = word.replace(" ", "")

            # 去重：如果去空格和删字后的词不为空且没有出现过，则保留
            if word_cleaned and word_cleaned not in seen:
                seen.add(word_cleaned)
                cleaned_words.append(word_cleaned)

        return ", ".join(cleaned_words)
    return str(rep_list)


for _, row in topic_info.iterrows():
    t_id = row['Topic']
    if t_id not in large_topics:
        raw_llm_output = row.get('LLM', row['Name'])

        final_topic_info_list.append({
            "Topic": t_id,
            "Name": clean_llm_output(raw_llm_output),
            "Representation": format_keywords(row['Representation']),
            "Count": row['Count']
        })

# 如果存在超大主题，进行强制拆分
if large_topics:
    for l_topic in large_topics:
        idxes = np.where(topics_arr == l_topic)[0]
        sub_docs = [docs[i] for i in idxes]

        print(f"检测到主题 {l_topic} 占比超过60% ({len(sub_docs)}/{total_docs})，正在进行强制拆分...")

        n_clusters = max(2, int(len(sub_docs) / (total_docs * 0.2)))
        sub_cluster_model = KMeans(n_clusters=n_clusters, random_state=42)

        sub_topic_model = BERTopic(
            embedding_model=EMBEDDING_MODEL,
            umap_model=umap_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_models,
            hdbscan_model=sub_cluster_model,
            top_n_words=10,
            calculate_probabilities=False,
            verbose=False
        )

        sub_topics, _ = sub_topic_model.fit_transform(sub_docs)
        sub_topic_info = sub_topic_model.get_topic_info()

        mapping = {}
        for st in sub_topic_info['Topic'].tolist():
            current_max_id += 1
            mapping[st] = current_max_id

            st_row = sub_topic_info[sub_topic_info['Topic'] == st].iloc[0]
            raw_llm_output = st_row.get('LLM', st_row['Name'])

            final_topic_info_list.append({
                "Topic": mapping[st],
                "Name": clean_llm_output(raw_llm_output),
                "Representation": format_keywords(st_row['Representation']),
                "Count": st_row['Count']
            })

        for i, st in zip(idxes, sub_topics):
            topics_arr[i] = mapping[st]

topics = topics_arr.tolist()

print("正在生成文档-主题概率数据...")
formatted_probs_list = []
original_unsplit_topics = {item["Topic"] for item in final_topic_info_list if
                           item["Topic"] <= (max(topic_info['Topic']) if not topic_info.empty else 0)}

for i in range(len(docs)):
    assigned_topic = topics[i]
    if assigned_topic in original_unsplit_topics and probs is not None and i < len(probs) and isinstance(probs[i], (list, np.ndarray)) and len(probs[i]) > 0:
        prob_array = probs[i]
        top_indices = np.argsort(prob_array)[::-1][:TOP_N_TOPICS]
        topic_prob_dict = {f"Topic_{idx}": round(float(prob_array[idx]), 4) for idx in top_indices}
        formatted_probs_list.append(str(topic_prob_dict))
    else:
        formatted_probs_list.append(str({f"Topic_{assigned_topic}": 1.0}))

# 新增：在此处将 sources 列表加入生成的 DataFrame 中
df_out1 = pd.DataFrame({
    "文本": docs,
    "时间": timestamps,
    "source": sources,  # 新增：保留 source 字段
    "所属主题及概率": formatted_probs_list
})

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_DOC_TOPICS_CSV), exist_ok=True)
df_out1.to_csv(OUTPUT_DOC_TOPICS_CSV, index=False, encoding='utf-8-sig')
print(f"文件1已保存至：{OUTPUT_DOC_TOPICS_CSV}")

print("正在提取主题信息与占比...")
final_topic_df = pd.DataFrame(final_topic_info_list)

final_topic_df["主题占比"] = (final_topic_df["Count"] / total_docs).apply(lambda x: f"{x:.2%}")

df_out2 = final_topic_df[["Topic", "Name", "Representation", "Count", "主题占比"]].copy()

df_out2.rename(columns={
    "Topic": "主题ID",
    "Name": "主题名称",
    "Representation": "关键词",
    "Count": "文本数量"
}, inplace=True)

df_out2.sort_values(by="文本数量", ascending=False, inplace=True)

df_out2.to_csv(OUTPUT_TOPIC_INFO_CSV, index=False, encoding='utf-8-sig')
print(f"文件2已保存至：{OUTPUT_TOPIC_INFO_CSV}")

print("全部任务处理完毕！")