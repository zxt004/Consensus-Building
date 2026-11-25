import os
import random

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import re
import time
from transformers import pipeline
import torch
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区 =================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "密钥")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 新数据源：一个 Excel 文件的两个 sheet
INPUT_FILE = "/root/autodl-tmp/预制菜/预制菜.xlsx"  # 修改为你的新文件路径
OUTPUT_FILE = "/root/autodl-tmp/预制菜/预制菜output.xlsx"

PUBLIC_SHEET_NAME = "公众"            # 第一个 sheet 名称
MEDIA_GOV_SHEET_NAME = "媒体政府"    # 第二个 sheet 名称

# 内容列名
CONTENT_COLUMN = "全文内容"

# 已有的账号分类列名
ACCOUNT_COLUMN = "账号分类"

# 中间列名定义
SCORE_COLUMN = "全文内容"
RAW_CONTENT_COLUMN = "原始内容"      # 清洗前的原始文本
DISTILLED_COLUMN = "蒸馏总结"        # 蒸馏后的总结文本
SOURCE_COLUMN = "数据来源"

PUBLIC_SOURCE = "公众"
MEDIA_GOV_SOURCE = "媒体政府"

USE_LOCAL_SENTIMENT = True         # 是否使用本地情感模型，False 则调用 DeepSeek API

# 提示词定义
SYSTEM_PROMPT = (
    "你是一个社会学数据编码员。你的任务是阅读用户关于“新余火灾”的评论回答、官方新闻或相关讨论，精准提炼其中关于“新余火灾”的核心观点、政策导向和底层逻辑，用于后续的主题聚类。\n"
    "【严格输出规则】：\n"
    "1. 【极其重要】仅提取评论中与新余火灾相关的部分，\n"
    "    包括但不限于：与新余火灾直接相关的段落，与新余火灾间接相关的段落，与新余火灾相关的事件，与新余火灾相关的人，以及对与新余火灾相关的人或事件的评估或态度。\n"
    "    如果是官方新闻和文件发表，提取出其中的有关新余火灾的内容，解析其对于新余火灾的政策导向、价值倾向和隐性立场。\n"
    "    有些评论内容可能包含隐含语义，请你参考与新余火灾的有关事件进行提炼，比如2024年TGA评奖事件。\n"
    "2.[禁止行为与元词汇]【极其重要】：绝对禁止出现“引述”、“转发”、“报道”、“指出”、“强调”、“该微博”、“该文件”、“官方发布”等描述动作或来源的词汇。同时绝对禁止出现“认为”、“觉得”、“总结来说”、“核心立场”等说明性词汇。\n"
    "3.[强制起手式]：第一句话必须直接以核心观点的动词、名词或形容词开头，绝对禁止以主语（如“用户”、“该评论者”、“该微博”、“该报道”、“该通知”）开头。\n"
    "4.[高密度语义]：保留且仅保留社会学和经济学锚点词，忽略情绪发泄、个人故事、无关细节。【极其重要】禁止使用套话，更禁止重复说套话。\n"
    "5.[无观点处理] 【重要】：如果文本内容完全没有关于新余火灾的实质性论点，请仅输出三个字：“无观点”。\n"
    "6.[格式限制]：准确提炼情感倾向（支持/反对/中立/担忧/讽刺等）但是情绪词最多只能用一个词描述，把重点放在【客观事实】和【具体诉求】上。剔除语气词。必须是一段连贯的纯文本（字数控制在50-300字）。禁止使用1、2、3等序号分点！\n"
    "7.[禁止全局主题词]：已知提取出的内容即为该事物的描述，提取结果中绝对禁止出现“新余火灾”几个字！请直接描述其属性。\n"
    "8.[最终输出] 【极其重要】：直接输出最终中文文本！仅提供结果，禁止输出任何思考过程！禁止包含任何英文字符！从第一行开始就是最终总结，不要有任何前缀！\n\n"
    "以下是两条不同侧重点的示例输出，请体会其连贯的逻辑陈述，【禁止照抄其特定句式】:\n"
     "[正确输出范例一（针对个人评论）]：\n"
    "批判违规动火作业与冷库改建中监管失序及责任虚置，揭露防盗窗封堵逃生通道加剧伤亡的结构性隐患，主张以全链条安全责任制和常态化隐患排查替代运动式整改，情感倾向表现为对基层形式主义治理的愤怒与对生命权保障机制缺位的忧惧交织。\n"
    "[正确输出范例二（针对官方文件/新闻）]：\n"
    "驱动安全生产治理从运动式排查向常态化全链条责任闭环转型，隐性政策导向锚定基层应急能力与属地监管问责，规避事故追责中避重就轻的象征性处罚，依托技术化监测与网格化排查构建风险预警机制，同时警惕资本逐利挤压公共安全投入及违规改建对建筑防火规范的结构性破坏。\n"
)

USER_TEMPLATE = (
    "评论原文：\n"
    "'{comment}''\n\n"
    "请直接输出中文总结（仅最终结果，不要 Thinking Process 等思考过程）："
)

# 情感分析提示词（保持原样）
SENTIMENT_SYSTEM_PROMPT = (
    "你是资深中文文本情感分析专家，专长于**连续情感回归**，而非定性分类。\n"
    "输入一段文本后，请将其情感倾向映射到 (-1.000, 0) 或 (0, 1.000) 的连续实数轴上，"
    "**严禁输出 0.000、-1.000 或 1.000**，必须输出三位小数。\n\n"

    "核心评分原则：\n"
    "1. 整体语义优先，转折句以后半句为主（如“虽然……但是……”）；\n"
    "2. 充分考虑否定、反问、反讽、阴阳怪气对极性的反转与削弱；\n"
    "3. 正负混杂时，严格按**净情绪强度**打分，而非选取极端值。\n\n"

    "**强制分散指令**：\n"
    "4. 严禁输出末尾为 000、333、500、667、750、997 的分数（如 0.667、0.750、0.997 均属违规）；\n"
    "5. 【重要】必须模拟真实连续标尺，输出**非对称、生僻、独一无二**的小数点后三位，例如 0.684、-0.237、0.812、-0.418；\n"
    "6. 除非文本情绪极端强烈且无任何保留，否则严格禁止进入 ±0.900 以上的极值区；"
    "日常复杂文本请严格落在 ±0.200 至 ±0.800 之间的任意非对称浮点位置；\n"
    "7. 如果文本高度中性，请输出 ±0.001 到 ±0.099 之间**任意非整、非对称**的小数（如 0.053、-0.071）；\n\n"

    "输出要求：严格遵守只输出一个 (-1.000, 0) 或 (0, 1.000) 范围内且保留三位小数的数字，"
    "不允许添加任何文字、空格、换行或标点符号。"
)

SENTIMENT_USER_TEMPLATE = (
    "文本：{text}\n"
    "请输出情感得分："
)

# 初始化 DeepSeek 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

if USE_LOCAL_SENTIMENT:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    LOCAL_MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"
    print("正在加载本地情感分析模型...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_NAME)
    model.to(device)
    model.eval()

    id2label = model.config.id2label
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

    print(f"本地模型加载完成，积极索引: {pos_idx}, 消极索引: {neg_idx}")


def get_local_sentiment(text):
    """使用本地模型计算情感得分（-1 ~ 1）"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        p_pos = probs[pos_idx]
        p_neg = probs[neg_idx]

        if neutral_idx is not None:
            p_neu = probs[neutral_idx]
        else:
            p_neu = 0.0

        NEUTRAL_TO_NEGATIVE_RATIO = 0.0
        NEGATIVE_WEIGHT = 1
        POSITIVE_WEIGHT = 1
        score = POSITIVE_WEIGHT * p_pos - NEGATIVE_WEIGHT * (p_neg + NEUTRAL_TO_NEGATIVE_RATIO * p_neu)
        GLOBAL_NEGATIVE_BIAS = 0
        score = score - GLOBAL_NEGATIVE_BIAS
        score = max(-1.0, min(1.0, score))
        return round(float(score), 4)
    except Exception as e:
        print(f"本地情感分析出错: {e}")
        return None


def get_local_sentiment_batch(texts, batch_size=128, weighted=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    scores = []

    # 根据是否加权选择权重参数
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


def call_deepseek_sentiment_api(text, retries=3):
    """调用 DeepSeek API 获取情感得分（-1 到 1）"""
    user_content = SENTIMENT_USER_TEMPLATE.format(text=text)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=10
            )
            result_text = response.choices[0].message.content.strip()

            match = re.search(r'[-+]?\d*\.?\d+', result_text)
            if match:
                score = float(match.group())
                score = max(-1.0, min(1.0, score))
                return round(score, 4)
            else:
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"\n情感分析API请求失败: {e}")
                return None


def clean_data(text):
    """数据清洗函数"""
    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) <= 25:
        return False

    pattern = r'\[[\u4e00-\u9fa5]{2}\]'
    matches = re.findall(pattern, text)
    tags_length = sum(len(m) for m in matches)

    if tags_length / len(text) > 0.4:
        return False

    return True


def call_deepseek_api(comment, retries=3):
    """调用 DeepSeek API 进行内容蒸馏"""
    user_content = USER_TEMPLATE.format(comment=comment)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"\n蒸馏API请求失败: {e}")
                return "API请求失败"


def safe_get(row, column, default):
    """安全读取 DataFrame 行字段，处理缺失值与空字符串"""
    value = row.get(column, default)

    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    if isinstance(value, str):
        value = value.strip()
        return value if value else default

    return value


def load_new_sheet(path, sheet_name, source_label, content_column, account_column):
    """
    读取一个 Excel 文件中的指定 sheet，并统一列名。
    返回包含“原始内容”、“数据来源”、“账号分类”的 DataFrame。
    """
    print(f"正在读取 sheet: {sheet_name} ...")

    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except Exception as e:
        print(f"读取 sheet {sheet_name} 失败: {e}")
        return None

    if content_column not in df.columns:
        print(f"错误: sheet {sheet_name} 中未找到内容列 '{content_column}'！")
        return None

    if account_column not in df.columns:
        print(f"警告: sheet {sheet_name} 中未找到账号分类列 '{account_column}'，将根据 sheet 名称设置默认值。")
        if sheet_name == PUBLIC_SHEET_NAME:
            df[account_column] = "public"
        else:
            df[account_column] = None  # 媒体政府 sheet 无法简单默认，建议补全该列
    else:
        # 确保账号分类列存在且为字符串
        df[account_column] = df[account_column].fillna("public")

    df = df.rename(columns={content_column: RAW_CONTENT_COLUMN})
    df[SOURCE_COLUMN] = source_label

    print(f"{sheet_name} 读取成功，共 {len(df)} 条数据。")
    return df


def main():
    # 读取两个 sheet
    # df_public = load_new_sheet(
    #     path=INPUT_FILE,
    #     sheet_name=PUBLIC_SHEET_NAME,
    #     source_label=PUBLIC_SOURCE,
    #     content_column=CONTENT_COLUMN,
    #     account_column=ACCOUNT_COLUMN
    # )
    # df_media_gov = load_new_sheet(
    #     path=INPUT_FILE,
    #     sheet_name=MEDIA_GOV_SHEET_NAME,
    #     source_label=MEDIA_GOV_SOURCE,
    #     content_column=CONTENT_COLUMN,
    #     account_column=ACCOUNT_COLUMN
    # )
    #
    # frames = [df for df in [df_public, df_media_gov] if df is not None and not df.empty]
    # if not frames:
    #     print("没有成功读取任何数据，程序结束。")
    #     return
    #
    # df_all = pd.concat(frames, ignore_index=True)
    # print("\n数据合并完成。")
    # print(df_all[SOURCE_COLUMN].value_counts())
    #
    # print("\n正在进行数据清洗...")
    # initial_count = len(df_all)
    # mask = df_all[RAW_CONTENT_COLUMN].apply(clean_data)
    # df_cleaned = df_all[mask].copy()
    # cleaned_count = len(df_cleaned)
    # print(f"清洗完成！原始数据 {initial_count} 条，清洗后剩余 {cleaned_count} 条。")
    #
    # if cleaned_count == 0:
    #     print("没有符合条件的数据，程序结束。")
    #     return
    #
    # print("\n开始调用 DeepSeek API 进行内容蒸馏...")
    # summaries = []
    # user_types = []
    #
    # for index, row in tqdm(df_cleaned.iterrows(), total=cleaned_count, desc="蒸馏进度"):
    #     text = safe_get(row, RAW_CONTENT_COLUMN, "")
    #     summary = call_deepseek_api(text)
    #     summaries.append(summary)
    #
    #     # 直接读取已有的账号分类，不再调用分类 API
    #     user_type = safe_get(row, ACCOUNT_COLUMN, "public")
    #     user_types.append(user_type)
    #
    #     time.sleep(0.3)
    #
    # df_cleaned[DISTILLED_COLUMN] = summaries
    # df_cleaned[ACCOUNT_COLUMN] = user_types
    #
    # print(f"\n蒸馏完毕，正在保存阶段性结果至: {OUTPUT_FILE} ...")
    # # 保存时保留：原始内容、数据来源、账号分类、蒸馏总结
    # df_cleaned.to_excel(OUTPUT_FILE, index=False)
    # print("阶段性保存成功！")

    print(f"\n开始读取 {OUTPUT_FILE} 并计算情感得分...")

    try:
        df_final = pd.read_excel(OUTPUT_FILE)
    except Exception as e:
        print(f"读取刚刚保存的文件失败: {e}")
        return

    if "发帖人用户类型" not in df_final.columns:
        print("警告：输出文件中未找到“发帖人用户类型”列，跳过账号分类更新。")
    else:
        if ACCOUNT_COLUMN not in df_final.columns:
            df_final[ACCOUNT_COLUMN] = None

        df_final["发帖人用户类型"] = df_final["发帖人用户类型"].astype(str).str.strip()
        df_final.loc[df_final["发帖人用户类型"] == "政府", ACCOUNT_COLUMN] = "government"
        df_final.loc[df_final["发帖人用户类型"] == "媒体", ACCOUNT_COLUMN] = "media"

        print("已根据“发帖人用户类型”更新账号分类：政府→government，媒体→media。")
        print(df_final[ACCOUNT_COLUMN].value_counts())

    # 情感分析部分
    gov_media_indices = []
    other_indices = []
    gov_media_texts = []
    other_texts = []

    for idx, row in df_final.iterrows():
        summary_text = str(row.get(DISTILLED_COLUMN, ''))
        if "无观点" in summary_text:
            continue

        text = str(row.get(DISTILLED_COLUMN, ''))[:512]
        account_type = str(row.get(ACCOUNT_COLUMN, '')).strip()

        if account_type in ("government", "media"):
            gov_media_indices.append(idx)
            gov_media_texts.append(text)
        else:
            other_indices.append(idx)
            other_texts.append(text)

    print(f"government/media 类文本数量: {len(gov_media_texts)}")
    print(f"其他类文本数量: {len(other_texts)}")

    # 判断是否需要加载本地模型
    need_local_model = USE_LOCAL_SENTIMENT or len(gov_media_texts) > 0

    if need_local_model:
        if not USE_LOCAL_SENTIMENT:
            print("检测到 government/media 数据，动态加载本地情感模型...")
            global tokenizer, model, pos_idx, neg_idx, neutral_idx
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            LOCAL_MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_NAME)
            model.to(device)
            model.eval()

            id2label = model.config.id2label
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
            print("本地情感模型加载完成。")

    # 初始化得分数组
    if '情感得分' in df_final.columns:
        sentiment_scores = df_final['情感得分'].apply(lambda x: None if pd.isna(x) else x).tolist()
    else:
        sentiment_scores = [None] * len(df_final)

    # 记录本次新计算的索引，用于后续随机偏置
    newly_computed_indices = []

    # 处理 government/media 组（始终使用本地模型）
    if gov_media_texts:
        print("正在对 government/media 类文本使用本地模型批量计算情感得分...")
        local_scores = get_local_sentiment_batch(gov_media_texts, batch_size=128, weighted=True)
        for idx, score in zip(gov_media_indices, local_scores):
            sentiment_scores[idx] = score
            newly_computed_indices.append(idx)

    # 处理其他组
    if other_texts:
        if USE_LOCAL_SENTIMENT:
            print("正在对其他类文本使用本地模型批量计算情感得分...")
            other_local_scores = get_local_sentiment_batch(other_texts, batch_size=128, weighted=False)
            for idx, score in zip(other_indices, other_local_scores):
                sentiment_scores[idx] = score
        else:
            print("正在对其他类文本调用 DeepSeek API 计算情感得分...")
            api_scores = []
            for text in tqdm(other_texts, desc="API 情感分析进度"):
                score = call_deepseek_sentiment_api(text)
                api_scores.append(score)
                time.sleep(0.2)
            for idx, score in zip(other_indices, api_scores):
                sentiment_scores[idx] = score

    # 填回得分列
    df_final['情感得分'] = sentiment_scores

    # 随机偏置
    print("正在为所有有效情感得分添加随机偏置...")
    max_bias_attempts = 10  # 最大循环次数，防止极端情况死循环

    for idx in df_final.index:
        score = df_final.loc[idx, '情感得分']
        # 跳过空值或 NaN
        if score is None or pd.isna(score):
            continue

        current_score = float(score)
        attempt = 0

        # 循环直到分数不再为 -1 或 1，或达到最大尝试次数
        while attempt < max_bias_attempts:
            # 生成随机偏置（75% 负偏置，25% 正偏置）
            if random.random() < 0.75:
                bias = random.uniform(0, 0.1)
            else:
                bias = -random.uniform(0, 0.1)

            new_score = current_score + bias
            new_score = max(-1.0, min(1.0, new_score))

            # 如果结果不是边界值，接受
            if new_score != -1.0 and new_score != 1.0:
                df_final.loc[idx, '情感得分'] = round(new_score, 4)
                break

            # 若仍为边界，则基于边界值继续添加偏置
            current_score = new_score
            attempt += 1

        # 如果达到最大次数后仍是边界，强制微调
        if attempt == max_bias_attempts:
            if current_score == 1.0:
                final_score = 1.0 - random.uniform(0.001, 0.01)
            elif current_score == -1.0:
                final_score = -1.0 + random.uniform(0.001, 0.01)
            else:
                final_score = current_score
            df_final.loc[idx, '情感得分'] = round(final_score, 4)

    # 保存最终结果
    df_final.to_excel(OUTPUT_FILE, index=False)
    print(f"\n所有流程处理完毕！最终带有情感得分的数据已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
