"""
阶段 1.5：情感分析 & AI 评论观点提炼
API Key: sk-70411e5d7f9e4c3da4517797dc730c97
"""
from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import json
import logging
import re
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import torch
from transformers import pipeline

# 路径 & 默认配置
DEEPSEEK_API_KEY = "sk-70411e5d7f9e4c3da4517797dc730c97"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
SENTIMENT_MODEL = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment" # 情感分析模型

CLEANED_DATA_PATH = "/root/pmd/processed_data/cleaned_data.csv"
OUTPUT_PATH = "./processed_data/ai_summarized.csv"
CHECKPOINT_PATH = "./processed_data/ai_summarizer_checkpoint.json"
LOG_PATH = "./processed_data/ai_summarizer.log"

SAMPLE_SIZE = 15000    # 抽取条数
BATCH_SIZE = 10         # API 并发批大小
MAX_TOKENS = 1024        # 每条总结最多生成 token 数
CHECKPOINT_N = 100      # 每处理 N 条保存一次检查点
API_TIMEOUT = 60        # API 请求超时时间（秒）
MAX_RETRIES = 3         # API 失败重试次数

# 提示词
SYSTEM_PROMPT = (
    "你是一个社会学数据编码员。你的任务是阅读用户关于“预制菜”的知乎回答，精准提炼用户评论中关于“预制菜（即食/半成品）”的核心观点和底层逻辑，用于后续的主题聚类。\n "
 "【严格输出规则】：\n "
 "1. 【极其重要】仅提取评论中与预制菜相关的部分，"
 "    包括但不限于：与预制菜直接相关的段落，与预制菜间接相关的段落，与预制菜相关的事件，与预制菜相关的人，以及对与预制菜相关的人或事件的评估或态度。\n"
 "2. 禁止使用“核心立场”、“理由是”、“总结来说”等说明性/元标签词汇。\n"
 "3. 直接用第三人称陈述该用户的观点内核，例如：“认为预制菜含有大量防腐剂等添加剂，严重威胁儿童身体健康，本质是资本逐利下对公众知情权的剥夺...”\n"
 "4. 忽略情绪发泄、个人故事和无关细节。"
 "5. 准确提炼评论者的情感倾向（支持/反对/中立/担忧/讽刺等），剔除无意义的语气词。\n"
 "6. 禁止使用 1、2、3 等序号分点排版，必须用一到两段连贯的纯文本进行总结，字数控制在200-400字，确保是一段完整的陈述。\n"
 "7. 【极其重要】直接输出最终中文文本！仅提供结果，禁止输出任何思考过程！禁止包含任何英文字符！禁止输出'Thinking Process'、'分析'、'步骤'等任何中间过程！从第一行开始就是最终总结，不要有任何前缀！\n")

USER_TEMPLATE = (
    "评论原文：\n"
    "'{comment}''\n\n"
    "请直接输出中文总结（仅最终结果，不要 Thinking Process 等思考过程）："
)

# 日志配置
os.makedirs("./processed_data", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class DeepSeekSummarizer:
    """
    用 Hugging Face 模型计算情感，并用 DeepSeek API 对评论做预制菜观点提炼。
    """

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        api_url: str = DEEPSEEK_API_URL,
        model: str = DEEPSEEK_MODEL,
        cleaned_data_path: str = CLEANED_DATA_PATH,
        output_path: str = OUTPUT_PATH,
        sample_size: int = SAMPLE_SIZE,
        batch_size: int = BATCH_SIZE,
        sampling_strategy: str = "stratified",
        checkpoint_path: str = CHECKPOINT_PATH,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.cleaned_data_path = cleaned_data_path
        self.output_path = output_path
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.sampling_strategy = sampling_strategy
        self.checkpoint_path = checkpoint_path

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        self.sampled_df = None

    # --- 新增：计算情感得分 ---
    def calculate_sentiment(self):
        """在原有评论基础上计算情感得分"""
        if self.sampled_df is None:
            logger.error("数据未加载，请先执行 load_and_sample。")
            return

        logger.info(f"加载情感分析模型: {SENTIMENT_MODEL} ...")
        device = 0 if torch.cuda.is_available() else -1
        sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=device)

        logger.info("开始计算情感得分...")
        texts = self.sampled_df["原始评论" if "原始评论" in self.sampled_df.columns else "评论"].fillna("").astype(str).tolist()

        # 手动截断稍微放宽一点，或者直接去掉限制留给 tokenizer 处理（建议留个余量比如 500）
        texts = [t[:500] if len(t) > 0 else "中立" for t in texts]

        # 批量推理：必须明确传入 max_length=512，让 tokenizer 正确截断 token 而非单纯截断字符串
        results = sentiment_pipeline(texts, batch_size=32, truncation=True, max_length=512)

        sentiments = []
        polarities = []
        for res in results:
            label = res['label']
            score = res['score']

            # 映射得分至 [-1, 1] 范围，并生成供报告使用的极性标签
            if label == 'Positive':
                sentiments.append(score)
                polarities.append("正向")
            else:
                sentiments.append(-score)
                polarities.append("负向")

        self.sampled_df['sentiment_score'] = sentiments
        self.sampled_df['sentiment_polarity'] = polarities # 兼容现有的 _print_report
        logger.info("情感得分计算完成！")

    # 1. API 调用
    def _call_api(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.5,
            "top_p": 0.9,
            "stream": False,
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    self.api_url,
                    json=payload,
                    timeout=API_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content
            except requests.exceptions.RequestException as e:
                logger.warning(f"API 请求失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error("API 请求多次失败，返回空字符串")
                    return ""
            except (KeyError, IndexError) as e:
                logger.error(f"API 响应格式错误：{e}")
                return ""

    # 2. 构建 prompt
    def _build_prompt(self, comment: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(comment=comment[:400])},
        ]

    # 3. 单条推理
    def _infer_single(self, comment: str) -> str:
        messages = self._build_prompt(comment)
        raw_text = self._call_api(messages)
        return self._postprocess(raw_text)

    # 4. 后处理
    def _postprocess(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        for prefix in ("assistant:", "Assistant:", "助手:", "回答:", "总结:", "输出:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        if "Thinking Process" in text or "思考过程" in text or "分析：" in text:
            parts = text.split("\n\n")
            chinese_parts = [p for p in parts if len(p) > 10 and any('\u4e00' <= c <= '\u9fff' for c in p)]
            if chinese_parts:
                text = chinese_parts[-1].strip()

        lines = text.split('\n')
        final_text = []
        for line in lines:
            line = line.strip()
            if (line.startswith(('1.', '2.', '3.', '*', '#', 'Thinking', '分析', '步骤')) or
                    'Thinking Process' in line or
                    line.startswith(('**', '##'))):
                continue
            if line and any('\u4e00' <= c <= '\u9fff' for c in line):
                final_text.append(line)

        text = '\n'.join(final_text).strip() if final_text else text
        text = ''.join(c for c in text if '\u4e00' <= c <= '\u9fff' or c in '，。！？；：""''（）【】…—')

        unwanted_phrases = [
            "该评论者", "该评论", "该用户", "该文本", "该回答",
            "该网友", "该作者", "此人", "这位用户",
            "认为", "表示", "指出", "提到", "说道",
            "评论者", "用户", "作者", "网友",
        ]

        prefix_len = min(10, len(text))
        prefix = text[:prefix_len]
        suffix = text[prefix_len:]

        for phrase in unwanted_phrases:
            if phrase in prefix:
                prefix = prefix.replace(phrase, "")

        text = prefix + suffix
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'^[，。！？；：]', '', text)

        if len(text) > 500:
            text = text[:500]

        text = re.sub(r'["''"""]{2,}', '"', text)
        text = re.sub(r'[，。！？；：]{2,}', '。', text)
        text = text.strip().strip('"').strip("'").strip('"').strip('"')

        return text

    # 5. 数据加载与抽样
    def load_and_sample(self) -> pd.DataFrame:
        logger.info(f"加载数据：{self.cleaned_data_path}")
        df = pd.read_csv(self.cleaned_data_path, encoding="utf-8-sig")

        if "评论" not in df.columns:
            cands = [c for c in df.columns
                     if any(k in c.lower() for k in ("comment", "text", "内容", "评论"))]
            if cands:
                df.rename(columns={cands[0]: "评论"}, inplace=True)
            else:
                raise ValueError("CSV 中未找到评论内容列，请检查列名。")

        df["parsed_time"] = pd.to_datetime(df.get("parsed_time", pd.NaT), errors="coerce")
        df["评论"] = df["评论"].fillna("").astype(str)
        df = df[df["评论"].str.strip() != ""].reset_index(drop=True)
        logger.info(f"有效评论：{len(df)} 条")

        n = min(self.sample_size, len(df))

        if self.sampling_strategy == "stratified" and "parsed_time" in df.columns:
            df["_ym"] = df["parsed_time"].dt.to_period("M").astype(str)
            ym_counts = df["_ym"].value_counts()
            quota = (ym_counts / ym_counts.sum() * n).apply(lambda x: max(1, round(x)))
            diff = int(quota.sum()) - n
            if diff > 0:
                for ym in quota.nlargest(diff).index:
                    if quota[ym] > 1:
                        quota[ym] -= 1
            elif diff < 0:
                for ym in quota.nlargest(-diff).index:
                    quota[ym] += 1
            parts = []
            for ym, q in quota.items():
                grp = df[df["_ym"] == ym]
                parts.append(grp.sample(min(int(q), len(grp)), random_state=42))
            sampled = pd.concat(parts).sample(frac=1, random_state=42)
            df.drop(columns=["_ym"], inplace=True)
        else:
            sampled = df.sample(n, random_state=42)

        sampled = sampled.reset_index(drop=True)
        sampled["original_comment"] = sampled["评论"]
        self.sampled_df = sampled

        logger.info(f"抽样完成：{len(sampled)} 条（策略={self.sampling_strategy}）")
        if "parsed_time" in sampled.columns:
            valid_t = sampled["parsed_time"].dropna()
            if not valid_t.empty:
                logger.info(f"时间范围：{valid_t.min().date()} → {valid_t.max().date()}")
        return sampled

    # 6. 检查点管理
    def _load_checkpoint(self) -> dict[str, str]:
        # 可以根据需要取消这里的注释开启断点续跑
        # if Path(self.checkpoint_path).exists():
        #     with open(self.checkpoint_path, "r", encoding="utf-8") as f:
        #         data = json.load(f)
        #     logger.info(f"断点续跑：已恢复 {len(data)} 条")
        #     return data
        return {}

    def _save_checkpoint(self, results: dict[str, str]):
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 7. 完整推理流程
    def run_summarization(self) -> pd.DataFrame:
        if self.sampled_df is None:
            self.load_and_sample()

        df = self.sampled_df.copy()
        completed = self._load_checkpoint()

        pending_idx = [i for i in df.index if str(i) not in completed]
        pending_comments = [str(df.at[i, "original_comment"]) for i in pending_idx]
        logger.info(f"待推理：{len(pending_idx)} 条 / 已完成：{len(completed)} 条")

        results = dict(completed)

        if pending_idx:
            t0 = time.time()
            save_interval = max(1, CHECKPOINT_N)
            batch_count = 0

            for i in tqdm(
                range(len(pending_idx)),
                desc="API 推理进度",
                unit="条",
            ):
                row_i = pending_idx[i]
                comment = pending_comments[i]

                summary = self._infer_single(comment)
                results[str(row_i)] = summary

                batch_count += 1
                if batch_count % save_interval == 0:
                    self._save_checkpoint(results)
                    logger.info(f"检查点：{len(results)}/{len(df)} 条")

                if batch_count % 10 == 0:
                    time.sleep(0.5)

            self._save_checkpoint(results)
            elapsed = time.time() - t0
            logger.info(
                f"推理完成：{len(pending_idx)} 条，"
                f"耗时 {elapsed:.1f}s ({elapsed / max(len(pending_idx), 1):.2f}s/条)"
            )

        df["ai_summary"] = df.index.map(lambda i: results.get(str(i), ""))

        before = len(df)
        df_relevant = df[df["ai_summary"].str.strip().ne("")].copy()
        dropped = before - len(df_relevant)
        logger.info(f"过滤无关条目：{dropped} 条 → 保留 {len(df_relevant)} 条")

        df_relevant["评论"] = df_relevant["ai_summary"]
        self.sampled_df = df_relevant.reset_index(drop=True)
        return self.sampled_df

    # 8. 保存结果
    def save_results(self) -> str:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.sampled_df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        logger.info(f"结果已保存：{self.output_path}（{len(self.sampled_df)} 条）")
        self._print_report()
        return self.output_path

    def _print_report(self):
        df = self.sampled_df
        sep = "=" * 64
        print(f"\n{sep}")
        print("  阶段 1.5 完成：情感分析 & AI 评论观点提炼")
        print(sep)
        print(f"  提取模型          : {self.model}")
        print(f"  情感模型          : {SENTIMENT_MODEL}")
        print(f"  有效总结条数      : {len(df)}")

        if "parsed_time" in df.columns:
            valid_t = df["parsed_time"].dropna()
            if not valid_t.empty:
                print(f"  时间范围          : {valid_t.min().date()} → {valid_t.max().date()}")

        lengths = df["ai_summary"].str.len()
        print(f"  总结长度（均值）  : {lengths.mean():.1f} 字")
        print(f"  总结长度（最短）  : {lengths.min()} 字")
        print(f"  总结长度（最长）  : {lengths.max()} 字")

        # 这里的打印现在会读取上一步计算出的 sentiment_polarity 了
        if "sentiment_polarity" in df.columns:
            print(f"\n  情感分布：")
            for label, cnt in df["sentiment_polarity"].value_counts().items():
                pct = cnt / len(df) * 100
                print(f"    {label:<10}: {cnt:>5} 条（{pct:.1f}%）")

        if "parsed_time" in df.columns:
            df["_ym"] = df["parsed_time"].dt.to_period("M").astype(str)
            top_months = df["_ym"].value_counts().head(5)
            print(f"\n  评论量前 5 月份：")
            for ym, cnt in top_months.items():
                print(f"    {ym}: {cnt} 条")
            df.drop(columns=["_ym"], inplace=True)

        print(f"\n  输出文件  : {self.output_path}")
        print(f"\n  下一步→ 阶段二（主题建模）:")
        print(f"    python topic_model.py")
        print(f"    （已自动检测并优先使用 ai_summarized_1000.csv）")
        print(sep + "\n")

    # 9. 一键运行 (修改了执行顺序)
    def run(self) -> str:
        """完整流程：抽样 → 情感分析 → API 推理 → 保存"""
        self.load_and_sample()
        self.calculate_sentiment() # 在抽样之后，API 提炼之前计算原文本情感
        self.run_summarization()
        return self.save_results()


def get_pipeline_data_path(
    ai_path: str = OUTPUT_PATH,
    base_path: str = CLEANED_DATA_PATH,
) -> str:
    if Path(ai_path).exists():
        return ai_path
    return base_path


def _parse_args():
    p = argparse.ArgumentParser(
        description="阶段 1.5：情感分析 & AI 评论观点提炼",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--api-key", default=DEEPSEEK_API_KEY,
                   help="DeepSeek API Key")
    p.add_argument("--model", default=DEEPSEEK_MODEL,
                   help="DeepSeek 模型名称")
    p.add_argument("--input", default=CLEANED_DATA_PATH,
                   help="cleaned_data.csv 路径")
    p.add_argument("--output", default=OUTPUT_PATH,
                   help="输出 CSV 路径")
    p.add_argument("--n", type=int, default=SAMPLE_SIZE,
                   help="抽取评论数量")
    p.add_argument("--strategy", choices=["stratified", "random"],
                   default="stratified",
                   help="抽样策略：stratified=按月分层，random=全局随机")
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                   help="断点续跑 JSON 文件路径")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summarizer = DeepSeekSummarizer(
        api_key=args.api_key,
        model=args.model,
        cleaned_data_path=args.input,
        output_path=args.output,
        sample_size=args.n,
        sampling_strategy=args.strategy,
        checkpoint_path=args.checkpoint,
    )

    output_file = summarizer.run()
    print(f"\n  阶段 1.5 完成！输出：{output_file}")