import os
import re
import json
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np

# 引入 vLLM 和 对应的 Tokenizer
from transformers import AutoTokenizer
import transformers
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


INPUT_CSVS = ["/root/autodl-tmp/opinion_distill.csv", "/root/autodl-tmp/data2.csv"]
OUTPUT_JSONL = "./processed_data/llm_classified_topics0.jsonl"
OUTPUT_FULL_JSON = "./processed_data/llm_full_outputs0.json"
# 修改了输出文件名，以区分时序图谱
OUTPUT_GEXF_TEMPORAL = "./processed_data/topics_temporal_graph0.gexf"
LLM_PATH = "/root/autodl-tmp/Qwen3.5-9B"

EVENT_LABELS: dict[str, list[str]] = {
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

TOPICS_DEF = {
    1: "政策支持预制菜产业发展，同时主张通过监管和标准建设规范行业",
    2: "预制菜代表餐饮工业化和标准化发展方向，强调效率提升、规模化生产和中央厨房模式",
    3: "资本或行业力量主导标准制定并规避监管，导致制度公信力和食品安全信任受损",
    4: "资本逐利导致预制菜品质下降和安全风险上升，公众因此表达抵制态度",
    5: "资本与权力关系被认为相互勾连，造成商业欺诈、舆论操控和结构性不公",
    6: "预制菜因价格较低、获取便利、口味尚可，在现实消费场景中具有一定接受度",
    7: "信息不对称使消费者难以识别预制菜，知情权受损并构成商业欺诈问题",
    8: "效率优先的餐饮工业化挤压传统烹饪文化和厨师职业价值",
    9: "预制菜存在高价低质和性价比失衡问题，由此引发消费层面的价值争议",
    10: "政府对预制菜产业的推动被公众解读为助长资本扩张，从而引发警惕和担忧",
    11: "价格低且质量尚可的预制菜产品因高性价比而被消费者接受",
    12: "预制菜与传统食品、半成品、腌制食品等概念边界模糊，引发认知混乱和定义争议",
    13: "预制菜较长保质期引发公众对防腐添加、储存方式和食品安全性的担忧",
    14: "消费者应享有对预制菜的知情权和自主选择权，反对隐瞒、替代或强制接受",
    15: "关于减少浪费与国家战略储备的支持讨论",
    16: "预制菜在年夜饭等节庆消费场景中的快速增长，被视为新的消费趋势和市场现象",
    17: "西方预制食品是工业化产物，与中餐现做现炒理念存在本质冲突，强调文化自信与本土优势",
    18: "餐饮工业化是效率提升的必然趋势，但需结合国情建立本土标准与监管体系，批判盲目照搬西方模式",
    19: "西方食品在效率、食品标识、法规约束方面的制度优势，反对以文化传统为由抵制工业化进程",
}

TOPICS_NAME = {
    1: "政策扶持规范产业发展",
    2: "餐饮工业化重塑行业发展",
    3: "标准垄断的制度信任危机",
    4: "餐饮工业化安全质疑",
    5: "资本权力合谋的信任危机",
    6: "餐饮工业化现实接受度高",
    7: "信息不对称侵犯知情权",
    8: "工业化冲击传统烹饪文化",
    9: "高价低质引发欺诈争议",
    10: "政策扶持预制菜引发担忧",
    11: "性价比高受大众认可",
    12: "预制菜的概念争议",
    13: "保质期与添加剂安全质疑",
    14: "保障知情权与选择权",
    15: "支持预制产品的功能价值",
    16: "市场爆发行业潜力高",
    17: "中餐现炒理念具独特价值",
    18: "餐饮工业化趋势需本土适配",
    19: "西方工业化食品制度优势",
}


def _parse_event_ts(key: str) -> pd.Timestamp:
    if len(key) == 7:
        return pd.Timestamp(key + "-01")
    return pd.Timestamp(key)


_SORTED_EVENTS: list[tuple[pd.Timestamp, str]] = sorted(
    [(_parse_event_ts(k), labels[0] if labels else k) for k, labels in EVENT_LABELS.items()],
    key=lambda x: x[0],
)


def get_event_segment(date) -> str:
    if pd.isna(date) or date is None:
        return "未知"
    ts = pd.Timestamp(date)
    result = "早期"
    for evt_ts, evt_label in _SORTED_EVENTS:
        if ts >= evt_ts:
            result = evt_label
        else:
            break
    return result


def load_and_clean_data(csv_paths: list[str]) -> pd.DataFrame:
    print(f"\n[1/4] 加载并拼接 {len(csv_paths)} 份数据…")
    dfs = []

    for csv_path in csv_paths:
        print(f"  - 正在读取: {csv_path}")
        df_temp = pd.read_csv(csv_path, encoding="utf-8-sig")
        df_temp['file_origin'] = csv_path
        if 'source' not in df_temp.columns:
            df_temp['source'] = 'public'

        for cand in ("distilled_text", "评论"):
            if cand in df_temp.columns:
                if cand != "distilled_text":
                    df_temp.rename(columns={cand: "distilled_text"}, inplace=True)
                break

        df_temp["distilled_text"] = df_temp["distilled_text"].fillna("").astype(str)
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)

    combined_time = pd.Series([np.nan] * len(df), index=df.index)
    for tcand in ("date", "时间", "time"):
        if tcand in df.columns:
            valid_times = df[tcand].astype(str).replace(['nan', 'None'], np.nan)
            combined_time = combined_time.fillna(valid_times)

    time_strs = combined_time.astype(str).str.extract(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})')[0]
    time_strs = time_strs.str.replace("年", "-").str.replace("月", "-").str.replace("日", "")
    df["date"] = pd.to_datetime(time_strs, errors="coerce")

    mask_date = (df["date"] >= pd.Timestamp("2023-01-01")) | df["date"].isna()
    df = df[mask_date].copy()

    mask_dedup = ~df.duplicated(subset=["distilled_text"]).values
    df = df.iloc[mask_dedup].copy()
    mask_empty = df["distilled_text"].str.strip().ne("").values
    df = df.iloc[mask_empty].reset_index(drop=True)

    print(f"  - 清洗后总数据量: {len(df)} 条")
    return df


def classify_topics_with_llm(df: pd.DataFrame, llm_path: str, output_jsonl: str):
    print(f"\n[2/4] 初始化 vLLM Qwen3.5-9B 模型并进行演绎式编码…")

    existing_results = {}

    system_prompt = "你是一名传播学研究编码助手。【最高指令】：严禁输出'Thinking Process'或任何分析步骤！你的回复必须且只能以 '{' 开头，以 '}' 结尾的合法JSON格式。不要任何多余的解释。"
    base_prompt = """请根据我提供的“主题定义”，判断目标文本涉及哪些主题。
这里的主题是“话题—立场单元”，判断标准不是关键词是否出现，而是文本是否明确表达或实质涉及该主题所描述的问题及其立场。

请严格遵守以下规则：
1. matched_topics列表禁止返回空列表。
2. 如果文本只是出现了相关词语，但实际语义并不构成该主题，不要归类。
3. 如果文本同时表达了多个不同主题，可返回多个主题编号。
4. 可多选，但不要滥选。只有当文本内容与主题定义在语义上明确匹配时，才可归入该主题。
5. 请优先依据文本的核心观点、问题界定、责任归因和价值判断进行判断，而非表面词语。
6. 输出必须为标准JSON格式，不要添加任何解释性文字。

主题定义如下：
Topic 1：政策支持预制菜产业发展，同时主张通过监管和标准建设规范行业。
Topic 2：预制菜代表餐饮工业化和标准化发展方向，强调效率提升、规模化生产和中央厨房模式。
Topic 3：资本或行业力量主导标准制定并规避监管，导致制度公信力和食品安全信任受损。
Topic 4：资本逐利导致预制菜品质下降和安全风险上升，公众因此表达抵制态度。
Topic 5：资本与权力关系被认为相互勾连，造成商业欺诈、舆论操控和结构性不公。
Topic 6：预制菜因价格较低、获取便利、口味尚可，在现实消费场景中具有一定接受度。
Topic 7：信息不对称使消费者难以识别预制菜，知情权受损并构成商业欺诈问题。
Topic 8：效率优先的餐饮工业化挤压传统烹饪文化和厨师职业价值。
Topic 9：预制菜存在高价低质和性价比失衡问题，由此引发消费层面的价值争议。
Topic 10：政府对预制菜产业的推动被公众解读为助长资本扩张，从而引发警惕和担忧。
Topic 11：价格低且质量尚可的预制菜产品因高性价比而被消费者接受。
Topic 12：预制菜与传统食品、半成品、腌制食品等概念边界模糊，引发认知混乱和定义争议。
Topic 13：预制菜较长保质期引发公众对防腐添加、储存方式和食品安全性的担忧。
Topic 14：消费者应享有对预制菜的知情权和自主选择权，反对隐瞒、替代或强制接受。
Topic 15：关于减少浪费与国家战略储备的支持讨论。
Topic 16：预制菜在年夜饭等节庆消费场景中的快速增长，被视为新的消费趋势和市场现象。
Topic 17：西方预制食品是工业化产物，与中餐现做现炒理念存在本质冲突，强调文化自信与本土优势。
Topic 18：餐饮工业化是效率提升的必然趋势，但需结合国情建立本土标准与监管体系，批判盲目照搬西方模式。
Topic 19：西方食品在效率、食品标识、法规约束方面的制度优势，反对以文化传统为由抵制工业化进程。

你的回复必须且只能以 '{' 开头，以 '}' 结尾的合法JSON格式，请按如下JSON格式输出：
{
 "matched_topics": [1, 3],
 "low_confidence_topics": [],
 "reason": {
   "1": "说明匹配原因",
   "3": "说明匹配原因"
 }
}

示例输出：
{
 "matched_topics": [7, 14, 15],
 "low_confidence_topics": [9],
 "reason": {
   "7": "控诉商家不告知使用料理包，使得消费者难以识别，知情权受损。",
   "14": "对长达一年的保质期和可能添加的防腐剂表达了强烈的安全担忧。",
   "15": "明确提出点餐时必须告知，赋予消费者自主选择是否接受的权利。",
   "9": "提到‘花现炒的钱吃罐头’，有高价低质的意味，但核心侧重于知情权，故列入低置信度。"
 }
}

{
 "matched_topics": [4, 8],
 "low_confidence_topics": [],
 "reason": {
   "4": "明确批评大资本涌入和逐利导致菜品质量下降，并表达了坚决抵制的态度。",
   "8": "感叹充满烟火气和考验火候的厨师现炒手艺被流水线机器挤压，失去生存空间。"
 }
}

目标文本如下：
"""

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    llm = LLM(
        model=llm_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        max_model_len=12288
    )

    json_schema = {
        "type": "object",
        "properties": {
            "matched_topics": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1
            },
            "low_confidence_topics": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "reason": {
                "type": "object",
                "additionalProperties": {"type": "string"}
            }
        },
        "required": ["matched_topics", "low_confidence_topics", "reason"],
        "additionalProperties": False
    }
    structured_outputs_params_json = StructuredOutputsParams(json=json_schema)

    sampling_params = SamplingParams(
        max_tokens=3072,
        temperature=0.0,
        structured_outputs=structured_outputs_params_json
    )

    print("  - 正在构建批处理 Prompts")
    pending_rows = []
    prompts = []

    for idx, row in df.iterrows():
        if idx in existing_results:
            continue

        text = row['distilled_text']
        if not str(text).strip():
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_prompt + f"{text}"}
        ]

        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(prompt_str)
        pending_rows.append((idx, row))

    if not prompts:
        print("  - 所有有效文本均已处理完毕。")
        return existing_results

    print(f"  - 开始 vLLM 批量推理，共 {len(prompts)} 条数据")
    outputs = llm.generate(prompts, sampling_params)

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    print("  - 正在解析结果")
    with open(output_jsonl, 'a', encoding='utf-8') as out_file:
        for (idx, row), output in zip(pending_rows, outputs):
            response_text = output.outputs[0].text.strip()

            print(f"response: {response_text}")

            # 3. 定位 JSON 边界
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')

            parsed_json = {"error": "未找到JSON结构", "raw": response_text}
            matched_topics = []

            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                json_str = response_text[start_idx:end_idx + 1]
                parsed_json = json.loads(json_str)
                matched_topics = parsed_json.get('matched_topics', [])

            record = {
                "doc_id": idx,
                "text": str(row['distilled_text']),
                "source": row.get('source', 'public'),
                "date": row['date'].strftime("%Y-%m-%d") if not pd.isna(row['date']) else None,
                "matched_topics": [t for t in matched_topics if t in TOPICS_DEF],
                "llm_output": response_text
            }

            out_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            existing_results[idx] = record

    print(f"  - 正在将所有 {len(existing_results)} 条输出整合至: {OUTPUT_FULL_JSON}")
    # 将字典值转换为列表
    full_output_list = list(existing_results.values())
    with open(OUTPUT_FULL_JSON, 'w', encoding='utf-8') as f:
        json.dump(full_output_list, f, ensure_ascii=False, indent=2)

    print("  ✓ vLLM 推理阶段完成。")
    return existing_results


def export_temporal_topic_graph(records: dict, output_gexf: str):
    print(f"\n[3/4] 正在导出时序演化网络图谱: {output_gexf}")

    G = nx.Graph()

    # 1. 获取有序的所有事件段落名
    ordered_segments = ["早期"]
    for _, evt_label in _SORTED_EVENTS:
        if evt_label not in ordered_segments:
            ordered_segments.append(evt_label)

    # 2. 将有效文档按事件段落分组
    seg_to_records = defaultdict(list)
    for doc_id, record in records.items():
        # 如果是已知日期，则计算其所在的事件段落
        date_val = record.get('date')
        seg = get_event_segment(pd.Timestamp(date_val) if date_val else pd.NaT)
        seg_to_records[seg].append(record)

    # 记录每个主题上一次出现的节点ID，用于绘制时序演化虚线
    last_seen_nodes = {}

    # 3. 按时间顺序生成网络结构
    for seg in ordered_segments:
        records_in_seg = seg_to_records[seg]
        if not records_in_seg:
            continue

        cooc_seg = defaultdict(int)
        doc_counts_seg = defaultdict(int)

        # 计算当前时间段内的节点词频和共现矩阵
        for rec in records_in_seg:
            topics = rec.get('matched_topics', [])
            valid_topics = list(set([t for t in topics if t in TOPICS_DEF]))

            # 增加节点命中频次
            for t in valid_topics:
                doc_counts_seg[t] += 1

            # 建立段落内共现边
            if len(valid_topics) >= 2:
                for t1, t2 in combinations(sorted(valid_topics), 2):
                    cooc_seg[(t1, t2)] += 1

        active_tids_in_seg = [tid for tid in doc_counts_seg.keys() if doc_counts_seg[tid] > 0]
        current_seg_nodes = {}

        for tid in active_tids_in_seg:
            node_id = f"{seg}_Topic_{tid}"

            G.add_node(
                node_id,
                label=f"[{seg}] {TOPICS_DEF[tid]}",
                topic_id=int(tid),
                topic_name=TOPICS_DEF[tid],
                doc_count=int(doc_counts_seg[tid]),
                event_segment=seg
            )
            current_seg_nodes[tid] = node_id

            # 构建跨时间段落的时序同构连线
            if tid in last_seen_nodes:
                G.add_edge(
                    last_seen_nodes[tid],
                    node_id,
                    weight=1.0,
                    edge_type="temporal_link",
                )

            # 更新最新露面记录
            last_seen_nodes[tid] = node_id

        # 构建当前段落内的子主题共现连线
        for i in range(len(active_tids_in_seg)):
            for j in range(i + 1, len(active_tids_in_seg)):
                tid_i = active_tids_in_seg[i]
                tid_j = active_tids_in_seg[j]

                pair = tuple(sorted([tid_i, tid_j]))
                weight = cooc_seg.get(pair, 0)

                if weight > 0:
                    G.add_edge(
                        current_seg_nodes[tid_i],
                        current_seg_nodes[tid_j],
                        weight=float(weight),
                        edge_type="cooccurrence",
                    )

    Path(output_gexf).parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, output_gexf)
    print(f"  ✓ 时序网络已保存：{output_gexf}")


def export_topic_adjacency_matrix(records: dict, output_csv: str):
    print(f"\n构建话题立场共现邻接矩阵")
    valid_tids = sorted([tid for tid in TOPICS_DEF.keys()])
    adj_matrix = pd.DataFrame(0, index=valid_tids, columns=valid_tids)

    for rec in records.values():
        topics = rec.get('matched_topics', [])
        valid_topics = [t for t in topics if t in TOPICS_DEF]
        for t1 in valid_topics:
            for t2 in valid_topics:
                adj_matrix.loc[t1, t2] += 1

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    adj_matrix.to_csv(output_csv, encoding="utf-8-sig")
    print(f"  ✓ 话题邻接矩阵已保存至: {output_csv}")

def main():
    df = load_and_clean_data(INPUT_CSVS)

    records = classify_topics_with_llm(df, LLM_PATH, OUTPUT_JSONL)
    export_topic_adjacency_matrix(records, "./processed_data/topic_adjacency_matrix.csv")

    records_media = {k: v for k, v in records.items() if v.get('source') == 'media'}
    records_gov = {k: v for k, v in records.items() if v.get('source') == 'government'}
    records_other = {k: v for k, v in records.items() if v.get('source') not in ['media', 'government']}

    print(f"  - Media 子集: {len(records_media)} 条")
    print(f"  - Government 子集: {len(records_gov)} 条")
    print(f"  - 其他混合子集: {len(records_other)} 条")

    output_base = OUTPUT_GEXF_TEMPORAL.replace(".gexf", "")

    # 使用新的时序图谱生成函数
    export_temporal_topic_graph(records_media, f"{output_base}_media0.gexf")
    export_temporal_topic_graph(records_gov, f"{output_base}_gov0.gexf")
    export_temporal_topic_graph(records_other, f"{output_base}_other0.gexf")
    export_temporal_topic_graph(records, f"{output_base}_all0.gexf")

    print(f"\n  ✓ 时序演化网络已全部运行完毕。导出结果可导入Gephi。")


if __name__ == "__main__":
    main()