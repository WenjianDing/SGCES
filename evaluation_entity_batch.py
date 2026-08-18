import os
import glob
import json
import numpy as np
from radgraph import F1RadGraph


def compute_rg_er_precision_recall(hyp_ann, ref_ann):
    """
    复现 exact_entity_token_if_rel_exists_reward 的逻辑，
    额外返回 precision 和 recall（对应 rg_er / partial）
    """
    candidates = []
    for annotation_list in [hyp_ann, ref_ann]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            else:
                candidate.append((entity["tokens"], entity["label"], True))
        candidates.append(set(candidate))

    hyp_set, ref_set = candidates

    if len(hyp_set) == 0 and len(ref_set) == 0:
        return 0.0, 0.0, 0.0
    if len(hyp_set) == 0 or len(ref_set) == 0:
        return 0.0, 0.0, 0.0

    precision = sum(1 for x in hyp_set if x in ref_set) / len(hyp_set)
    recall = sum(1 for x in ref_set if x in hyp_set) / len(ref_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def calculate_entity(baseline_data_path, f1radgraph):
    # 提取并打印当前正在处理的文件名
    file_name = os.path.basename(baseline_data_path)
    print(f"\n{'=' * 20} 正在处理文件: {file_name} {'=' * 20}")

    with open(baseline_data_path, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)

    try:
        filtered = [item for item in baseline_data]
        predictions = [item["predicted"] for item in filtered]
        references = [item["ground truth"] for item in filtered]
    except KeyError:  # 明确捕获 KeyError 更加规范
        filtered = [item for item in baseline_data]
        predictions = [item["best_impression"] for item in filtered]
        references = [item["ground_truth"] for item in filtered]

    # 防止文件为空或数据提取失败报错
    if not predictions or not references:
        print(f"警告：文件 {file_name} 中没有找到合法的预测或参考数据，跳过。")
        return

    # 使用外部传入的模型实例进行预测
    mean_reward, reward_list, hyp_annotations, ref_annotations = f1radgraph(
        hyps=predictions, refs=references
    )

    rg_e, rg_er, rg_bar_er = mean_reward

    #print("--- RadGraph 全局评估结果 ---")
    #print(f"F1 (Entity only):        {rg_e:.4f}")
    #print(f"F1 (Entity + Relation):  {rg_er:.4f}  ← 论文中最常用")
    #print(f"F1 (Graph matching):     {rg_bar_er:.4f}\n")

    # 逐样本计算 precision / recall（针对 rg_er）
    per_sample_er = reward_list[1]  # 每个样本的 rg_er f1

    all_precision, all_recall = [], []
    #print("--- 单条明细 (rg_er) ---")
    for i, item in enumerate(filtered):
        p, r, f = compute_rg_er_precision_recall(hyp_annotations[i], ref_annotations[i])
        all_precision.append(p)
        all_recall.append(r)
        #print(f"UID: {item.get('uid', i)}")
        #print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f:.4f}  (lib F1: {per_sample_er[i]:.4f})")

    print("\n--- 全局 Precision / Recall (rg_er, macro-average) ---")
    print(f"Mean Precision: {np.mean(all_precision):.4f}")
    print(f"Mean Recall:    {np.mean(all_recall):.4f}")
    print(f"Mean F1:        {np.mean(per_sample_er):.4f}")
    print(f"{'=' * 20} 文件 {file_name} 处理完毕 {'=' * 20}\n")


if __name__ == '__main__':
    # 1. 指定包含 json 文件的文件夹路径
    folder_path = r"C:\Work\pricai\result\mimic"

    # 获取文件夹下所有以 .json 结尾的文件列表
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print(f"指定的目录 {folder_path} 中没有找到任何 .json 文件。")
    else:
        print(f"共找到 {len(json_files)} 个 json 文件。")

        # 2. 提前在循环外加载模型，避免在每次循环中重复加载导致耗时过长
        print("正在加载 RadGraph 模型 (这可能需要一些时间)...")
        f1radgraph_model = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        print("模型加载完成，开始处理文件。\n")

        # 3. 遍历并执行所有文件
        for file_path in json_files:
            calculate_entity(file_path, f1radgraph_model)