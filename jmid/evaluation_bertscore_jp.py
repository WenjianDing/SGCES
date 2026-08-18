# coding=utf-8
import os
import glob
import json
import csv
import torch
from bert_score import score as bert_score


# ==================== 配置区 ====================

# 推荐用于日文 BERTScore 的多语言模型
#JAPANESE_BERTSCORE_MODEL = "xlm-roberta-large"

# 如果你已经下载到本地，也可以改成本地路径，例如：
JAPANESE_BERTSCORE_MODEL = r"...\xlm-roberta-large"

BATCH_SIZE = 16

SUMMARY_SAVE_PATH = r"C:\Work\pricai\result\jmid\japanese_bertscore_summary.csv"

# =================================================


def normalize_text(text):
    text = "" if text is None else str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def get_prediction_and_reference(item):
    if "predicted" in item and "ground truth" in item:
        return item["predicted"], item["ground truth"]

    if "best_impression" in item and "ground_truth" in item:
        return item["best_impression"], item["ground_truth"]

    if "final_pred" in item and "ground_truth" in item:
        return item["final_pred"], item["ground_truth"]

    raise KeyError("Cannot find prediction and ground-truth fields.")


def compute_japanese_bertscore(predictions, references):
    """
    使用 xlm-roberta-large 计算日文 BERTScore。
    xlm-roberta-large 支持多语言，包括日文。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    precision, recall, f1 = bert_score(
        cands=predictions,
        refs=references,
        model_type=JAPANESE_BERTSCORE_MODEL,
        num_layers=17,
        lang="ja",
        idf=True,
        batch_size=BATCH_SIZE,
        device=device,
        rescale_with_baseline=False,
        verbose=False
    )

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item()
    }


def calculate_metrics(baseline_data_path):
    file_name = os.path.basename(baseline_data_path)
    print(f"\n{'=' * 20} 正在处理文件: {file_name} {'=' * 20}")

    with open(baseline_data_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    predictions = []
    references = []

    for idx, item in enumerate(baseline_data):
        try:
            pred, ref = get_prediction_and_reference(item)
        except KeyError as e:
            print(f"跳过第 {idx} 条样本，原因: {e}")
            continue

        pred = normalize_text(pred)
        ref = normalize_text(ref)

        if pred == "" or ref == "":
            continue

        predictions.append(pred)
        references.append(ref)

    if not predictions or not references:
        print(f"警告：文件 {file_name} 中没有找到合法的预测或参考数据，跳过。")
        return None

    bertscore_result = compute_japanese_bertscore(
        predictions,
        references
    )

    print("\n--- BERTScore using xlm-roberta-large ---")
    print(f"BERTScore Precision: {bertscore_result['precision']:.4f}")
    print(f"BERTScore Recall:    {bertscore_result['recall']:.4f}")
    print(f"BERTScore F1:        {bertscore_result['f1']:.4f}")

    print(f"{'=' * 20} 文件 {file_name} 处理完毕 {'=' * 20}\n")

    return {
        "filename": file_name,
        "file_path": baseline_data_path,
        "num_samples": len(predictions),
        "bertscore_precision": bertscore_result["precision"],
        "bertscore_recall": bertscore_result["recall"],
        "bertscore_f1": bertscore_result["f1"]
    }


def save_summary_csv(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "filename",
        "file_path",
        "num_samples",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1"
    ]

    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({
                "filename": row["filename"],
                "file_path": row["file_path"],
                "num_samples": row["num_samples"],
                "bertscore_precision": f"{row['bertscore_precision']:.6f}",
                "bertscore_recall": f"{row['bertscore_recall']:.6f}",
                "bertscore_f1": f"{row['bertscore_f1']:.6f}"
            })


if __name__ == "__main__":
    folder_path = r"C:\Work\pricai\result\jmid"

    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print(f"指定的目录 {folder_path} 中没有找到任何 .json 文件。")
    else:
        print(f"共找到 {len(json_files)} 个 json 文件。")
        print("BERTScore 将使用日文兼容的多语言模型:")
        print(f"  {JAPANESE_BERTSCORE_MODEL}\n")

        all_results = []

        for file_path in json_files:
            result = calculate_metrics(file_path)

            if result is not None:
                all_results.append(result)

        if all_results:
            save_summary_csv(all_results, SUMMARY_SAVE_PATH)
            print(f"\n所有文件的 BERTScore 汇总已保存至: {SUMMARY_SAVE_PATH}")