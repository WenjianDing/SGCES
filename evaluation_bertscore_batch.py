import os
import glob
import json
import numpy as np
import torch
from bert_score import score as bert_score


BIO_CLINICAL_BERT = ".../Bio_ClinicalBERT"



def compute_bioclinicalbert_bertscore(predictions, references):
    """
    使用 Bio_ClinicalBERT 计算 BERTScore。
    Bio_ClinicalBERT 是 BERT-base 架构，因此 num_layers=12。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    precision, recall, f1 = bert_score(
        cands=predictions,
        refs=references,
        model_type=BIO_CLINICAL_BERT,
        num_layers=12,
        idf=True,
        batch_size=16,
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

    try:
        filtered = [item for item in baseline_data]
        predictions = [item["predicted"] for item in filtered]
        references = [item["ground truth"] for item in filtered]
    except KeyError:
        filtered = [item for item in baseline_data]
        predictions = [item["best_impression"] for item in filtered]
        references = [item["ground_truth"] for item in filtered]

    if not predictions or not references:
        print(f"警告：文件 {file_name} 中没有找到合法的预测或参考数据，跳过。")
        return


    # Bio_ClinicalBERT BERTScore
    bertscore_result = compute_bioclinicalbert_bertscore(predictions, references)

    print("\n--- BERTScore using Bio_ClinicalBERT ---")
    #print(f"BERTScore Precision: {bertscore_result['precision']:.4f}")
    #print(f"BERTScore Recall:    {bertscore_result['recall']:.4f}")
    print(f"BERTScore F1:        {bertscore_result['f1']:.4f}")

    print(f"{'=' * 20} 文件 {file_name} 处理完毕 {'=' * 20}\n")


if __name__ == "__main__":
    folder_path = r"C:\Work\pricai\result\baseline\total"

    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print(f"指定的目录 {folder_path} 中没有找到任何 .json 文件。")
    else:
        print(f"共找到 {len(json_files)} 个 json 文件。")


        print("BERTScore 将使用 Bio_ClinicalBERT:")
        print(f"  {BIO_CLINICAL_BERT}\n")

        for file_path in json_files:
            calculate_metrics(file_path)