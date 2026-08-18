# coding=utf-8
import json
import os
import re
import sys

sys.path.append("C:/Work/TextbookQAD/nlg-eval-master")
from nlgeval import compute_metrics


try:
    from fugashi import Tagger
    JA_TAGGER = Tagger()
except Exception:
    JA_TAGGER = None


def normalize_text(text):
    text = "" if text is None else str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ja_tokenize(text):
    text = normalize_text(text)

    if text == "":
        return ""

    if JA_TAGGER is not None:
        return " ".join([word.surface for word in JA_TAGGER(text)])

    return " ".join(list(text.replace(" ", "")))


def get_prediction_and_gt(each_result):
    if "predicted" in each_result and "ground truth" in each_result:
        return each_result["predicted"], each_result["ground truth"]

    if "best_impression" in each_result and "ground_truth" in each_result:
        return each_result["best_impression"], each_result["ground_truth"]

    if "final_pred" in each_result and "ground_truth" in each_result:
        return each_result["final_pred"], each_result["ground_truth"]

    raise KeyError("Cannot find prediction / ground-truth fields in result item.")


def generate_qad(total_result_path, prediction_path, gt_path):
    with open(total_result_path, "r", encoding="utf-8") as f:
        total_result = json.load(f)

    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)

    total_count = 0

    with open(prediction_path, mode="w", encoding="utf-8") as pred_file, \
            open(gt_path, mode="w", encoding="utf-8") as gt_file:

        for each_result in total_result:
            try:
                predict_raw, gt_raw = get_prediction_and_gt(each_result)
            except Exception as e:
                print(f"Skip one item because of missing fields: {e}")
                continue

            predict = ja_tokenize(predict_raw)
            gt = ja_tokenize(gt_raw)

            pred_file.write(predict + "\n")
            gt_file.write(gt + "\n")
            total_count += 1

    print(f"Total results: {len(total_result)}")
    print(f"Valid evaluated results: {total_count}")


if __name__ == "__main__":
    total_result_path = r"C:\Work\pricai\result\jmid\jmid_medgemma_ours_results.json"
    prediction_path = r"C:\Work\pricai\result\prediction.txt"
    gt_path = r"C:\Work\pricai\result\groundtruth.txt"

    generate_qad(total_result_path, prediction_path, gt_path)

    metrics_dict = compute_metrics(
        hypothesis=prediction_path,
        references=[gt_path]
    )

    print(metrics_dict)

    # metrics_save_path = r"C:\Work\pricai\result\jmid\nlg_metrics_japanese.json"
    # with open(metrics_save_path, "w", encoding="utf-8") as f:
    #     json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    #
    # print(f"Metrics saved to: {metrics_save_path}")