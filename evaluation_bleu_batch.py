import os
import glob
import json
import sys

# 确保 nlg-eval 的路径正确添加
sys.path.append("C:/Work/TextbookQA/nlg-eval-master")
# 改为直接引入 NLGEval 类，而不是用 compute_metrics 这个无脑封装
from nlgeval import NLGEval


def evaluate_single_file(json_path, nlg_eval_model):
    """
    处理单个 json 文件，直接在内存中提取数据并评估，不再使用临时文件读写。
    """
    file_name = os.path.basename(json_path)
    print(f"\n{'=' * 20} 正在处理文件: {file_name} {'=' * 20}")

    with open(json_path, 'r', encoding='utf-8') as f:
        total_result = json.load(f)

    if not total_result:
        print(f"警告：文件 {file_name} 为空，已跳过。")
        return

    pred_list = []
    gt_list = []

    for each_result in total_result:
        # 兼容不同的 key 命名格式
        try:
            predict_raw = each_result['predicted']
            gt_raw = each_result['ground truth']
        except KeyError:
            try:
                predict_raw = each_result['best_impression']
                gt_raw = each_result['ground_truth']
            except KeyError:
                predict_raw = each_result.get('final_pred', '')
                gt_raw = each_result.get('ground_truth', '')

        # 清理文本
        predict = predict_raw.replace('\n', '').strip()
        gt = gt_raw.replace('\n', '').strip()

        # 只要有一方非空就加入列表
        if predict or gt:
            pred_list.append(predict)
            gt_list.append(gt)

    if not pred_list:
        print(f"警告：文件 {file_name} 中未提取到有效数据，已跳过。")
        return

    # 直接在内存中传递列表进行评估
    # nlg_eval 要求 ref_list 是一个嵌套列表 (用于支持多参考答案)，hyp_list 是单层列表
    metrics_dict = nlg_eval_model.compute_metrics(ref_list=[gt_list], hyp_list=pred_list)

    # 提取你指定的 4 个指标
    bleu_4 = metrics_dict.get('Bleu_4', 0.0)
    meteor = metrics_dict.get('METEOR', 0.0)
    rouge_l = metrics_dict.get('ROUGE_L', 0.0)
    cider = metrics_dict.get('CIDEr', 0.0)

    # 仅输出指定的指标
    print("--- 评估结果 ---")
    print(f"BLEU-4:  {bleu_4:.4f}")
    print(f"METEOR:  {meteor:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")
    print(f"CIDEr:   {cider:.4f}")
    print(f"{'=' * 20} 文件 {file_name} 处理完毕 {'=' * 20}\n")


if __name__ == '__main__':
    # 指定包含 json 文件的文件夹路径
    folder_path = r"C:\Work\pricai\result\mimic"

    # 获取文件夹下所有以 .json 结尾的文件列表
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print(f"指定的目录 {folder_path} 中没有找到任何 .json 文件。")
    else:
        print(f"共找到 {len(json_files)} 个 json 文件。")

        # 【核心修复与优化区】
        # 1. 显式禁用 SPICE (metrics_to_omit=['SPICE']) 解决 Java 报错问题
        # 2. 禁用 GloVe 和 SkipThought 节省内存并极大地加速运行
        # 3. 实例化放在循环外，100个文件也只需要加载1次模型！
        print("正在初始化 NLG Evaluator (已跳过 SPICE 和词向量模型)...")
        nlg_evaluator = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['SPICE'])
        print("初始化完成，开始批量处理...\n")

        # 遍历并执行所有文件
        for file_path in json_files:
            evaluate_single_file(file_path, nlg_evaluator)

        print("所有文件处理完毕！")