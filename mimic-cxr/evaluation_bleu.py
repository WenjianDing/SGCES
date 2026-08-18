import json
import sys
sys.path.append("C:/Work/TextbookQA/nlg-eval-master")
from nlgeval import compute_metrics


def generate_qad(total_result_path, prediction_path, gt_path):
    total_count = 0
    with open(total_result_path, 'r', encoding='utf-8') as f1:
        total_result = json.load(f1)
    file1 = open(prediction_path, mode='w', encoding='utf-8')
    file2 = open(gt_path, mode='w', encoding='utf-8')
    for each_result in total_result:
        uid = int(each_result['uid'])
        #if uid not in normal_id_list:
        if True:
            try:
                predict_raw = each_result['predicted']
                gt_raw = each_result['ground truth']
            except:
                predict_raw = each_result['best_impression']
                gt_raw = each_result['ground_truth']

            # try:
            #     predict_raw = each_result['best_impression']
            #     gt_raw = each_result['ground_truth']
            # except:
            #     predict_raw = each_result['final_pred']
            #     gt_raw = each_result['ground_truth']
            predict = predict_raw.replace('\n','')
            gt = gt_raw.replace('\n', '')
            #print(normal_id_list)
            # if int(uid) in normal_id_list:
            file1.write(predict)
            file1.write('\n')
            file2.write(gt)
            file2.write('\n')
            total_count += 1

    print(len(total_result))
    print(total_count)



if __name__ == '__main__':
    #total_result_path = r"C:\Work\pricai\result\baseline\gpt5mini_baseline.json"
    total_result_path = r"C:\Work\pricai\result\ours\gpt5mini_rag_symbolic_226_results.json"

    prediction_path = r"C:\Work\pricai\result\prediction.txt"
    gt_path = r"C:\Work\pricai\result\groundtruth.txt"
    generate_qad(total_result_path, prediction_path, gt_path)
    metrics_dict = compute_metrics(hypothesis=prediction_path, references=[gt_path])
