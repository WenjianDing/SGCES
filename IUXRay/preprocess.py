import json
import csv
from collections import defaultdict

file_path = r"C:\Work\pricai\dataset\indiana_projections.csv"

# 使用 defaultdict 可以自动处理不存在的键
result_dict = defaultdict(dict)

with open(file_path, mode='r', encoding='utf-8') as f:
    # DictReader 会自动将第一行作为字典的 key
    reader = csv.DictReader(f)
    for row in reader:
        # 提取每一列的数据
        # 注意：从csv读出的uid默认是字符串，如果您希望它是整数，可以加上 int()
        uid = int(row['uid'])
        filename = row['filename']
        projection = row['projection']
        # 组合成目标字典结构
        result_dict[uid][projection] = filename
# 将 defaultdict 转换回普通的 dict（可选）
id2img_dict = dict(result_dict)
normal_uid = []
final_data_list = []
count = 0
# 读取印第安纳报告数据集
with open(r"C:\Work\pricai\dataset\indiana_reports.csv", mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # csv读出来的数据都是字符串，将 uid 转换为整数以匹配 image_dict 的 key
        uid = int(row['uid'])

        # 获取该 uid 对应的图像信息，如果该患者没有图片，则返回空字典作为默认值
        images = id2img_dict.get(uid, {})

        # 直接在原有的字典上添加这两个新 key-value 对
        # 如果没有对应的图片，可以默认填入 None 或者空字符串 ''
        row['Frontal'] = images.get('Frontal', None)
        row['Lateral'] = images.get('Lateral', None)
        problem = row['Problems']
        findings = row['findings']
        impression = row['impression']
        if problem == 'normal':
            count += 1
            normal_uid.append(uid)
        if findings != "" and impression != "":
            # if problem != "normal":
            #     # 将组装好的字典添加到最终的列表中
            #     final_data_list.append(row)

            # 将组装好的字典添加到最终的列表中
            final_data_list.append(row)

print(count)
print('total data length', len(final_data_list))
print(count/len(final_data_list))
import json
output_filename = 'ct_data.json'
print(normal_uid)
with open(output_filename, 'w', encoding='utf-8') as f:
    # ensure_ascii=False 确保如果文本中有特殊字符或非英文字符时能正常显示
    # indent=4 则是让生成的 json 文件具有良好的排版和缩进，方便肉眼查看
    json.dump(final_data_list, f, ensure_ascii=False, indent=4)