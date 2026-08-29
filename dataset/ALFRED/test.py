import json
import os

# 以脚本自身位置定位数据，避免依赖运行时的当前路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.dirname(BASE_DIR)

# 打开并读取 JSON 文件
with open(os.path.join(BASE_DIR, 'ALFRED.json'), 'r', encoding='utf-8') as file:
    data = json.load(file)

path = os.path.join(DATASET_DIR, "MMCoQA", "combined_1_images")
# instruction = data['meta_data']['task_instruction'][0]
for item in data['data']:
    print(item['task_instance'])
    prompt = item['task_instance']['context']
    image = item['task_instance']['combined_1_images'][0]
    image = os.path.join(path, image)
    response = item['response']
    import ipdb; ipdb.set_trace()

for i in item['task_instance'].keys():
    print(i)
# print(data['meta_data']['task_instruction'][0])