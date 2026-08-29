import os

# 以脚本自身位置定位数据，避免依赖运行时的当前路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 指定目录路径
directory = BASE_DIR
# 检查目录是否存在
if not os.path.exists(directory):
    raise FileNotFoundError(f"目录不存在: {directory}")
if not os.path.isdir(directory):
    raise NotADirectoryError(f"路径不是一个目录: {directory}")

# all_entries = os.listdir(directory)
# print(all_entries)
# json_files = [f for f in all_entries if f.lower().endswith('.json')]


json_files = []
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        if filename.lower().endswith('.json'):
            full_path = os.path.join(dirpath, filename)
            json_files.append(full_path)
print("递归找到的所有 .json 文件完整路径:")


# 打印结果
print("找到的 JSON 文件:")
for file in json_files:
    print(file)

import json

data_list = {}
cnt = 0
for file in json_files:
    if cnt > 2000:
        break
    # file_path = os.path.join(directory, file)
    file_path = file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
            data[str(0)]['prediction']
            for key in data.keys():
                data_list[str(cnt)] = data[key]
                cnt += 1
            print(f"✅ 成功加载: {file}")
    except Exception as e:
        print(f"❌ 加载失败 {file}: {e}")

print(len(data_list))

output_file = os.path.join(BASE_DIR, "processed_data.json")
# 将 data_list 保存为 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)


