import os
import json
import random
from sklearn.model_selection import train_test_split

# 以脚本自身位置定位数据，避免依赖运行时的当前路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(BASE_DIR, 'processed_data.json')
data_list = []

def transfome_dict(data):
    cnt = 0
    data_list = {}
    for i in range(len(data)):
        data_list[str(cnt)] = data[i]
        cnt += 1
    return data_list

def split_dataset(data, val_ratio: float = 0.1, test_ratio: float = 0.1, shuffle: bool = True, seed: int = 42):
    data_value = list(data.values())
    # print(type(data_value))
    if shuffle:
        random.seed(seed)
        random.shuffle(data_value)
    test_size = test_ratio
    val_size = val_ratio / (1 - test_ratio)  # 在剩余中再分 val

    train_val, test = train_test_split(data_value, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)
    train = transfome_dict(train)
    print(len(train))
    val = transfome_dict(val)
    test = transfome_dict(test)
    
    return {
        'train': train,
        'val': val,
        'test': test
    }

with open(file_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
        data = split_dataset(data)
        with open(os.path.join(BASE_DIR, 'train.json'), 'w', encoding='utf-8') as f:
            json.dump(data['train'], f, ensure_ascii=False, indent=2)

        with open(os.path.join(BASE_DIR, 'test.json'), 'w', encoding='utf-8') as f:
            json.dump(data['test'], f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(BASE_DIR, 'val.json'), 'w', encoding='utf-8') as f:
            json.dump(data['val'], f, ensure_ascii=False, indent=2)