import os

from datasets import load_dataset

# 本地数据集根目录，用 HF_DATASET_ROOT 指定，缺省为当前目录下的 hug_dataset/
DATASET_ROOT = os.environ.get("HF_DATASET_ROOT", "hug_dataset")

# ✅ Correct: Specify the config name
dataset = load_dataset(os.path.join(DATASET_ROOT, "BLINK-Benchmark"))['validation']
print(dataset)
for i in range(len(dataset)):
    answer = dataset[i]['answer']
    image = dataset[i]['image_1']
    question = dataset[i]['question']
    print(dataset[i])
    import ipdb; ipdb.set_trace();