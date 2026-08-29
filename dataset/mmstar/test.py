import os

from datasets import load_dataset

# 本地数据集根目录，用 HF_DATASET_ROOT 指定，缺省为当前目录下的 hug_dataset/
DATASET_ROOT = os.environ.get("HF_DATASET_ROOT", "hug_dataset")

# ✅ Correct: Specify the config name
dataset = load_dataset(os.path.join(DATASET_ROOT, "Lin-Chen/MMStar"))['val']
print(dataset)
import ipdb;ipdb.set_trace()
# 单选
for i in range(len(dataset)):
    question = dataset[i]['question']
    image = dataset[i]['image']
    answer = dataset[i]['answer']
    # print(answer)
    import ipdb;ipdb.set_trace()