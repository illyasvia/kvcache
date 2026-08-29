import os

from datasets import load_dataset

# 本地数据集根目录，用 HF_DATASET_ROOT 指定，缺省为当前目录下的 hug_dataset/
DATASET_ROOT = os.environ.get("HF_DATASET_ROOT", "hug_dataset")

# ✅ Correct: Specify the config name
dataset = load_dataset(os.path.join(DATASET_ROOT, "lmms-lab/MMMU"))['validation']
print(dataset)
import ipdb;ipdb.set_trace()
for i in range(len(dataset)):
    question = dataset[i]['question']
    instruction = "options:" + str(dataset[i]['options'])
    images = []
    for j in range(1,8):
        image = dataset[i][f'image_{j}']
        if image is not None:
            images.append(image)
    print(dataset[j]['question_type']) # multiple-choice
    answer = dataset[i]['answer']
    print(dataset[i])
    import ipdb;ipdb.set_trace()