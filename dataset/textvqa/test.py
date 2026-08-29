import os

from datasets import load_dataset

# 本地数据集根目录，用 HF_DATASET_ROOT 指定，缺省为当前目录下的 hug_dataset/
DATASET_ROOT = os.environ.get("HF_DATASET_ROOT", "hug_dataset")

# ✅ Correct: Specify the config name
dataset = load_dataset(os.path.join(DATASET_ROOT, "lmms-lab/TextVQA"))['validation']
for i in range(len(dataset)):
    question = dataset[i]['question']
    image = dataset[i]['image']
    answers = dataset[i]['answers']
    # classes = dataset[i]['image_classes']
    print(image)
    # import ipdb; ipdb.set_trace()