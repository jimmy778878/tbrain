# 分割資料集，將完整的資料集切成 training, developing, testing sets。
import sys
sys.path.append("..")
import random
import json
from util.saving import save_json

dev_ratio = 0.10
test_ratio = 0.05
seed = 42

input = "train_all.json"
train_output = "train.json"
dev_output = "dev.json"
test_output = "test.json"
outputs = [train_output, dev_output, test_output]


if __name__ == "__main__":
    all_data = json.load(
        open(input, "r", encoding="utf-8")
    )

    total_data_num = len(all_data)
    dev_num = int(total_data_num * dev_ratio)
    test_num = int(total_data_num * test_ratio)
    train_num = total_data_num - dev_num - test_num

    # 把資料打亂後再依序切割出三份。
    random.seed(seed)
    random.shuffle(all_data)
    train_data = all_data[:train_num]
    dev_data = all_data[train_num: train_num + dev_num]
    test_data = all_data[train_num + dev_num:]

    output_data = [train_data, dev_data, test_data]
    for output_path, json_data in zip(outputs, output_data):
        save_json(output_path, json_data)