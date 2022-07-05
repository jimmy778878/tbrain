import sys
sys.path.append("..")
import os
import json
import sys
from tqdm import tqdm
from util.saving import save_json

# 已經分割後的資料集路徑
train_input = "train.json"
dev_input = "dev.json"
test_input = "test.json"
inputs = [train_input, dev_input, test_input]

# 前處理完後的資料要保存的 root path
train_out_root = "train/"
dev_out_root = "dev/"
test_out_root = "test/"
outputs = [train_out_root, dev_out_root, test_out_root]


if __name__ == "__main__":
    for input_path, output_root in zip(inputs, outputs):
        with open(input_path, "r", encoding="utf-8") as in_file:
            input_json_data = json.load(in_file)

        output_jsons = {
            "ref_text": {},
            "hyps_text": {},
            "hyps_phone": {}
        }

        for utt in tqdm(input_json_data):

            utt_id = utt["id"]
            hyps_text = utt["sentence_list"]
            hyps_phone = utt["phoneme_sequence_list"]
            ref_text = utt["ground_truth_sentence"]
            
            output_jsons["ref_text"][utt_id] = ref_text
            output_jsons["hyps_text"][utt_id] = {}
            output_jsons["hyps_phone"][utt_id] = {}

            for hyp_id, (hyp_text, hyp_phone) in enumerate(zip(hyps_text, hyps_phone), start=1):
                hyp_text = hyp_text.replace(" ", "")
                output_jsons["hyps_text"][utt_id].update({f"hyp_{hyp_id}": hyp_text})
                output_jsons["hyps_phone"][utt_id].update({f"hyp_{hyp_id}": hyp_phone})

        for file_name, data in output_jsons.items():
            if not os.path.isdir(output_root):
                os.mkdir(output_root)
            save_json(f"{output_root}{file_name}.json", data)
