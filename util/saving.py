import os
import json
import torch

def save_model(file_path: str, model_dict, checkpoint_num: int):
    torch.save(
        model_dict,
        os.path.join(file_path, f"checkpoint_{checkpoint_num}.pth")
    )


def save_json(file_path: str, json_data: dict):
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)