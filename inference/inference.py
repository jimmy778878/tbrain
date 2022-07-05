import sys
sys.path.append("..")
import json
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from transformers import BertForMaskedLM

from util.saving import save_json
from util.arg_parser import ArgParser
from model.data import get_dataloader_for_mlm_bert, get_dataloader_for_distill_bert
from model.model import DistillBert
from jiwer import cer


def compute_error_rate(hyp, ref, max_utt):
    error_rate = cer(list(ref.values())[:max_utt], hyp[:max_utt])
    return error_rate


def get_top_one_text(score, text):
    top_one_text = []
    for hyps_score, hyps_text in zip(score.values(), text.values()):
        hyps_score = list(hyps_score.values())
        hyps_text = list(hyps_text.values())
        max_score = max(hyps_score)
        top_one_hyp_index = hyps_score.index(max_score)
        top_one_text.append(hyps_text[top_one_hyp_index])
    return top_one_text


class run_mlm_bert_score():
    def __call__(self, config, model, dataloader):
        output = {}
        model.eval()
        for batch in tqdm(dataloader, total=len(dataloader)):
            utt_ids = batch[0]
            hyp_ids = batch[1]
            input_ids = batch[2].to(config.device)
            attention_masks = batch[3].to(config.device)
            mask_pos = batch[4]
            masked_token_id = batch[5]

            with torch.set_grad_enabled(False):
                model_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    return_dict=True
                )

            token_logits = model_output.logits[range(len(model_output.logits)), mask_pos, :]
            token_score = token_logits.log_softmax(dim=-1)
            token_score = token_score[range(len(token_score)), masked_token_id].tolist()

            for utt_id, hyp_id, score in zip(utt_ids, hyp_ids, token_score):
                if utt_id not in output.keys():
                    output[utt_id] = {}
                if hyp_id not in output[utt_id].keys():
                    output[utt_id][hyp_id] = 0
                output[utt_id][hyp_id] += score

        return output


class run_distill_bert_score():
    def __call__(self, config, model, dataloader):
        output = {}
        model.eval()
        for batch in tqdm(dataloader, total=len(dataloader)):
            utt_ids = batch[0]
            hyp_ids = batch[1]
            input_ids = batch[2].to(config.device)
            attention_masks = batch[3].to(config.device)

            with torch.set_grad_enabled(False):
                lm_score = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                )

            for utt_id, hyp_id, s in zip(utt_ids, hyp_ids, lm_score):
                if utt_id not in output.keys():
                    output[utt_id] = {}
                if hyp_id not in output[utt_id].keys():
                    output[utt_id][hyp_id] = 0
                output[utt_id][hyp_id] += s.item()

        return output


def score(config):
    if config.type == "mlm_bert":
        train_loader = None
        if config.train_hyp_text_path != None:
            train_loader = get_dataloader_for_mlm_bert(
                model=config.model,
                task=config.task,
                data_path=config.train_hyp_text_path,
                masking_strategy="one_by_one",
                batch_size=config.batch_size,
                shuffle=False,
                max_utt=config.max_utt,
                n_best=config.n_best
            )

        dev_loader = get_dataloader_for_mlm_bert(
            model=config.model,
            task=config.task,
            data_path=config.dev_hyp_text_path,
            masking_strategy="one_by_one",
            batch_size=config.batch_size,
            shuffle=False,
            max_utt=config.max_utt,
            n_best=config.n_best
        )

        test_loader = get_dataloader_for_mlm_bert(
            model=config.model,
            task=config.task,
            data_path=config.test_hyp_text_path,
            masking_strategy="one_by_one",
            batch_size=config.batch_size,
            shuffle=False,
            max_utt=config.max_utt,
            n_best=config.n_best
        )
        model = BertForMaskedLM.from_pretrained(config.model)
        run_score = run_mlm_bert_score()
    
    elif config.type == "distill_bert":
        train_loader = None

        dev_loader = get_dataloader_for_distill_bert(
            model=config.model,
            task=config.task,
            hyp_text_path=config.dev_hyp_text_path,
            batch_size=config.batch_size,
            shuffle=False,
            max_utt=config.max_utt,
            n_best=config.n_best
        )

        test_loader = get_dataloader_for_distill_bert(
            model=config.model,
            task=config.task,
            hyp_text_path=config.test_hyp_text_path,
            batch_size=config.batch_size,
            shuffle=False,
            max_utt=config.max_utt,
            n_best=config.n_best
        )
        model = DistillBert(config.model)
        run_score = run_distill_bert_score()

    if config.checkpoint_path != None:
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint)
    model = model.to(config.device)

    # 對 training set 的 hypothesis 進行評分
    if train_loader != None:
        train_output = run_score(
            config=config,
            model=model,
            dataloader=train_loader
        )
        save_json(f"{config.output_path}/train_lm.json", train_output)
        top_one_text = get_top_one_text(
            score=train_output,
            text=json.load(open(f"{config.train_hyp_text_path}", "r", encoding="utf-8"))
        )
        error_rate = compute_error_rate(
            hyp=top_one_text,
            ref=json.load(open(f"{config.train_ref_text_path}", "r", encoding="utf-8")),
            max_utt=config.max_utt
        )
        logging.info(f"training set error rate: {error_rate}")

    # 對 developing set 的 hypothesis 進行評分
    dev_output = run_score(
        config=config,
        model=model,
        dataloader=dev_loader
    )
    save_json(f"{config.output_path}/dev_lm.json", dev_output)
    top_one_text = get_top_one_text(
        score=dev_output,
        text=json.load(open(f"{config.dev_hyp_text_path}", "r", encoding="utf-8"))
    )
    error_rate = compute_error_rate(
        hyp=top_one_text,
        ref=json.load(open(f"{config.dev_ref_text_path}", "r", encoding="utf-8")),
        max_utt=config.max_utt
    )
    logging.info(f"developing set error rate: {error_rate}")

    # 對 testing set 的 hypothesis 進行評分
    test_output = run_score(
        config=config,
        model=model,
        dataloader=test_loader
    )
    save_json(f"{config.output_path}/test_lm.json", test_output)
    top_one_text = get_top_one_text(
        score=test_output,
        text=json.load(open(f"{config.test_hyp_text_path}", "r", encoding="utf-8"))
    )
    error_rate = compute_error_rate(
        hyp=top_one_text,
        ref=json.load(open(f"{config.test_ref_text_path}", "r", encoding="utf-8")),
        max_utt=config.max_utt
    )
    logging.info(f"testing set error rate: {error_rate}")
    return


if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()

    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{config.output_path}/{config.task}.log",
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    if config.seed != None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    score(config)