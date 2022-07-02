import sys
sys.path.append("..")
import json
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import BertForMaskedLM

from util.saving import save_model, save_json
from util.arg_parser import ArgParser
from data import get_dataloader_for_mlm_bert


def run_one_epoch(config, model, dataloader, grad_update=False):
    if grad_update:
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        optimizer.zero_grad()
    else:
        model.eval()

    epoch_loss = 0
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = batch[0].to(config.device)
        attention_masks = batch[1].to(config.device)
        labels = batch[2].to(config.device)

        with torch.set_grad_enabled(grad_update):
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels,
                return_dict=True
            )

            epoch_loss +=  model_output.loss.item()
            loss = (model_output.loss / config.accum_step).requires_grad_()
            loss.backward()
            
            if grad_update: 
                if ((step + 1) % config.accum_step == 0 ) or (step + 1 == len(dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()

    return epoch_loss / len(dataloader)
    

def run_score(config, model, dataloader, output_template):
    model.eval()
    for batch in tqdm(dataloader, total=len(dataloader)):
        utt_id = batch[0]
        hyp_id = batch[1]
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
        for u_id, h_id, score in zip(utt_id, hyp_id, token_score):
            output_template[u_id][h_id] += score
    return output_template


def train(config):
    train_loader = get_dataloader_for_mlm_bert(
        model=config.model,
        task=config.task,
        data_path=config.train_ref_text_path,
        masking_strategy=config.masking_strategy,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        max_utt=config.max_utt,
    )

    dev_loader = get_dataloader_for_mlm_bert(
        model=config.model,
        task=config.task,
        data_path=config.dev_ref_text_path,
        masking_strategy=config.masking_strategy,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        max_utt=config.max_utt,
    )

    model = BertForMaskedLM.from_pretrained(config.model)

    if config.resume.epoch_id != None and config.resume.checkpoint_path != None:
        resume = True
        checkpoint = torch.load(config.resume.checkpoint_path)
        model.load_state_dict(checkpoint)
        loss_record = json.load(
            open(f"{config.output_path}/epoch_loss.json", "r", encoding="utf-8")
        )
        train_loss_record = loss_record["train"]
        dev_loss_record = loss_record["dev"]
    else:
        resume = False
        train_loss_record = []
        dev_loss_record = []

    model = model.to(config.device)
    
    for epoch_id in range(config.resume.epoch_id + 1 if resume else 1, config.epoch + 1):
        logging.info(f"Epoch {epoch_id}/{config.epoch}")
        
        train_loss = run_one_epoch(
            config=config,
            model=model,
            dataloader=train_loader,
            grad_update=True,
        )
        logging.info(f"epoch {epoch_id} train loss: {train_loss}")
        train_loss_record.append(train_loss)

        dev_loss = run_one_epoch(
            config=config,
            model=model,
            dataloader=dev_loader,
            grad_update=False,
        )
        logging.info(f"epoch {epoch_id} dev loss: {dev_loss}")
        dev_loss_record.append(dev_loss)
        
        save_model(config.output_path, model.state_dict(), epoch_id)
        save_json(
            f"{config.output_path}/epoch_loss.json",
            {"train": train_loss_record, "dev": dev_loss_record}
        )


def score(config):
    dev_loader = get_dataloader_for_mlm_bert(
        model=config.model,
        task=config.task,
        data_path=config.dev_hyp_text_path,
        masking_strategy="one_by_one",
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = get_dataloader_for_mlm_bert(
        model=config.model,
        task=config.task,
        data_path=config.test_hyp_text_path,
        masking_strategy="one_by_one",
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = BertForMaskedLM.from_pretrained(config.model)
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to(config.device)
    
    dev_output_template = json.load(
        open(f"{config.dev_output_template}", "r", encoding="utf-8")
    )

    test_output_template = json.load(
        open(f"{config.test_output_template}", "r", encoding="utf-8")
    )

    dev_output = run_one_epoch(
        config=config,
        model=model,
        dataloader=dev_loader,
        output_template=dev_output_template,
    )
    save_json(f"{config.output_path}/dev_lm.json", dev_output)

    test_output = run_score(
        config=config,
        model=model,
        dataloader=test_loader,
        output_template=test_output_template,
    )
    save_json(f"{config.output_path}/test_lm.json", test_output)
    return


if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()

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

    if config.task == "training":
        train(config)
    elif config.task == "scoring":
        score(config)