import sys
sys.path.append("..")
import json
import logging
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import BertForMaskedLM

from util.saving import save_model, save_json
from util.arg_parser import ArgParser
from data import get_dataloader_for_mlm_bert, get_dataloader_for_distill_bert
from model import DistillBert

class run_mlm_bert_one_epoch():
    def __call__(self, config, model, dataloader, grad_update=False):
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


class run_distill_bert_one_epoch():
    def __call__(self, config, model, dataloader, grad_update=False):
        if grad_update:
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=config.lr)
            optimizer.zero_grad()
        else:
            model.eval()

        mse_loss_fn = torch.nn.MSELoss(reduction="sum")
        epoch_loss = 0
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch[0].to(config.device)
            attention_masks = batch[1].to(config.device)
            labels = batch[2].to(config.device)

            with torch.set_grad_enabled(grad_update):
                model_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                )

                batch_loss = mse_loss_fn(model_output, labels)
                epoch_loss +=  batch_loss.item()
                loss = (batch_loss / config.accum_step).requires_grad_()
                loss.backward()
                
                if grad_update: 
                    if ((step + 1) % config.accum_step == 0 ) or (step + 1 == len(dataloader)):
                        optimizer.step()
                        optimizer.zero_grad()

        return epoch_loss / len(dataloader)


def train(config):
    if config.type == "mlm_bert":
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
        run_one_epoch = run_mlm_bert_one_epoch()

    elif config.type == "distill_bert":
        train_loader = get_dataloader_for_distill_bert(
            model=config.model,
            task=config.task,
            hyp_text_path=config.train_hyp_text_path,
            batch_size=config.batch_size,
            hyp_score_path=config.train_hyp_score_path,
            shuffle=config.shuffle,
            max_utt=config.max_utt,
            n_best=config.n_best
        )

        dev_loader = get_dataloader_for_distill_bert(
            model=config.model,
            task=config.task,
            hyp_text_path=config.dev_hyp_text_path,
            batch_size=config.batch_size,
            hyp_score_path=config.dev_hyp_score_path,
            shuffle=config.shuffle,
            max_utt=config.max_utt,
            n_best=config.n_best
        )
        model = DistillBert(config.model)
        run_one_epoch = run_distill_bert_one_epoch()

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

    train(config)