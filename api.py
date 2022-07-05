from argparse import ArgumentParser
import datetime
import hashlib
import time
import logging

from flask import Flask
from flask import request
from flask import jsonify

import numpy as np

from model.model import DistillBert 
from transformers import BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = ''                      #
SALT = 'my_salt'                        #
#########################################


def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def predict(sentence_list, phoneme_sequence_list):
    """ Predict your model result.

    @param:
        sentence_list (list): an list of sentence sorted by probability.
        phoneme_sequence_list (list): an list of phoneme sequence sorted by probability.
    @returns:
        prediction (str): a sentence.
    """

    # 拿到 sentence 後，把空格去掉
    sentence_list_remove_space = []
    for sentence in sentence_list:
        sentence_list_remove_space.append(sentence.replace(" ", ""))

    # 把所有 hypothesis 轉成 model 的 input_ids 以及 attention mask
    input_ids = []
    attention_masks = []
    for sentence in sentence_list_remove_space:
        token_seq = tokenizer.tokenize(sentence)
        id_seq = tokenizer.convert_tokens_to_ids(
            [tokenizer.cls_token] + token_seq + [tokenizer.sep_token]
        )
        input_ids.append(torch.tensor(id_seq, dtype=torch.long))
        attention_mask = [1] * len(id_seq)
        attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))

    # 由於 hypothesis 之間長度不一，整理成一個 batch 時要先進行 padding 才能送入模型
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)

    lm_score = model(
        input_ids=input_ids,
        attention_mask=attention_masks
    )

    # 取最高分的句子回傳
    pred_id = torch.argmax(lm_score)
    prediction = sentence_list_remove_space[pred_id]

    if _check_datatype_to_string(prediction):
        return prediction


def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)
    logging.info(f"get return data: {data}")
    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 sentence list 中文
    sentence_list = data['sentence_list']
    # 取 phoneme sequence list (X-SAMPA)
    phoneme_sequence_list = data['phoneme_sequence_list']

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))

    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    current_time = time.time()
    if current_time - esun_timestamp < 2:
        try:
            answer = predict(sentence_list, phoneme_sequence_list)
        except TypeError as type_error:
            # You can write some log...
            raise type_error
        except Exception as e:
            # You can write some log...
            raise e
    else:
        # 如果已經超時就不要再浪費時間 inference
        answer = "超時了啦"
    
    server_return_timestamp = time.time()
    logging.info( "\n"
        "esun_timestamp: " + str(esun_timestamp) + "\n"
        "sentence list: " + str(sentence_list) + "\n"
        "phoneme list: " + str(phoneme_sequence_list) + "\n"
        "prediction: " + str(answer) + "\n"
        "server return timestamp: " + str(server_return_timestamp) +"\n"
    )

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_return_timestamp})


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    arg_parser.add_argument('--checkpoint_path', required=True)
    options = arg_parser.parse_args()
    
    # 預先載入 model 和 tokenizer
    print("Loading tokenizer and model...")    
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = DistillBert("bert-base-chinese")
    checkpoint = torch.load(options.checkpoint_path, map_location=torch.device('cpu'))   
    model.load_state_dict(checkpoint)
    model.eval()
    print("Loading finish.")

    logging.basicConfig(
        filename="api.log",
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    app.run(host='0.0.0.0', port=options.port, debug=options.debug)
