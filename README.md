# tbrain

## 程式架構
```
.
│  .gitignore
│  api.py
│  README.md
│  requirements.txt
│  
├─data_preprocess
│  │  data_preprocess.py
│  └─ split.py
│          
├─inference
│  │  inference.py
│  │  
│  └─ configs
│                  
├─model
│  │  data.py
│  │  model.py
│  │  training.py
│  │              
│  └─ configs
│                  
└─ util
      arg_parser.py
      compute_cer.py
      saving.py
        
```


# 程式環境
* 程式語言：
python 3.10
* 安裝相關套件
``` 
$ pip install -r requirements.txt 
```


# 程式功能解釋與執行範例

* 資料前處理
```
# 請先將官方提供的 train_all.json 複製到 data_preprocess 目錄下

# 將路徑移動到前處理程式的目錄下
$ cd data_preprocess

# 將官方提供的 train_all.json 打亂並進行切分
# 程式預設是 training 85%, developing 10%, testing 5%
$ python split.py

# 將切分完的資料進行整理成特定的 json 格式
# 這邊會順便將「已被斷詞的 hypothesis 」，透過去除空格還原為原始句子
$ python data_preprocess.py
```


* 用 MLM 的方式微調 BERT
```
# 將路徑移動到訓練模型的目錄下
$ cd model

# 執行訓練程式
# python training.py --config [config file path] 
$ python training.py --config configs/mlm_bert/static/train.yaml
```


* inference MLM BERT
```
# 將路徑移動到 inference 的目錄下
$ cd inference

# 使用 MLM BERT 以 Pseudo Log Likelihood (PLL) scoring 的方式為 training, developing, testing sets 的 hypothesis 進行評分
# 評分後選出最高分者當作修正結果，並計算修正後錯誤率
# python inference.py --config [config file path]
$ python inference.py --config configs/mlm_bert/static/score.yaml
```


* 重新利用剛才的 MLM BERT 為 training set 打出來的分數訓練一個 distill BERT
```
# 將路徑移動到訓練模型的目錄下
$ cd model

# 執行訓練程式
# python training.py --config [config file path] 
$ python training.py --config configs/distill_bert/train.yaml
```


* inference distill BERT
```
# 將路徑移動到 inference 的目錄下
$ cd inference

# 使用 distill BERT 為 developing, testing sets 的 hypothesis 進行評分
# 評分後選出最高分者當作修正結果，並計算修正後錯誤率
# python inference.py --config [config file path]
$ python inference.py --config configs/distill_bert/score.yaml
```


* 執行 API 程式
```
# python api.py --checkpoint_path [distill BERT checkpoint file path]
$ python api.py --checkpoint_path model/result/distill_bert/checkpoint_1.pth
```

## 模型 training 與 inference 參數設定
* 請參考 config file 內的註解。

## 比賽期間使用的運算機器與平台
* 模型訓練
  * Colab GPU
  * NVIDIA GeForce GTX 1080 Ti

* API 運行 
Google Cloud Platform, 8 CPU, 32 GB Memory


## 參考論文
[Masked Language Model Scoring](https://aclanthology.org/2020.acl-main.240) (Salazar et al., ACL 2020)
