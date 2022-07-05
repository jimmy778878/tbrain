# tbrain

## 程式架構
```
.
│  .gitignore
│  api.py
│  README.md
│  requirements.txt
│  temp.txt
│  
├─data_preprocess
│  │  data_preprocess.py
│  │  dev.json
│  │  split.py
│  │  test.json
│  │  train.json
│  │  train_all.json
│  │  
│  ├─dev
│  │      
│  ├─test
│  │      
│  └─train
│          
├─inference
│  │  inference.py
│  │  
│  └─configs
│                  
├─model
│  │  data.py
│  │  model.py
│  │  training.py
│  │              
│  └─configs
│                  
└─util
        arg_parser.py
        compute_cer.py
        saving.py
        
```