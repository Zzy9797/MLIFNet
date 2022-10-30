# MLIFNet

"[Lightweight Multi-level Information Fusion Network for Facial Expression Recognition]"

## Requirements

- Python >= 3.6
- PyTorch >= 1.2
- torchvision >= 0.4.0

## Training

- Step 1: prepare the datasets, and make sure they have the structures like following (take RAF-DB as an example):
 
```
./RAF-DB/
         train/
               0/
                 train_09748.jpg
                 ...
                 train_12271.jpg
               1/
               ...
               6/
         test/
              0/
              ...
              6/

[Note] 0: Neutral; 1: Happiness; 2: Sadness; 3: Surprise; 4: Fear; 5: Disgust; 6: Anger
```

- Step 2: put the pretrained model to***./checkpoint***(Limited by upload size, we will provide it later).
    
- Step 3: change ***data_path*** in *main_MLIFNet.py* to your path 

- Step 4: run ```python main_MLIFNet.py ```


## Note
When training from scratch or pre-training, use *main_MLIFNet.py* as well.