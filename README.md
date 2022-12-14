# MLIFNet

Lightweight Multi-level Information Fusion Network for Facial Expression Recognition

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

- Step 2: put the teacher network [pre-trained](https://drive.google.com/file/d/1tkWxUj8yBV0KQCvuTOuOLhOHTF0AID4D/view?usp=share_link) model and MLIFNet [pre-trained](https://drive.google.com/file/d/1VigGoHijB-uOHQxn7r9mZEVUFZiyXpVx/view?usp=share_link) model to ***./checkpoint***. The fine-tuned model on RAF of the teacher network can be download [here](https://drive.google.com/file/d/1SRq35moA2x0xP82VVNxZawdpEMQds_fy/view?usp=share_link)
    
- Step 3: change ***data_path*** in *main_MLIFNet.py* to your path 

- Step 4: run ```python main_MLIFNet.py ```


## Note
When training from scratch or pre-training, use *main_MLIFNet.py* as well.
