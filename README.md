# Real-Time Multi-Task Perception Model for Railway Obstacle Detection 
## The Overview of MTP-Rail

<img src="paper/Doc/1.png" width="800px">


## Requirement

## Install
```python
git clone https://github.com/ccl-1/obstacle_detection.git  # clone
cd obstacle_detection
```
## Dataset
- **ROD:**
Railway obstacle intrusion dataset. Dowload.
- **BDD100K:**
We uses the same  datasets as <a href="https://github.com/hustvl/YOLOP" title="YOLOP">YOLOP</a>.

We recommend the dataset directory structure to be the following:
```
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─labels
│ │ ├─train
│ │ ├─val
│ ├─semantic_labels
│ │ ├─train
│ │ ├─val
```

## Training
``` bash
# ROD
python semantic/train.py  \
    --cfg ./models/semantic/yolov5s-seg.yaml \
    --data ./data/obstacle.yaml \
    --batch-size 8 --epochs 150 \
    --label_map obstacle

# BDD100K
python semantic/train.py  \
    --label_map bdd100k \
    --cfg ./models/semantic/yolov5s-seg-bdd100k.yaml \
    --data ./data/bdd100k-seg.yaml \
    --batch-size 4 --epochs 150 \
    --use_bdd100k_5 True 
```

## Evaluation
``` bash
python semantic/val.py  \
    --weights rod.pt  \
    --data ./data/rs19.yaml \
    --label_map obstacle \

# BDD100K
python semantic/val.py  \
    --weights BDD100K.pt  \
    --data ./data/bdd100k-seg.yaml \
    --use_bdd100k_5 True \
    --label_map bdd100k 
```

## Results



## Visualization

![](paper/Doc/2.png)

## BibTeX

