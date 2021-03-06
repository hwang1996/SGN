# [Structure-Aware Network for Recipe Generation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720358.pdf)
## Codes of SGN (PyTorch)
*Structure-Aware Generation Network for Recipe Generation from Images*  
Hao Wang, Guosheng Lin, Steven C. H. Hoi, and Chunyan Miao  
ECCV 2020  

## Requirements
* pytorch 1.2 or higher
* python 3.6 or higher

## Dataset

The dataset is from [Recipe1M](http://pic2recipe.csail.mit.edu/). 

We filter out some food items without any images from the original dataset. You may download the processed dataset [train](https://entuedu-my.sharepoint.com/:u:/g/personal/hao005_e_ntu_edu_sg/EQoTVWLgNsRDlY-rqkTQgOgBa1uVt41uvxBH0IliNpZUQg?e=fcy18t) and [test](https://entuedu-my.sharepoint.com/:u:/g/personal/hao005_e_ntu_edu_sg/EeS_dCAwkvVDrK2RISeTPKMB2V7JpD4OBzoIkiiIBz71hQ?e=U4EykI) to the ```data``` folder.

## Produce the Sentence-Level Tree Structures

### Tokenize the recipes
```
cd hierarchical_on_lstm
python tokenize_sentence.py
```

### Unsupervisedly train the model
```
python train.py
```

### Output the parsing trees
```
python test_phrase_grammar.py
```


## Pretrained Model

The pretrained Model is available at [here](https://entuedu-my.sharepoint.com/:u:/g/personal/hao005_e_ntu_edu_sg/ETr0ecOhVSBLik452hW2NmsBkZw74WwlQxb_jEQ8SFC5Xw?e=6nxhFk).

## Reference
This code is modified based on [ON-LSTM](https://github.com/yikangshen/Ordered-Neurons). If you find this repo useful, please cite:
```
@inproceedings{wang2020structure,
  title={Structure-Aware Generation Network for Recipe Generation from Images},
  author={Wang, Hao and Lin, Guosheng and Hoi, Steven CH and Miao, Chunyan},
  booktitle={European Conference on Computer Vision},
  pages={359--374},
  year={2020},
  organization={Springer}
}
```
