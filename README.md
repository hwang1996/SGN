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

We filter out some food items without any images from the original dataset. You may download the processed dataset [train](https://drive.google.com/file/d/1-HxNAee0OEUdzs1MHXHHPE0muktRxZFL/view?usp=share_link) and [test](https://drive.google.com/file/d/17IUdRc9MsDjbSUFPixNLW9saHsMG4U_N/view?usp=share_link) to the ```data``` folder.

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

The pretrained Model is available at [here](https://drive.google.com/file/d/1oT1dKCTznZ2xW_ZgVbCAxd5EEwtacxuF/view?usp=share_link).

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
