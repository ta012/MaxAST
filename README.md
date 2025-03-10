
# MaxAST



## Introduction  

<p align="center"><img src="readme_max_ast_arch.png" alt="Illustration of MaxAST." width="1200"/></p>

Pytorch Implementation of **[Max-AST: Combining Convolution, Local and Global Self-Attentions for Audio Event Classification(ICASSP 2024)](https://ieeexplore.ieee.org/abstract/document/10447697)**

### Setting Up  
 Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd MaxAST/ 
conda env create -f ast.yml
conda activate ast
```

#### Data Preparation Audioset  
Since the AudioSet data is downloaded from YouTube directly, videos get deleted and the available dataset decreases in size over time. So you need to prepare the following files for the AudioSet copy available to you.

Prepare data files as mentioned in [AST](https://github.com/YuanGongND/ast.git)

#### Validation 
We have provided the best model. Please download the [model weight](https://drive.google.com/file/d/10qB6ByUooMLMGUv2B2nWEstpEs3fgTii/view?usp=sharing) and keep it in `pretrained_models/audioset_fullset/`. 

You can validate the model performance on your AudioSet evaluation data as follows,
```
cd MaxAST/egs/audioset
bash eval_run.sh
```
This script create log file with date time stamp in the same directory. You can find the mAP in the end of the log file.



## Acknowledgements
We are using the [AST](https://github.com/YuanGongND/ast) repo for model training and [timm](https://github.com/huggingface/pytorch-image-models/tree/main/timm)(do not install timm) for model implementation and ImageNet-1K pretrained weights.


## Citation

If you find our work useful, please cite it as:  

```bibtex
@inproceedings{alex2024max,
  title={Max-ast: Combining convolution, local and global self-attentions for audio event classification},
  author={Alex, Tony and Ahmed, Sara and Mustafa, Armin and Awais, Muhammad and Jackson, Philip JB},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1061--1065},
  year={2024},
  organization={IEEE}
}