`This repository is for rebuttal purposes only, not the final version.`

## Dataset

#### MPID https://github.com/lsy17096535/Single-Image-Deraining

Note that MPID-drop dataset and DerainDropGAN dataset are same.

#### SPA-REAL https://stevewongv.github.io/derain-project.html

#### RainCityScapes https://www.cityscapes-dataset.com/downloads/
#### Rain100H & Rain100L http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html

## train 
python train.py --net='nas' --ssimloss --blocks=18 --ss=3

## test
python test.py --net='nas' --ssimloss --blocks=18 --ss=3

