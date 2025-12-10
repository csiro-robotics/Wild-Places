# Training on LoGG3D-Net
This folder contains the code needed to train LoGG3D-Net on the WildCross dataset

**Note: If you are reading this, you are on the WildCross branch and will be using the WildCross crossfold training splits.  Make sure that this is what you want!**

To set up training, the following steps need to be taken:

## 1. Clone and copy files
Firstly, clone the LoGG3D-Net repository onto your machine using the following command:
```
git clone https://github.com/csiro-robotics/LoGG3D-Net.git
```

Then, for all of the files in this folder and its subfolders copy that file into the corresponding subfolder in the LoGG3D-Net repository, overwriting the existing files when present.  We also recommend following the instructions provided by the LoGG3D-Net authors for setting up the environment to run the training script out of.

## 2. Generate pickle files
Secondly, we need to generate the pickle file containing the training splits.  To do this, cd into the root folder of the cloned LoGG3D-Net and run the generation script as follows:
```
 cd /path/to/LoGG3D-Net
 export PYTHONPATH=$PWD:$PYTHONPATH 
 python utils/data_loaders/general/make_wildcross_pickles.py \
    --dataset_root /path/to/wildcross/root \
    --save_folder pickles 
 ```

 ## 3. Train
 Now you should be ready to train the network by running `train.py` as follows:
 ```
 _NGPU=4
 _WILDCROSS_ROOT=/path/to/wildcross
 _SAVE_DIRECTORY=/path/to/savedir
 _SPLIT_IDX=0 # Change this depending on the train / testing split you're using 

cd /path/to/LoGG3D-Net

torchpack dist-run -np ${_NGPU} python training/train.py \
    --train_pipeline 'LOGG3D' \
    --dataset wildplaces \
    --train_pickles pickles/training_wildcross_split_idx${_SPLIT_IDX}.pickle \
    --wildplaces_dir $_WILDCROSS_ROOT \
    --gp_rem False \
    --train_num_workers 6 \
    --negatives_per_query 18 \
    --num_points 35000 \
    --voxel_size 0.5 \
    --max_epoch 18 \
    --collation_type 'reg_sparse_tuple' \
    --out_dir $_SAVE_DIRECTORY$ \
 ```
 **Note**: We train LoGG3D-Net using four GPUS and 18 negative examples per query.  Depending on your available resources you may want to change these values; however, expect performance to drop if the number of GPUS or negatives is reduced.
