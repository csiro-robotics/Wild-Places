# Training on LoGG3D-Net
This folder contains the code needed to train MinkLoc3Dv2 on the Wild-Places dataset

**Note: If you are reading this, you are on the WildCross branch and will be using the WildCross crossfold training splits.  Make sure that this is what you want!**

To set up training, the following steps need to be taken:

## 1. Clone and copy files
Firstly, clone the MinkLoc3Dv2 repository onto your machine using the following command:
```
git clone https://github.com/jac99/MinkLoc3Dv2.git
```

Then, for all of the files in this folder and its subfolders copy that file into the corresponding subfolder in the LoGG3D-Net repository, overwriting the existing files when present.  We also recommend following the instructions provided by the MinkLoc3Dv2 authors for setting up the environment to run the training script out of.

## 2. Generate pickle files and edit config
Secondly, we need to generate the pickle file containing the training splits.  To do this, cd into the root folder of the cloned MinkLoc3Dv2 and run the generation script as follows:
```
 cd /path/to/MinkLoc3Dv2
 export PYTHONPATH=$PWD:$PYTHONPATH 
 python datasets/wildplaces/make_train_tuples_wildcross.py \
    --dataset_root /path/to/wildcross/root \
    --save_folder pickles 
 ```

 In addition, edit the file `configs/config_wildplaces.txt` so that `dataset_folder` is set to the path to Wild-Places on your machine.

 ## 3. Train
 Now you should be ready to train the network by running `train.py` as follows:
 ```
cd /path/to/MinkLoc3Dv2
export PYTHONPATH=$PWD:$PYTHONPATH

python training/train.py \
    --config configs/config_wildplaces.txt \
    --model_config configs/model_wildplaces.txt \
    --save_folder /path/to/savedir
```