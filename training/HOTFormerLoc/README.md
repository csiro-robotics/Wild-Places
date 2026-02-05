# Training on HOTFormerLoc
This folder contains the code needed to train HOTFormerLoc on the Wild-Places dataset

**Note: If you are reading this, you are on the WildCross branch and will be using the WildCross crossfold training splits.  Make sure that this is what you want!**

To set up training, the following steps need to be taken:

## 1. Clone and copy files
Firstly, clone the HOTFormerLoc repository onto your machine using the following command:
```
git clone https://github.com/csiro-robotics/HOTFormerLoc.git
```

Then, for all of the files in this folder and its subfolders copy that file into the corresponding subfolder in the HOTFormerLoc repository, overwriting the existing files when present.  We also recommend following the instructions provided by the HOTFormerLoc authors for setting up the environment to run the training script out of.

## 2. Generate pickle files and edit config
Secondly, we need to generate the pickle file containing the training splits.  To do this, cd into the root folder of the cloned HOTFormerLoc and run the generation script as follows:
```
 cd /path/to/HOTFormerLoc
 export PYTHONPATH=$PWD:$PYTHONPATH 
 python datasets/WildCross/generate_training_tuples.py \
    --dataset_root /path/to/wildcross/root \
    --save_folder pickles 
 ```

 In addition, edit the file `config/config_wildcross.txt` so that `dataset_folder` is set to the path to Wild-Cross on your machine and `train_file` is set to the path of a pickle file generated above.

 ## 3. Train
 Now you should be ready to train the network by running `train.py` as follows:
 ```
cd /path/to/HOTFormerLoc
export PYTHONPATH=$PWD:$PYTHONPATH

python training/train.py \
    --config config/config_wildcross.txt \
    --model_config config/hotformerloc_wildcross_cfg.txt \
    --save_dir /path/to/savedir
```

If you run into CUDA out-of-memory errors, you may need to decrease `batch_split_size` in the config file. This should not impact model performance. If running out of RAM or SHM, you may need to decrease num_workers.

In our experiments, we use the checkpoint from 40 training epochs (will be saved as `model_e40.ckpt`).

## 4. Evaluation

See `Wild-Places/eval/` for evaluation instructions.