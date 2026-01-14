# Evaluation
This subfolder contains info on how to use the provided evaluation scripts for the WildCross dataset.

**Note: If you are reading this, you are on the WildCross branch and will be using the WildCross crossfold evaluation splits.  Make sure that this is what you want!**

## Testing Split Generation
First, you need to generate the testing splits using the provided script in `generate_splits/testing_sets-wildcross.py`.  To do this, run the following terminal command out of the `generate_splits` folder:

```
python testing_sets-wildcross.py --dataset_root /path/to/wildcross/folder --save_folder .
```

This should generate a number of pickle files which contain the necessary information for running evaluation on the WildCross dataset.

## Running evaluation
We provide scripts for running inter and intra-sequence evaluation on WildCross.
This comprises both generalised evaluation scripts (for custom methods) as well as scripts designed to replicate our evaluation process for for LoGG3D-Net and MinkLoc3Dv2.

### Generalised Evaluation Process
We provide the template for a generic evaluation script for inter and intra-sequence place recognition.  
To use this script, replace the `model_factory` and `get_latent_vector` placeholder functions with ones which load the model to be evaluated and extract place descriptors in order to get the LPR performance for a given pre-trained model and checkpoint.


#### __Inter-run Evaluation__

To perform inter-run evaluation on the WildCross dataset splits, run the following command:
```
python eval/inter-sequence-wildcross.py \
    --test_pickle_files $_PATH_TO_TEST_PICKLE_FILES \
    --location_names $_LOCATION_NAMES \
    --save_dir $_PATH_TO_SAVE_EVAL_RESULTS \ 
    --split_idx $_SPLIT_IDX \
    --ckpt $_PATH_TO_MODEL_CHECKPOINT \
    --dataset_root $_PATH_TO_WILDCROSS_DATABASE
```

parser.add_argument('--test_pickle_files', type=str, nargs='+', required=True)
    parser.add_argument('--location_names', type=str, nargs = '+', default=['Venman','Karawatha'])
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--split_idx', type=int, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--debug', action='store_true', default=False)

Where:
- `$_PATH_TO_TEST_PICKLE_FILES` is a string/set of strings pointing to the location/s of the generated testing_info.pickle file/s for venman and/or karawatha
- `$_LOCATION_NAMES` is a string/set of strings giving the name/s of the location/s being tested ("Venman" and/or "Karawatha")
- `$_PATH_TO_SAVE_EVAL_RESULTS` is a string pointing towards the folder where evaluation results are to be saved.
- `$_SPLIT_IDX` is an integer describing which split of the data is being tested. Note that this uses 0-indexing in comparison to the paper which uses 1-indexing. (e.g. Split 1 in the paper is `split_idx=0` here)
- `$_PATH_TO_MODEL_CHECKPOINT` is a string to the model weight checkpoint to be used within the evaluation
- `_PATH_TO_WILDCROSS_DATABASE` is a string pointing to the location of the WildCross dataset

#### __Intra-run Evaluation__
To perform intra-run evaluation on the Wild-Places dataset, run the following command:
```
python eval/intra-sequence-wildcross.py \
    --test_pickle_files $_PATH_TO_TEST_PICKLE_FILES \
    --location_names $_LOCATION_NAMES \
    --save_dir $_PATH_TO_SAVE_EVAL_RESULTS \ 
    --split_idx $_SPLIT_IDX \
    --ckpt $_PATH_TO_MODEL_CHECKPOINT \
    --dataset_root $_PATH_TO_WILDCROSS_DATABASE
```
Where:
- `$_PATH_TO_TEST_PICKLE_FILES` is a string/set of strings pointing to the location/s of the generated testing_info.pickle file/s for venman and/or karawatha
- `$_LOCATION_NAMES` is a string/set of strings giving the name/s of the location/s being tested ("Venman" and/or "Karawatha")
- `$_PATH_TO_SAVE_EVAL_RESULTS` is a string pointing towards the folder where evaluation results are to be saved.
- `$_SPLIT_IDX` is an integer describing which split of the data is being tested. Note that this uses 0-indexing in comparison to the paper which uses 1-indexing. (e.g. Split 1 in the paper is `split_idx=0` here)
- `$_PATH_TO_MODEL_CHECKPOINT` is a string to the model weight checkpoint to be used within the evaluation
- `$_PATH_TO_WILDCROSS_DATABASE` is a string pointing to the location of the WildCross dataset


### LoGG3D-Net
The following evaluation procedure assumes that you have set up LoGG3D-Net according to the instructions in the `training/LoGG3D-Net` folder of this repository and are operating in a python evnironment able to support LoGG3D-Net code.
To run evaluation with LoGG3D-Net and replicate the evaluation process used in the WildCross paper, first run the following commands.
```
LOGG3D_PATH=/path/to/LoGG3D-Net
export PYTHONPATH=$LOGG3D_PATH:$PYTHONPATH
```
Then, run the following commands out of the `eval/LoGG3D-Net` folder for inter and intra-sequence evaluation respectively.

**Note** this assumes you generated the splits as described at the beginning of the README.

#### __Inter-run Evaluation__
```
python logg3d_inter_sequence_wildcross.py \
    --test_pickle_files ../generate_splits/venman_testing_info.pickle ../generate_splits/karawatha_testing_info.pickle \
    --location_names Venman Karawatha \
    --save_dir $_PATH_TO_SAVE_EVAL_RESULTS \
    --split_idx $_SPLIT_IDX \
    --ckpt $_PATH_TO_MODEL_CHECKPOINT \
    --dataset_root $_PATH_TO_WILDCROSS_DATABASE
```

#### __Intra-run Evaluation__
```
python logg3d_intra_sequence_wildcross.py \
    --test_pickle_files ../generate_splits/venman_testing_info.pickle ../generate_splits/karawatha_testing_info.pickle \
    --location_names Venman Karawatha \
    --save_dir $_PATH_TO_SAVE_EVAL_RESULTS \
    --split_idx $_SPLIT_IDX \
    --ckpt $_PATH_TO_MODEL_CHECKPOINT \
    --dataset_root $_PATH_TO_WILDCROSS_DATABASE
```

### MinkLoc3Dv2
The following evaluation procedure assumes that you have set up MinkLoc3Dv2 according to the instructions in the `training/MinkLoc3Dv2` folder of this repository and are operating in a python evnironment able to support MinkLoc3Dv2 code.
To run evaluation with MinkLoc3Dv2, first run the following commands.
```
MINKLOC_PATH=/path/to/MinkLoc3Dv2
export PYTHONPATH=$MINKLOC_PATH:$PYTHONPATH
```
Then, run the following commands out of the `eval/MinkLoc3Dv2` folder for inter and intra-sequence evaluation respectively.

**Note** this assumes you generated the splits as described at the beginning of the README.

#### __Inter-run Evaluation__
```
python minkloc_inter_sequence_wildcross.py \
    --test_pickle_files ../generate_splits/venman_testing_info.pickle ../generate_splits/karawatha_testing_info.pickle \
    --location_names Venman Karawatha \
    --save_dir $_PATH_TO_SAVE_EVAL_RESULTS \
    --split_idx _SPLIT_IDX \
    --ckpt $_PATH_TO_MODEL_CHECKPOINT \
    --dataset_root $_PATH_TO_WILDCROSS_DATABASE \
    --config $MINKLOC_PATH/configs/config_wildplaces.txt \
    --model_config $MINKLOC_PATH/configs/model_wildplaces.txt
```

#### __Intra-run Evaluation__
```
python minkloc_intra_sequence_wildcross.py \
    --test_pickle_files /pickle/save/dir/venman_testing_info.pickle /pickle/save/dir/karawatha_testing_info.pickle \
    --location_names Venman Karawatha \
    --save_dir /path/to/results/save/dir \
    --split_idx CROSSFOLD_SPLIT_IDX \
    --ckpt /path/to/pretrained/ckpt.pth \
    --dataset_root /path/to/wildcross/root \
    --config /MinkLoc3Dv2/save/dir/configs/config_wildplaces.txt \
    --model_config /MinkLoc3Dv2/save/dir/configs/model_wildplaces.txt
```