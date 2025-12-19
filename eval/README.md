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
We provide scripts for running inter and intra-sequence evaluation on WildCross for LoGG3D-Net and MinkLoc3Dv2. For these two models, please refer to the next sections.

We also provide the template for a generic evaluation script for inter and intra-sequence place recognition.  To use this script, replace the `model_factory` and `get_latent_vector` placeholder functions with ones which load the model to be evaluated and extract place descriptors in order to get the LPR performance for a given pre-trained model and checkpoint.


#### __Inter-run Evaluation__

To perform inter-run evaluation on the WildCross dataset, run the following command:
```
python eval/inter-sequence.py \
    --queries $_PATH_TO_QUERIES_PICKLES \
    --databases $_PATH_TO_DATABASES_PICKLES \
    --query_features $_PATH_TO_QUERY_FEATURES \ 
    --database_features $_PATH_TO_DATABASE_FEATURES \
    --location_names $_LOCATION_NAMES \
```

Where:
- `$_PATH_TO_QUERIES_PICKLES` is a string pointing to the location of the generated query set pickle for an environment
- `$_PATH_TO_DATABASES_PICKLES` is a string pointing to the location of the generated database set pickle for an environment
- `$_PATH_TO_QUERY_FEATURES` is a string pointing towards a pickle file containing the query set features to be used in evaluation.  These features should be a list of Nxd numpy arrays or tensors, where N is the number of point cloud frames in the query set of each sequence in the environment.
- `$_PATH_TO_DATABASE_FEATURES` is a string pointing towards a pickle file containing the database set features to be used in evaluation.  These features should be a list of Nxd numpy arrays or tensors, where N is the number of point cloud frames in the database set of each sequence in the environment.
- `$_LOCATION_NAMES` is a string containing the name of the environment being evaluated

#### __Intra-run Evaluation__
To perform intra-run evaluation on the Wild-Places dataset, run the following command:
```
python eval/intra-sequence.py \
    --databases $_PATH_TO_DATABASES_PICKLES \
    --database_features $_PATH_TO_DATABASE_FEATURES \
    --run_names $_LOCATION_NAMES \
```
Where:
- `$_PATH_TO_DATABASES_PICKLES` is a string pointing to the location of the generated database set pickle for a single sequence
- `$_PATH_TO_DATABASE_FEATURES` is a string pointing towards a pickle file containing the run features to be used in evaluation.  These features should be a single Nxd numpy array or tensor, where N is the number of point cloud frames in that sequence
- `$_LOCATION_NAMES` is a string containing the name of the sequence being evaluated 


### LoGG3D-Net
To run evaluation with LoGG3D-Net, first run the following commands.
```
LOGG3D_PATH=/path/to/LoGG3D-Net
export PYTHONPATH=$LOGG3D_PATH:$PYTHONPATH
```
Then, run the following commands out of the `eval/LoGG3D-Net` folder for inter and intra-sequence evaluation respectively.

#### __Inter-run Evaluation__
```
python logg3d_inter_sequence_wildcross.py \
    --test_pickle_files /pickle/save/dir/venman_testing_info.pickle /pickle/save/dir/karawatha_testing_info.pickle \
    --location_names Venman Karawatha \
    --save_dir /path/to/results/save/dir \
    --split_idx CROSSFOLD_SPLIT_IDX \
    --ckpt /path/to/pretrained/ckpt.pth \
    --dataset_root /path/to/wildcross/root
```

#### __Intra-run Evaluation__
```
python logg3d_intra_sequence_wildcross.py \
    --test_pickle_files /pickle/save/dir/venman_testing_info.pickle /pickle/save/dir/karawatha_testing_info.pickle \
    --location_names Venman Karawatha \
    --save_dir /path/to/results/save/dir \
    --split_idx CROSSFOLD_SPLIT_IDX \
    --ckpt /path/to/pretrained/ckpt.pth \
    --dataset_root /path/to/wildcross/root
```

### MinkLoc3Dv2
To run evaluation with MinkLoc3Dv2, first run the following commands.
```
MINKLOC_PATH=/path/to/MinkLoc3Dv2
export PYTHONPATH=$MINKLOC_PATH:$PYTHONPATH
```
Then, run the following commands out of the `eval/MinkLoc3Dv2` folder for inter and intra-sequence evaluation respectively.

#### __Inter-run Evaluation__
```
python minkloc_inter_sequence_wildcross.py \
    --test_pickle_files /pickle/save/dir/venman_testing_info.pickle /pickle/save/dir/karawatha_testing_info.pickle \
    --location_names Venman Karawatha \
    --save_dir /path/to/results/save/dir \
    --split_idx CROSSFOLD_SPLIT_IDX \
    --ckpt /path/to/pretrained/ckpt.pth \
    --dataset_root /path/to/wildcross/root \
    --config /MinkLoc3Dv2/save/dir/configs/config_wildplaces.txt \
    --model_config /MinkLoc3Dv2/save/dir/configs/model_wildplaces.txt
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