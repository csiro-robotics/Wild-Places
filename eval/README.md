# Evaluation
This subfolder contains info on how to use the provided evaluation scripts for the Wild-Places dataset.

## Testing Split Generation
First, you need to generate the testing splits using the provided script in `generate_splits/testing_sets.py`.  To do this, run the following terminal command out of the `generate_splits` folder:

```
python testing_sets.py --dataset_root /path/to/wildplaces/folder --save_folder .
```

This should generate a number of pickle files which contain the necessary information for running evaluation on the WildPlaces dataset

## Running evaluation
We provide scripts for running both inter and intra-sequence evaluation on Wild-Places, as well as an implementation of ScanContext.

* For example, to evaluate the released checkpoint of [LoGG3D-Net](https://github.com/csiro-robotics/LoGG3D-Net) ([LoGG3D-Net.pth](https://www.dropbox.com/s/h1ic00tvfnstvfm/LoGG3D-Net.pth?dl=0)), generate global descriptors for the query and database sets using the testing pickles as guidance for selecting which clouds to extract features before running the evaluation code as outlined below:

#### __Inter-run Evaluation__

To perform inter-run evaluation on the Wild-Places dataset, run the following command:
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

### ScanContext
To run evaluation with ScanContext, modify `scancontext/sc_eval_config.yaml` to point towards the Wild-Places dataset root and generated pickle files for testing and then run the following commands out of the `scancontext` folder for inter and intra-sequence evaluation respectively:
```
python sc_inter_sequence.py --config sc_eval_config.yaml

python sc_intra_sequence.py --config sc_eval_config.yaml
```

