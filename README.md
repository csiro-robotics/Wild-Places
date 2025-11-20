# Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments
<!-- ## [Website](https://csiro-robotics.github.io/Wild-Places/) | [Paper](https://arxiv.org/abs/2211.12732) | [Data Download Portal](https://data.csiro.au/collection/csiro:56372?q=wild-places&_st=keyword&_str=1&_si=1) -->
![](./utils/docs/teaser_image.png)
<div align="center">
<a href="https://arxiv.org/abs/2211.12732"><img src='https://img.shields.io/badge/arXiv-Wild Places-red' alt='Paper PDF'></a>
<a href='https://csiro-robotics.github.io/Wild-Places/'><img src='https://img.shields.io/badge/Project_Page-Wild Places-green' alt='Project Page'></a>
<a href='https://data.csiro.au/collection/csiro:56372?q=wild-places&_st=keyword&_str=1&_si=1'><img src='https://img.shields.io/badge/Download-Wild Places-blue' alt='Project Page'></a>
</div>


This repository contains the code implementation used in the paper *Wild-Places: A Large-Scale Dataset for Lidar Place Recognition in Unstructured Natural Environments*, which has been published at ICRA2023.  

If you find this dataset helpful for your research, please cite our paper using the following reference:
```
@inproceedings{2023wildplaces,
  title={Wild-places: A large-scale dataset for lidar place recognition in unstructured natural environments},
  author={Knights, Joshua and Vidanapathirana, Kavisha and Ramezani, Milad and Sridharan, Sridha and Fookes, Clinton and Moghadam, Peyman},
  booktitle={2023 IEEE international conference on robotics and automation (ICRA)},
  pages={11322--11328},
  year={2023},
  organization={IEEE}
}
```

## Contents
1. [Updates](#updates)
2. [Download Instructions](#download-instructions)
3. [Benchmarking](#benchmarking)
    * [Checkpoints](#checkpoints)
    * [Performance](#performance)
4. [Scripts](#scripts)
    * [Loading Point Clouds](#loading-point-clouds)
    * [Training](#training)
    * [Evaluation](#evaluation)
4. [Thanks](#thanks)

## Updates 
- **Oct 2022** Wild-Places v1.0 Uploaded
- **Jan 2023** Wild-Places is accepted to ICRA2023!
- **Jan 2023** Wild-Places v2.0 Uploaded.  This update to the dataset integrates GPS into the SLAM solution to alleviate vertical drift in the larger loops of the traversal in both environments. NOTE: Sequence K-04 is currently unavailable for v2.0 due to a failed loop closure in the ground truth.  We are currently working on remedying this, and will release the sequence as soon this issue is rectified.
- **Feb 2025** Fix the broken timestamps in the poses files. 
- **Nov 2025** Wild-Places v3.0 Uploaded.  This update to the dataset includes:
    - Updated point clouds / trajectories using the latest version of WildCat to bring the dataset in line with the pointclouds available in the WildScenes and WildCross datasets
    - Updated benchmarking results and instructions for training on LoGG3D-Net and MinkLoc3Dv2 
    - Updated dataset and repository file structure


## Download Instructions

Our dataset can be downloaded through [The CSIRO Data Access Portal](https://data.csiro.au/collection/csiro:56372?q=wild-places&_st=keyword&_str=1&_si=1). Detailed instructions for downloading the dataset can be found in the README file provided on the data access portal page. 


## Benchmarking
Here we provided pre-trained checkpoints and results for benchmarking several state-of-the-art LPR methods on the Wild-Places dataset.

**Update Nov. 2025**:  With the release of Wild-Places v3.0, we have also re-run training for two state-of-the-art methods (LoGG3D-Net, MinkLoc3Dv2) on the Wild-Places dataset using expanded batch sizes to provide new training checkpoints which better reflect the capabilities of recent state-of-the-art GPUs.  We provide checkpoints and benchmarked results for both the recently trained models and the checkpoints released with the ICRA2023 paper.

### Checkpoints
|Release| Model      | Checkpoint |
|------------|------------|------------|
|ICRA2023| TransLoc3D | [Link]()       |
|ICRA2023| MinkLoc3D  | [Link]()       |
|ICRA2023| LoGG3D-Net | [Link]()       |
|November 2025| MinkLoc3D  | [Link]()       |
|November 2025| LoGG3D-Net  | [Link]()       |


### Performance
![](./utils/docs/nov2025_wildplaces_benchmarking_results.png)

## Scripts
### Loading Point Clouds
A code snippet to load a pointcloud file from our dataset can be found in `eval/load_pointcloud.py`
### Training
We provide instructions for how to add Wild-Places as a training dataset for two state-of-the-art LPR methods: [LoGG3D-Net](https://github.com/csiro-robotics/LoGG3D-Net) and [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2).  For more detailed instructions, please consult the `README.md` files in `training/LoGG3D-Net` and `training/MinkLoc3Dv2`.
### Evaluation
We provide generic evaluation code for evaluating performance on the Wild-Places dataset for both the inter and intra-sequence testing scenarios, as well as an implementation of ScanContext.  For more details, please see the `README.md` file in the `eval` folder.
<!-- 


## 3. Scripts

### 3.1 Environment
To create a python environment to use the scripts in this repository run the following command:
```
conda env create -f scripts/Wild-Places.yaml -n Wild-Places
```

### 3.2 Loading Point Clouds


A code snippet to load a pointcloud file from our dataset can be found in `eval/load_pointcloud.py`

### 3.2 Generating Training & Testing Splits

In this repository we provide several scripts for partitioning our dataset into splits for training and evaluation.  
The output of these scripts are pickle files containing training and evaluation splits in a format compatible with existing repositories such as [PointNetVLAD](https://github.com/mikacuy/pointnetvlad), [MinkLoc3D](https://github.com/jac99/MinkLoc3D)([v2](https://github.com/jac99/MinkLoc3Dv2)), [TransLoc3D](https://github.com/slothfulxtx/TransLoc3D) and [PPT](https://github.com/fpthink/PPT-Net).

#### __Training__
To generate the training splits run the following command:
```
python scripts/generate_splits/training_sets.py --dataset_root $_PATH_TO_DATASET --save_folder --$_SAVE_FOLDER_PATH
```
Where `$_PATH_TO_DATASET` is the path to the downloaded dataset, and `$_SAVE_FOLDER_PATH` is the path to the directory where the generated files will be saved.

#### __Testing__

To generate the testing splits run the following command:
```
python scripts/generate_splits/testing_sets.py --dataset_root $_PATH_TO_DATASET --save_folder --$_SAVE_FOLDER_PATH
```
This script will generate seperate testing pickles for the inter-run and intra-run evaluation modes on each environment.  The inter-run pickles will produce query and database files for each testing environment, while the intra-run pickles will produce a seperate training pickle for each individual point cloud sequence.

### 3.3 Evaluation
We provide evaluation scripts for both inter and intra-run evaluation on our dataset.

* For example, to evaluate the released checkpoint of [LoGG3D-Net](https://github.com/csiro-robotics/LoGG3D-Net) ([LoGG3D-Net.pth](https://www.dropbox.com/s/h1ic00tvfnstvfm/LoGG3D-Net.pth?dl=0)), use the utility function provided in [scripts/eval/logg3d/logg3d_utils.py](https://github.com/csiro-robotics/Wild-Places/blob/eafde14a4a1aeb5e96a5d56c12ed046bcfdb02d9/scripts/eval/logg3d/logg3d_utils.py#L76) to generate global descriptors for `database_features` and `query_features`. Then, use the evaluation instructions provided below:

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
- `$_LOCATION_NAMES` is a string containing the name of the sequence being evaluated  -->

## Thanks
Special thanks to the authors of the [PointNetVLAD](https://github.com/mikacuy/pointnetvlad) and [MinkLoc3D](https://github.com/jac99/MinkLoc3D), whose excellent code was used as a basis for the generation and evaluation scripts used in this repository. 

