# I-MuPPET: Interactive Multi-Pigeon Pose Estimation and Tracking
This repository provides code for [I-MuPPET](https://urs-waldmann.github.io/i-muppet/) (GCPR 2022, oral).

**Abstract**

Most tracking data encompasses humans, the availability of annotated tracking data for animals is limited, especially for multiple objects. To overcome this obstacle, we present I-MuPPET, a system to estimate and track 2D keypoints of multiple pigeons at interactive speed. We train a Keypoint R-CNN on single pigeons in a fully supervised manner and infer keypoints and bounding boxes of multiple pigeons with that neural network. We use a state of the art tracker to track the individual pigeons in video sequences. I-MuPPET is tested quantitatively on single pigeon motion capture data, and we achieve comparable accuracy to state of the art 2D animal pose estimation methods in terms of Root Mean Square Error (RMSE). Additionally, we test I-MuPPET to estimate and track poses of multiple pigeons in video sequences with up to four pigeons and obtain stable and accurate results with up to 17 fps. To establish a baseline for future research, we perform a detailed quantitative tracking evaluation, which yields encouraging results. 

If you find a bug, have a question or know how to improve the code, please open an issue.

## Conda environment
Set up a conda environment with `conda env create -f environment.yml`.

## Data
**Multi-pigeon video sequences from the project page**

Our multi-pigeon video sequences from the [project page](https://urs-waldmann.github.io/i-muppet/) can be downloaded [here](https://zenodo.org/record/7037403). Unzip and copy the "videos" folder to `./data/`. You can use these video sequences to run I-MuPPET with pre-trained weights that we provide.

**Labeled single pigeon data**

If you are interested to train I-MuPPET on our labeled single pigeon data set, we kindly ask you to reach out to us. Unzip and copy the "pigeon_data" folder that we will provide to `./data/annotations/`.

**Multi-pigeon video sequences with ground truth for the quantitative tracking evaluation**

Our multi-pigeon video sequences with ground truth for the quantitative tracking evaluation can be downloaded [here](https://zenodo.org/record/7038391). Unzip and copy the "data" folder to `./`.

**Odor trail tracking video sequence (mouse) from DeepLabCut**

A video sequence of the [odor trail tracking data](https://zenodo.org/record/4008504#.YYK5IdbMIeZ) from [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y) can be found [here](https://github.com/DeepLabCut/DeepLabCut/tree/master/examples/openfield-Pranav-2018-10-30/videos). Download the video and copy it to `./data/videos/`.

**Odor trail tracking data (mice) from DeepLabCut preprocessed for I-MuPPET**

We also provide [odor trail tracking data](https://zenodo.org/record/4008504#.YYK5IdbMIeZ) from [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y) that we preprocessed. You can download this data [here](https://zenodo.org/record/7037327). Unzip and copy the "dlc_data" folder to `./data/annotations/`. Use this data to train I-MuPPET.

**Cowbird data from "3D Bird Reconstruction"**

["3D Bird Reconstruction"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630001.pdf) provides a cowbird data set. You can find it [here](https://drive.google.com/file/d/1vyXYIJIo9jneIqC7lowB4GVi17rjztjn/view). Download and copy to `./data/annotations/`. Use this data to train I-MuPPET for cowbirds.

## Pre-trained weights
Pre-trained weights for pigeons can be downloaded [here](https://zenodo.org/record/7037589), while for cowbirds and mice you find the pre-trained weights [here](https://zenodo.org/record/7037558). Unzip and copy the "weights" folder to `./data/`. You can use these pre-trained weights e.g. to run I-MuPPET on the multi-pigeon video sequences that we provide.

## I-MuPPET
**Preliminary task**

Clone the [SORT GitHub repository](https://github.com/abewley/sort) into `./`.

### Interactive Multi-Pigeon Pose Estimation and Tracking
To run I-MuPPET on a predefined video sequence, run:

    python muppet.py --plot_id --plot_pose --plot_tracker_bbox
    
The processed video will show the ID (`--plot_id`), the pose (`--plot_pose`) and the bounding box of the tracker (`--plot_tracker_bbox`).

To run I-MuPPET on another pigeon video sequence, specify the video with `--video`, e.g.:

    python muppet.py --plot_id --plot_pose --plot_tracker_bbox --video '3p_2118670.avi'
    
To run I-MuPPET on the [odor trail tracking data](https://zenodo.org/record/4008504#.YYK5IdbMIeZ) from [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y), use `--species` to specify the species and `--weights` to specify the pre-trained weights, e.g.:

    python muppet.py --plot_id --plot_pose --plot_tracker_bbox --species 'mouse' --weights 'dlc_comparison/mouse_split_1' --video 'm3v1mp4.mp4'
    
To run I-MuPPET in full screen, use `--full_screen`. To end video processing, press the "q" key on your keyboard.

## Train
**Preliminary task**

From this [PyTorch GitHub repository](https://github.com/pytorch/vision/tree/main/references/detection) download "coco_eval.py", "coco_utils.py", "engine.py" and "utils.py" and place them under `./utils/`.

### Training
Every experiment is defined by a configuration file. Configuration files with experiments from the paper can be found in `./experiments/`.

#### Pigeons
To train I-MuPPET on our labeled single pigeon data, run e.g.:

    python train.py --config './experiments/muppet_600.yaml'
    
The training will start with the configuration file specified by `--config`. The new weights will be stored in `./data/weights/my_weights/`.

#### Odor trail tracking data (mice) from DeepLabCut
To train I-MuPPET on the odor trail tracking data (mice) from [DeepLabCut](https://www.nature.com/articles/s41593-018-0209-y), run e.g.:

    python train.py --config './experiments/dlc_comparison/mouse_split_1.yaml'
    
#### Cowbirds from "3D Bird Reconstuction"
Before training, download "geometry.py", "img_utils.py" and "renderer.py" from the ["3D Bird Reconstruction" GitHub Repository](https://github.com/marcbadger/avian-mesh/tree/master/utils) and place them under `./data/utils/`.

To train I-MuPPET on the cowbird data from ["3D Bird Reconstruction"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630001.pdf), run e.g.:

    python train.py --config './experiments/3dbr_comparison/cowbird_45_epochs.yaml'

#### Visualization
To display some samples of the data sets with their annotations, use `--display_data`, e.g.:

    python train.py --config './experiments/muppet_600.yaml' --display_data
    
To quit, press the "q" key on your keyboard.

## Cite us

    @inproceedings{waldmann2022imuppet,
      title={I-MuPPET: Interactive Multi-Pigeon Pose Estimation and Tracking},
      author={Waldmann, Urs and Naik, Hemal and M\'{a}t\'{e}, Nagy and Kano, Fumihiro and Couzin, Iain D. and Deussen, Oliver and Goldl\"{u}cke, Bastian},
      booktitle={Pattern Recognition (Proc. of DAGM GCPR)},
      year={2022}
      }
      

