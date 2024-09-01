# OMR

# [ECCV 2024] OMR: Occlusion-Aware Memory-Based Refinement for Video Lane Detection

### Dongkwon Jin and Chang-Su Kim


Official implementation for **"OMR: Occlusion-Aware Memory-Based Refinement for Video Lane Detection"** [[arxiv]](https://arxiv.org/abs/2408.07486) [[paper]](https://arxiv.org/abs/2408.07486) [[supp]](https://drive.google.com/file/d/1PrXyYWONMdnZW1eeBn4aeve10u3zy0Qh/view?usp=sharing)

**"OpenLane-V"** is available at [here](https://drive.google.com/file/d/1Jf7g1EG2oL9uVi9a1Fk80Iqtd1Bvb0V7/view?usp=sharing).
**"VIL-100"** is available at [here](https://github.com/yujun0-0/MMA-Net).
**"KINS"** is available at [here](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset?tab=readme-ov-file).

<img src="https://github.com/dongkwonjin/OMR/blob/main/Overview.png" alt="overview" width="80%" height="80%" border="0"/>


### Requirements
- PyTorch >= 1.10
- CUDA >= 10.0
- CuDNN >= 7.6.5
- python >= 3.6

### Installation
1. Download repository. We call this directory as `ROOT`:
```
$ git clone https://github.com/dongkwonjin/OMR.git
```

2. Download [pre-trained model](https://drive.google.com/file/d/19_3Tc3wXIbMxoiqcWbWRA6puAjEYZn4U/view?usp=sharing) parameters and [preprocessed data](https://drive.google.com/file/d/1S6_rQQ3P5B2EpzbqMzjOdNP-Ahu4McdK/view?usp=sharing) in `ROOT`:
```
$ cd ROOT
$ unzip pretrained.zip
$ unzip preprocessing.zip
```
4. Create conda environment:
```
$ conda create -n OMR python=3.8 anaconda
$ conda activate OMR
```
4. Install dependencies:
```
$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
$ pip install -r requirements.txt
```
Pytorch can be installed on [here](https://pytorch.org/get-started/previous-versions/). Other versions might be available as well.

### Dataset
Download [OpenLane-V](https://drive.google.com/file/d/1Jf7g1EG2oL9uVi9a1Fk80Iqtd1Bvb0V7/view?usp=sharing) into the original OpenLane dataset directory. VIL-100 can be downloaded in [here](https://github.com/yujun0-0/MMA-Net).
    
### Directory structure
    .                           # ROOT
    ├── Preprocessing           # directory for data preprocessing
    │   ├── VIL-100             # dataset name (VIL-100, OpenLane-V)
    |   |   ├── P00             # preprocessing step 1
    |   |   |   ├── code
    |   |   ├── P01             # preprocessing step 2
    |   |   |   ├── code
    |   │   └── ...
    │   └── ...                 # etc.
    ├── Modeling                # directory for modeling
    │   ├── VIL-100             # dataset name (VIL-100, OpenLane-V)
    |   |   ├── ILD_cls         # a part of ILD for predicting lane probability map and latent obstacle mask
    |   |   |   ├── code
    |   |   ├── ILD_reg         # a part of ILD for regressing lane coefficient maps
    |   |   |   ├── code
    |   |   ├── OMR             # occlusion-aware memory-based refinement module
    |   |   |   ├── code
    │   ├── OpenLane-V           
    |   |   ├── ...             # etc.
    ├── pretrained              # pretrained model parameters 
    │   ├── VIL-100              
    │   ├── OpenLane-V            
    │   └── ...                 # etc.
    ├── preprocessed            # preprocessed data
    │   ├── VIL-100             # dataset name (VIL-100, OpenLane-V)
    |   |   ├── P00             
    |   |   |   ├── output
    |   |   ├── P02             
    |   |   |   ├── output
    |   │   └── ...
    │   └── ...
    .
    .                           
    ├── OpenLane                # dataset directory
    │   ├── images              # Original images
    │   ├── lane3d_1000         # We do not use this directory
    │   ├── OpenLane-V
    |   |   ├── label           # lane labels formatted into pickle files
    |   |   ├── list            # training/test video datalists
    ├── VIL-100
    │   ├── JPEGImages          # Original images
    │   ├── Annotations         # We do not use this directory
    |   └── ...
    
### Evaluation (for VIL-100)
To test on VIL-100, you need to install official CULane evaluation tools. The official metric implementation is available [here](https://github.com/yujun0-0/MMA-Net/blob/main/INSTALL.md). Please downloads the tools into `ROOT/Modeling/VIL-100/MODEL_NAME/code/evaluation/culane/`. Then, you compile the evaluation tools. We recommend to see an [installation guideline](https://github.com/yujun0-0/MMA-Net/blob/main/INSTALL.md).
```
$ cd ROOT/Modeling/VIL-100/MODEL_NAME/code/evaluation/culane/
$ make
```

### Train
1. Set the dataset you want to train on (`DATASET_NAME`). Also, set the model (ILD or OMR) you want to train (`MODEL_NAME`).
2. Parse your dataset path into the `-dataset_dir` argument.
3. Edit `config.py` if you want to control the training process in detail
```
$ cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
$ python main.y --run_mode train --pre_dir ROOT/preprocessed/DATASET_NAME/ --dataset_dir /where/is/your/dataset/path 
```
 
### Test
1. Set the dataset you want to train on (`DATASET_NAME`). Also, set the model (ILD or OMR) you want to train (`MODEL_NAME`).
2. Parse your dataset path into the `-dataset_dir` argument.
3. If you want to get the performances of our work,
```
$ cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
$ python main.y --run_mode test_paper --pre_dir ROOT/preprocessed/DATASET_NAME/ --paper_weight_dir ROOT/pretrained/DATASET_NAME/ --dataset_dir /where/is/your/dataset/path
```
4. If you want to evaluate a model you trained,
```
$ cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
$ python main.y --run_mode test --pre_dir ROOT/preprocessed/DATASET_NAME/ --dataset_dir /where/is/your/dataset/path
```
5. (optional) If you set `disp_test_result=True` in code/options/config.py file, you can visualize the detection results.

### Preprocessing
You can obtain the preprocessed data, by running the codes in Preprocessing directories. Data preprocessing is divided into several steps. Below we describe each step in detail.
1. In P00, the type of ground-truth lanes in a dataset is converted to pickle format. (only for VIL-100)
2. In P01, each lane in a training set is represented by 2D points sampled uniformly in the vertical direction.
3. In P02, a lane matrix is constructed and SVD is performed. Then, each lane is transformed into its coefficient vector.
4. In P03, video-based datalists are generated for training and test sets.

```
$ cd ROOT/Preprocessing/DATASET_NAME/PXX_each_preprocessing_step/code/
$ python main.py --dataset_dir /where/is/your/dataset/path
```

### Reference
```
@Inproceedings{
    Jin2024omr,
    title={OMR: Occlusion-Aware Memory-Based Refinement for Video Lane Detection},
    author={Jin, Dongkwon and Kim, Chang-Su},
    booktitle={ECCV},
    year={2024}
}
```
