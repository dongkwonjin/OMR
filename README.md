# [ECCV 2024] OMR: Occlusion-Aware Memory-Based Refinement for Video Lane Detection

**Dongkwon Jin and Chang-Su Kim**

Official implementation of **"[OMR: Occlusion-Aware Memory-Based Refinement for Video Lane Detection](https://arxiv.org/abs/2408.07486)"**.

- **[Arxiv Paper](https://arxiv.org/abs/2408.07486)**
- **[Supplementary Material](https://drive.google.com/file/d/1PrXyYWONMdnZW1eeBn4aeve10u3zy0Qh/view?usp=sharing)**

<p align="center">
  <img src="https://github.com/dongkwonjin/OMR/blob/main/Overview.png" alt="Overview" width="100%" height="100%" />
</p>

## Requirements
- Python >= 3.6
- PyTorch >= 1.10
- CUDA >= 10.0
- CuDNN >= 7.6.5

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/dongkwonjin/OMR.git
    cd OMR
    ```

2. **Download pre-trained models and preprocessed data**:
    - [Pre-trained model parameters](https://drive.google.com/file/d/19_3Tc3wXIbMxoiqcWbWRA6puAjEYZn4U/view?usp=sharing)
    - [Preprocessed data](https://drive.google.com/file/d/1S6_rQQ3P5B2EpzbqMzjOdNP-Ahu4McdK/view?usp=sharing)

    ```bash
    unzip pretrained.zip
    unzip preprocessing.zip
    ```

3. **Create and activate a conda environment**:
    ```bash
    conda create -n OMR python=3.8 anaconda
    conda activate OMR
    ```

4. **Install dependencies**:
    ```bash
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```

    You can also find more options for installing PyTorch [here](https://pytorch.org/get-started/previous-versions/).

## Datasets
1. **OpenLane-V**: Download [OpenLane-V](https://drive.google.com/file/d/1Jf7g1EG2oL9uVi9a1Fk80Iqtd1Bvb0V7/view?usp=sharing) and place it in the original OpenLane dataset directory.

2. **VIL-100**: Download from [here](https://github.com/yujun0-0/MMA-Net).

3. **KINS**: Download from [here](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset?tab=readme-ov-file).

## Directory Structure
```bash
ROOT
├── Preprocessing           # Data preprocessing code
│   ├── VIL-100             # Dataset: VIL-100, OpenLane-V
│   │   ├── P00             # Preprocessing step 1
│   │   │   ├── code
│   │   ├── P01             # Preprocessing step 2
│   │   │   ├── code
│   │   └── ...
│   └── ...
├── Modeling                # Model code
│   ├── VIL-100             # Dataset: VIL-100, OpenLane-V
│   │   ├── ILD_cls         # ILD module for predicting lane probability map and latent obstacle mask
│   │   │   ├── code
│   │   ├── ILD_reg         # ILD module for regressing lane coefficient maps
│   │   │   ├── code
│   │   ├── OMR             # OMR module
│   │   │   ├── code
│   ├── OpenLane-V
│   │   ├── ...
├── pretrained              # Pretrained model parameters
│   ├── VIL-100
│   ├── OpenLane-V
│   └── ...
├── preprocessed            # Preprocessed data
│   ├── VIL-100
│   │   ├── P00             
│   │   │   ├── output
│   │   ├── P02             
│   │   │   ├── output
│   └── ...
├── OpenLane                # Dataset directory
│   ├── images
│   ├── lane3d_1000         # Not used
│   ├── OpenLane-V
│   │   ├── label
│   │   ├── list
├── VIL-100
│   ├── JPEGImages
│   ├── Annotations         # Not used
└── ...
```

## Evaluation (for VIL-100)

To evaluate using VIL-100, follow these steps:

1. **Install Evaluation Tools**
   - Download the official CULane evaluation tools from [here](https://github.com/yujun0-0/MMA-Net/blob/main/INSTALL.md).
   - Save the tools in the `ROOT/Modeling/VIL-100/MODEL_NAME/code/evaluation/culane/` directory.

   ```bash
   cd ROOT/Modeling/VIL-100/MODEL_NAME/code/evaluation/culane/
   make
   ```
2. **Refer to the Installation Guidelines**
   - For detailed installation instructions, refer to the installation guideline.
  
## Training

1. **Configure Training**
   - Set the dataset (`DATASET_NAME`) and model (`MODEL_NAME`) you want to train.
   - Specify your dataset path using the `-dataset_dir` argument.

   ```bash
   cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
   python main.y --run_mode train --pre_dir ROOT/preprocessed/DATASET_NAME/ --dataset_dir /path/to/your/dataset
   ```
2. **Optional: Edit Configuration**
  - Modify `config.py` to adjust the training parameters as needed.


## Testing

1. **Evaluate Pre-Trained Models**
   - To get performances of pre-trained models:
   
   ```bash
   cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
   python main.y --run_mode test_paper --pre_dir ROOT/preprocessed/DATASET_NAME/ --paper_weight_dir ROOT/pretrained/DATASET_NAME/ --dataset_dir /path/to/your/dataset

   ```
2. **Evaluate Your Trained Model**
   - To evaluate a model you have trained:
  ```bash
   cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
python main.y --run_mode test --pre_dir ROOT/preprocessed/DATASET_NAME/ --dataset_dir /path/to/your/dataset
   ```
3. **(Optional) Visualize Results**
   - To visualize detection results, set `disp_test_result=True` in `code/options/config.py`.


## Preprocessing

Preprocessing data involves several steps:
1. **Convert Ground-Truth Lanes**
   - Convert ground-truth lanes to pickle format (VIL-100 specific).
2. **2D Point Representation**
   - Represent each lane in the training set as 2D points sampled uniformly in the vertical direction.
3. **Lane Matrix Construction**
   - Construct a lane matrix, perform SVD, and transform each lane into its coefficient vector.
4. **Generate Video-Based Datalists**
   - Create datalists for training and test sets.
  ```bash
   cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
python main.y --run_mode test --pre_dir ROOT/preprocessed/DATASET_NAME/ --dataset_dir /path/to/your/dataset
   ```

## Reference
```
@Inproceedings{
    Jin2024omr,
    title={OMR: Occlusion-Aware Memory-Based Refinement for Video Lane Detection},
    author={Jin, Dongkwon and Kim, Chang-Su},
    booktitle={ECCV},
    year={2024}
}
```
