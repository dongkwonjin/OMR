# [ECCV 2024] OMR: Occlusion-Aware Memory-Based Refinement for Video Lane Detection

**Dongkwon Jin and Chang-Su Kim**

Official implementation of **"[OMR: Occlusion-Aware Memory-Based Refinement for Video Lane Detection](https://arxiv.org/abs/2408.07486)"**.

- **[Arxiv Paper](https://arxiv.org/abs/2408.07486)**
- **[Supplementary Material](https://drive.google.com/file/d/1PrXyYWONMdnZW1eeBn4aeve10u3zy0Qh/view?usp=sharing)**

## Datasets
- **[OpenLane-V](https://drive.google.com/file/d/1Jf7g1EG2oL9uVi9a1Fk80Iqtd1Bvb0V7/view?usp=sharing)**
- **[VIL-100](https://github.com/yujun0-0/MMA-Net)**
- **[KINS](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset?tab=readme-ov-file)**

<p align="center">
  <img src="https://github.com/dongkwonjin/OMR/blob/main/Overview.png" alt="Overview" width="80%" height="80%" />
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
│   │   ├── ILD_cls         # ILD module for lane probability map and obstacle mask
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
