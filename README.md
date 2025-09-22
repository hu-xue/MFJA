# Visible and Thermal Infrared Tracking via Multiple Adapters using Deep Feature Fusion and Enhancement
The official implementation for paper "**[Visible and Thermal Infrared Tracking via Multiple Adapters using Deep Feature Fusion and Enhancement](https://ieeexplore.ieee.org/document/11112670)**".

## News

- ⭐ Our paper has been accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) 2025 for early access. Please cite our paper if you find our work useful in your research.
```
@article{xue2025feature,
  author  = {Xue, Hu and Zhu, Hao and Ran, Zhidan and Tang, Xianlun and Qi, Guanqiu and Zhu, Zhiqin and Kuok, Sin-Chi and Leung, Henry},
  journal = {IEEE Trans. Circuits Syst. Video Technol.},
  title   = {Feature Fusion and Enhancement for Lightweight {Visible-Thermal} Infrared Tracking via Multiple Adapters},
  year    = {2025},
  volume  = {},
  number  = {},
  pages   = {1-1},
  doi     = {10.1109/TCSVT.2025.3595632}
}
```
## Models and Raw Results

[Models and Raw Results](https://pan.baidu.com/s/1G1nMA1Xlxqz8b-DB_XlwgA?pwd=xg6h) (Baidu Driver: xg6h)

## Usage

### Installation

Create and activate a conda environment:

```bash
conda create -n mfja python=3.9
conda activate mfja
```

Install the required packages:

```bash
bash install_mfja.sh
```

### Data Preparation

Download the training datasets, It should look like:

```bash
$<PATH_of_Datasets>
    -- LasHeR/TrainingSet
        |-- 1boygo
        |-- 1handsth
        ...
```

### Path Setting

Run the following command to set paths:

```bash
cd <PATH_of_MFJA>
python tracking/create_default_local_file.py --workspace_dir . --data_dir <PATH_of_Datasets> --save_dir ./output
```

You can also modify paths by these two files:

```bash
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training

Dowmload the pretrained [foundation model](https://pan.baidu.com/s/1G1nMA1Xlxqz8b-DB_XlwgA?pwd=xg6h) (filename: OSTrack_ep0300.pth.tar) (Baidu Driver: xg6h)
and put it under ./pretrained/.

```bash
bash train.sh
```

You can train models with various modalities and variants by modifying ``train.sh``.

### Testing For RGB-T benchmarks

[LasHeR & RGBT234]
Modify the <DATASET_PATH> and <SAVE_PATH> in ``./RGBT_workspace/test_rgbt_mgpus.py``, then run:

```bash
bash eval_rgbt.sh
```

We refer you to use [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation,
and refer you to use [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.

## Acknowledgment

- This repo is based on [ViPT](https://github.com/jiawen-zhu/ViPT) which is an exellent work, helps us to quickly implement our ideas.
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library.
- Thanks for the [SDSTrack](https://github.com/hoqolo/SDSTrack) which is an excellent work.
- Thanks for the [BAT](https://github.com/SparkTempest/BAT) which is an excellent work.
