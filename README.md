# Visible and Thermal Infrared Tracking via Multiple Adapters using Deep Feature Fusion and Enhancement

The official implementation for paper [**Visible and Thermal Infrared Tracking via Multiple Adapters using Deep Feature Fusion and Enhancement**]().

## Models

[Models &amp; Raw Results](https://drive.google.com/drive/folders/1l8j8Ns8dGyrKrFrmetHPdqKPO0wNrZ1n?usp=sharing) (Google Drive)

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

Dowmload the pretrained [foundation model](https://pan.baidu.com/s/1JX7xUlr-XutcsDsOeATU1A?pwd=4lvo) (OSTrack) (Baidu Driver: 4lvo) / [foundation model](https://drive.google.com/file/d/1WSkrdJu3OEBekoRz8qnDpnvEXhdr7Oec/view?usp=sharing) (Google Drive)
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

## Citation

Please cite our work if you think it is useful for your research.

```bibtex
@article{xue2025visible,
  title={Bi-directional Adapter for Multimodal Tracking},
  author={Bing Cao, Junliang Guo, Pengfei Zhu, Qinghua Hu},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## Acknowledgment

- This repo is based on [ViPT](https://github.com/jiawen-zhu/ViPT) which is an exellent work, helps us to quickly implement our ideas.
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library.
- Thanks for the [SDSTrack](https://github.com/hoqolo/SDSTrack) which is an excellent work.
