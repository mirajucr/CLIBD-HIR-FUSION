# CLiBD-HiR Release

This repository is a paper-ready release for the work:

Hierarchy-Guided Multimodal Representation Learning for Taxonomic Inference  
Sk Miraj Ahmed, Xi Yu, Yunqi Li, Yuewei Lin, Wei Xu  
Accepted to the ICLR 2026 Workshop on Foundation Models for Science (FM4Science)

The repository is organized around the taxonomic inference setting studied in the paper: predicting taxonomic labels from specimen images, DNA barcodes, or both. The key idea is that taxonomic labels are hierarchical rather than flat, and that hierarchy can be used to learn more robust multimodal representations, especially when one modality is noisy or missing.

The original bioscan_clip directory inside BIO-Project was left unchanged. Everything in this release lives inside clibd_hir_release.

## What is in this release

Source code:

- train.py for training
- inference.py for evaluation and inference
- configs/train_bioclip_fusion.yaml as a runnable training config template
- model/ with the model and loss code needed by the release
- utils/ with dataset, sampler, and training helpers
- release_assets/ as a placeholder showing which large files should be downloaded separately

## Model variants in this release

The released checkpoints correspond to the main paper variants at a practical level.

- ours_no_fusion.pt: hierarchy-guided model without the fusion module
- ours_fusion.pt: hierarchy-guided model with the fusion module
- baseline.pt: baseline model used for comparison

The repository is structured so that readers can plug in released checkpoints and CSV files once those assets are hosted externally.

## Files you still need to download

This GitHub-ready repository does not include the large model checkpoints or released CSV files directly. Those should be hosted separately and linked here later.

### External model backbones

1. DNABERT-2 117M  
   https://huggingface.co/zhihan1996/DNABERT-2-117M

2. BioCLIP weights  
   https://huggingface.co/imageomics/bioclip

3. BIOSCAN-1M image HDF5 file  
   https://huggingface.co/datasets/bioscan-ml/BIOSCAN-1M/tree/main

The code expects the BIOSCAN image file in HDF5 format with a top-level group named bioscan_dataset. The official BIOSCAN-1M Hugging Face dataset page includes original_256.hdf5, which matches the format used by this code.

### Release assets to host separately

The following files should be uploaded to an external host such as Hugging Face, Zenodo, Google Drive, or institutional storage:

- baseline.pt
- ours_no_fusion.pt
- ours_fusion.pt
- train.csv
- test.csv
- train_split.csv
- val.csv

Once you host them, add the download links in this README.

## Suggested local layout after downloading everything

A simple local layout is:

    clibd_hir_release/
      external_weights/
        DNABERT-2-117M/
        bioclip/
          open_clip_pytorch_model.bin
      data/
        original_256.hdf5
      downloaded_release_assets/
        baseline.pt
        ours_no_fusion.pt
        ours_fusion.pt
        train.csv
        test.csv
        train_split.csv
        val.csv

You can place the downloaded assets somewhere else if you prefer. The command examples below just assume a simple layout.

## Quick start: run inference

### 1. Clone the repository

    git clone <your-repo-url>
    cd clibd_hir_release

### 2. Create the environment

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### 3. Download the external model backbones and BIOSCAN HDF5

Download:

- DNABERT-2 117M from https://huggingface.co/zhihan1996/DNABERT-2-117M
- BioCLIP from https://huggingface.co/imageomics/bioclip
- original_256.hdf5 from https://huggingface.co/datasets/bioscan-ml/BIOSCAN-1M/tree/main

### 4. Download the released checkpoints and test CSV

After you host them externally, download:

- baseline.pt
- ours_no_fusion.pt
- ours_fusion.pt
- test.csv

and place them under a local folder such as downloaded_release_assets.

### 5. Run inference

Run the hierarchy-guided model without fusion:

    python inference.py \
      --infer-mode fused \
      --backbone bioclip \
      --dnabert2-ckpt external_weights/DNABERT-2-117M \
      --openclip-ckpt external_weights/bioclip/open_clip_pytorch_model.bin \
      --full-model-ckpt downloaded_release_assets/ours_no_fusion.pt \
      --test-csv downloaded_release_assets/test.csv \
      --test-hdf5 data/original_256.hdf5

Run the hierarchy-guided model with fusion:

    python inference.py \
      --infer-mode fused \
      --backbone bioclip \
      --dnabert2-ckpt external_weights/DNABERT-2-117M \
      --openclip-ckpt external_weights/bioclip/open_clip_pytorch_model.bin \
      --full-model-ckpt downloaded_release_assets/ours_fusion.pt \
      --test-csv downloaded_release_assets/test.csv \
      --test-hdf5 data/original_256.hdf5

Run the baseline model:

    python inference.py \
      --infer-mode fused \
      --backbone bioclip \
      --dnabert2-ckpt external_weights/DNABERT-2-117M \
      --openclip-ckpt external_weights/bioclip/open_clip_pytorch_model.bin \
      --full-model-ckpt downloaded_release_assets/baseline.pt \
      --test-csv downloaded_release_assets/test.csv \
      --test-hdf5 data/original_256.hdf5

To test robustness under DNA corruption, add --noise to the same command.

## Quick start: run training

### 1. Make sure the environment and external downloads are in place

Use the same environment setup and external downloads described above.

You need:

- DNABERT-2 117M
- BioCLIP weights
- original_256.hdf5
- train_split.csv
- val.csv

### 2. Edit the training config

Open:

- configs/train_bioclip_fusion.yaml

Update the paths so they match where you downloaded the external files. In particular, set:

- dataset_config.csv_train_path
- dataset_config.csv_val_path
- dataset_config.image_hdf5_path
- pre_trained_model_config.dnabert2
- pre_trained_model_config.open_clip_bioclip

### 3. Run training

Single GPU:

    python train.py --config configs/train_bioclip_fusion.yaml

Distributed training:

    torchrun --nproc_per_node=4 train.py --config configs/train_bioclip_fusion.yaml

Checkpoints are written to the directory specified by model_config.save_path.

## Data format expected by the code

### Inference CSV

The inference CSV should include at least:

- image_file
- taxonomy
- nucraw

### Training CSV

Training CSV files should include:

- image_file
- taxonomy
- nucraw
- unique_label
- order
- family
- genus
- species
- All_level_label

### HDF5 image file

The HDF5 file should contain a bioscan_dataset group keyed by image_file.

## Citation

If you use this release, please cite the paper:

    @inproceedings{ahmed2026hierarchy,
      title={Hierarchy-Guided Multimodal Representation Learning for Taxonomic Inference},
      author={Ahmed, Sk Miraj and Yu, Xi and Li, Yunqi and Lin, Yuewei and Xu, Wei},
      booktitle={ICLR 2026 Workshop on Foundation Models for Science (FM4Science)},
      year={2026}
    }

## External resources referenced here

- Paper: https://arxiv.org/abs/2603.25573
- DNABERT-2 117M: https://huggingface.co/zhihan1996/DNABERT-2-117M
- BioCLIP: https://huggingface.co/imageomics/bioclip
- BIOSCAN-1M dataset: https://huggingface.co/datasets/bioscan-ml/BIOSCAN-1M

## Final note

This repository is now set up as a GitHub-friendly code release. Large checkpoints and CSV files should be hosted separately and linked from this README when you are ready to publish them.
