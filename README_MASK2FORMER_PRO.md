# Mask2FormerPro: Depth-Guided Mask2Former for Image Segmentation

This repository extends [Mask2Former](https://github.com/facebookresearch/Mask2Former),
the masked-attention mask transformer for universal image segmentation, with a
depth/disparity clustering pipeline for Cityscapes instance segmentation.

The project is intended for research on using geometric proposals from stereo
disparity to improve Mask2Former. It includes GPU-accelerated depth clustering,
parameter-search utilities, visualization tools, and Cityscapes training
configurations.

> **Research status:** This is experimental research code. Configuration names,
> paths, and pretrained checkpoints may need to be adapted to your system.

## Main additions

- PyTorch/CUDA depth clustering for Cityscapes disparity maps
- Batched processing with per-image camera calibration
- Optional ground removal and adaptive minimum cluster sizes
- Parameter search against Cityscapes instance ground truth
- Evaluation of proposal recall, precision, false positives, fragmentation,
  F1, and proposal PQ
- Support for parameter search with the same resize and crop augmentations used
  during training
- Visualization tools for inspecting the depth-processing pipeline
- Modified Mask2Former training configurations for depth-guided experiments

## Repository layout

The relevant files are expected to follow this structure:

```text
Mask2FormerPro/
├── configs/
│   └── cityscapes/
│       └── instance-segmentation/
├── datasets/
│   └── cityscapes/                 # optional; may be stored elsewhere
├── demo/
├── mask2former/
│   ├── modeling/
│   │   └── pixel_decoder/
│   │       └── ops/
│   └── preprocessing/
├── models/
│   └── instance/
├── output/
├── depth_clustring_torch/
├── demo.py
├── train_net.py
└── requirements.txt
```

## Requirements

- Linux
- Python 3.10
- Conda or Miniconda
- NVIDIA GPU
- CUDA Toolkit compatible with the installed PyTorch build
- PyTorch 2.4.0 with CUDA 12.1 (the tested environment)

Check that the CUDA compiler is available:

```bash
which nvcc
nvcc --version
nvidia-smi
```

The commands below assume that the repository root is the current directory.

## Installation

### 1. Clone the project

```bash
git clone <MASK2FORMER_PRO_REPOSITORY_URL>
cd Mask2FormerPro
```

### 2. Create the environment

Create the tested Conda environment from the included `maskfo.yml` file:

```bash
conda env create -f maskfo.yml
conda activate maskfo
```

The environment name is defined in `maskfo.yml`. If it differs from `maskfo`,
activate the name specified by the file.

Set the CUDA paths if they are not already configured:

```bash
export CUDA_HOME=/usr
export PATH=/usr/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"
```

Change `TORCH_CUDA_ARCH_LIST` to the compute capability of the GPU on the
training machine.

### 3. Install Detectron2

Detectron2 should be built against the same PyTorch and CUDA versions:

```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2 --no-build-isolation
```

### 4. Install the project requirements

```bash
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/mcordts/cityscapesScripts.git
```

### 5. Build Multi-Scale Deformable Attention

```bash
cd mask2former/modeling/pixel_decoder/ops
python -m pip install . --no-build-isolation
cd ../../../..

python -c "import MultiScaleDeformableAttention; print('MultiScaleDeformableAttention: OK')"
```

If the dynamic linker cannot find the PyTorch libraries, run:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}"
```

### 6. Install the depth-clustering extension

If `depth_clustring_torch` is included as a subdirectory:

```bash
python -m pip install -e depth_clustring_torch --no-build-isolation
```

Otherwise, clone and install it:

```bash
git clone https://github.com/fardinayar/depth_clustring_torch.git
python -m pip install -e depth_clustring_torch --no-build-isolation
```

Verify the installation:

```bash
python -c "import depth_clustering_torch; print('depth_clustering_torch: OK')"
```

## Cityscapes dataset

Register at the [Cityscapes website](https://www.cityscapes-dataset.com/) and
download the required packages. The depth-clustering parameter search requires:

- left images
- stereo disparity maps
- camera calibration files
- fine instance annotations

The extracted dataset should look like:

```text
cityscapes/
├── camera/
│   ├── train/
│   └── val/
├── disparity/
│   ├── train/
│   └── val/
├── gtFine/
│   ├── train/
│   └── val/
└── leftImg8bit/
    ├── train/
    └── val/
```

Configure the dataset location:

```bash
export CITYSCAPES_ROOT=/absolute/path/to/cityscapes
export DETECTRON2_DATASETS="$(dirname "$CITYSCAPES_ROOT")"
```

If your project expects the dataset directly below `datasets/cityscapes`, use:

```bash
export DETECTRON2_DATASETS="$PWD/datasets"
export CITYSCAPES_ROOT="$PWD/datasets/cityscapes"
```

## Pretrained weights

Create a directory for the backbone or pretrained Mask2Former checkpoint:

```bash
mkdir -p models/instance
```

Place the checkpoint in this directory and update `MODEL.WEIGHTS` in the config,
or pass it on the command line as shown in the demo below. Model files are
normally not committed to Git.

## Training

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config-file configs/cityscapes/instance-segmentation/maskformer2_R50_bs8_90k.yaml \
  --num-gpus 1
```

For the batch-size-16 configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config-file configs/cityscapes/instance-segmentation/maskformer2_R50_bs16_90k.yaml \
  --num-gpus 1
```

### Two GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_net.py \
  --config-file configs/cityscapes/instance-segmentation/maskformer2_R50_bs8_90k.yaml \
  --num-gpus 2
```

The effective batch size and learning rate should be checked when the GPU count
is changed.

### Run training in tmux

```bash
tmux new -s train
conda activate maskfo
cd /absolute/path/to/Mask2FormerPro

# Run one of the training commands above.
```

Detach with `Ctrl-b`, then `d`, and reconnect with:

```bash
tmux attach -t train
```

## Monitoring

View training metrics:

```bash
tensorboard --logdir output
```

Monitor GPU utilization:

```bash
watch -n 2 nvidia-smi
```

Inspect evaluation results recorded in a log:

```bash
grep -A 13 "Evaluation results" output/<RUN_NAME>/log.txt | tail -20
```

## Demo and inference

```bash
mkdir -p demo_outputs

python demo.py \
  --config-file configs/cityscapes/instance-segmentation/maskformer2_R50_bs16_90k.yaml \
  --input demo/street.jpg \
  --output demo_outputs/street_out.png \
  --opts MODEL.WEIGHTS models/instance/R50.pkl
```

Replace the config and checkpoint with those used for the experiment.

## Tuning depth-clustering parameters

Run the search from the `depth_clustring_torch` directory, or set
`PYTHONPATH` so its package can be imported.

The following search reproduces the training resize/crop distribution and tests
adaptive minimum cluster size:

```bash
cd depth_clustring_torch

PYTHONPATH=. python tools/tune_cityscapes_disparity.py \
  --cityscapes-root "$CITYSCAPES_ROOT" \
  --split train \
  --device cuda \
  --batch-size 16 \
  --training-augmentations \
  --train-min-sizes 512,614,716,819,921,1024,1126,1228,1331,1433,1536,1638,1740,1843,1945,2048 \
  --train-max-size 4096 \
  --crop-height 512 \
  --crop-width 1024 \
  --augmentation-repeats 1 \
  --adapt-min-size \
  --scale 0.5 \
  --max-images 600 \
  --theta-deg 3,4,5,8,10 \
  --min-size 250,300,600,800,900,1000 \
  --ground true \
  --ground-thresh-deg 2,3,4,5 \
  --ranking-metric pq \
  --false-positive-purity 0.5 \
  --fragment-min-gt-fraction 0.05 \
  --false-positive-penalty 0.02 \
  --split-penalty 0 \
  --query-budget 100 \
  --cluster-penalty 0 \
  --seed 0 \
  --output cluster_search_adaptive_min_size.csv
```

Set `--max-images 0` to evaluate all available images. Search results are
written to the CSV passed to `--output`.

The ranking score can include recall, false-positive, split, and query-count
terms:

```text
score =
    recall_score
    - false_positive_penalty * false_positive_rate
    - split_penalty * split_rate
    - query_penalty
```

### Reference search result

One experiment with training augmentations produced the following best setting:

| Parameter | Value |
| --- | ---: |
| `theta_deg` | 5.0 |
| `min_size` | 600 |
| `ground` | true |
| `ground_thresh_deg` | 3.0 |
| Mean best IoU | 0.1269 |
| Recall@0.50 | 0.1186 |
| Recall@0.75 | 0.0351 |
| Proposal PQ@0.50 | 0.0897 |
| Proposal PQ@0.75 | 0.0313 |
| Clusters per image | 6.83 |

These values are a starting point, not universal defaults. Results depend on
the resize/crop policy, evaluation subset, GPU, and dataset preparation.

## Visualizing the depth pipeline

From the repository root:

```bash
python mask2former/preprocessing/visualize_cityscapes_depth.py \
  --pipeline-file mask2former/preprocessing/depth_pipeline.py \
  --cityscapes-root "$CITYSCAPES_ROOT" \
  --split val \
  --sample frankfurt_000000_000294 \
  --output outputs/frankfurt_depth.png
```

Create the output directory first if required:

```bash
mkdir -p outputs
```

## Troubleshooting

### CUDA extension build lock

If a build was interrupted and a stale lock remains, first make sure no other
process is compiling the extension. Then locate the cache:

```bash
python - <<'PY'
from torch.utils.cpp_extension import _get_build_directory
print(_get_build_directory("depth_cluster_cuda", verbose=False))
PY
```

Remove only the `lock` file in that printed `depth_cluster_cuda` directory and
retry the installation. If the cache itself is corrupt, remove that one
extension directory and rebuild it. Do not delete the entire PyTorch cache.

### `nvcc` not found

```bash
export CUDA_HOME=/usr
export PATH=/usr/bin:$PATH
which nvcc
```

Adapt `CUDA_HOME` if the toolkit is installed elsewhere, for example
`/usr/local/cuda`.

### CUDA architecture mismatch

Set the compute capability that matches the target GPU before rebuilding:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

### Export the tested Conda environment

For an exact record of the working environment:

```bash
conda env export --from-history > maskfo.yml
```

Commit `maskfo.yml` if you want collaborators to reproduce the environment.

## Original Mask2Former

Mask2Former provides a single architecture for panoptic, instance, and semantic
segmentation and supports datasets including ADE20K, Cityscapes, COCO, and
Mapillary Vistas.

- [Original repository](https://github.com/facebookresearch/Mask2Former)
- [Paper](https://arxiv.org/abs/2112.01527)
- [Project page](https://bowenc0221.github.io/mask2former)
- [Original model zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)

## Citation

If you use this work, cite the accompanying paper for this repository when its
citation is available. Also cite Mask2Former:

```bibtex
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and
          Alexander Kirillov and Rohit Girdhar},
  booktitle={CVPR},
  year={2022}
}
```

The depth-clustering component is inspired by:

```bibtex
@inproceedings{bogoslavskyi16iros,
  title={Fast Range Image-Based Segmentation of Sparse 3D Laser Scans for
         Online Operation},
  author={Bogoslavskyi, Igor and Stachniss, Cyrill},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and
             Systems (IROS)},
  year={2016}
}
```

## Acknowledgements

This code is based on
[Mask2Former](https://github.com/facebookresearch/Mask2Former), which is itself
based on [MaskFormer](https://github.com/facebookresearch/MaskFormer).
The depth clustering implementation is inspired by
[PRBonn/depth_clustering](https://github.com/PRBonn/depth_clustering).

## License

The original Mask2Former code is primarily licensed under the MIT License.
Some included or derived components have separate terms:

- Swin-Transformer-Semantic-Segmentation: MIT License
- Deformable DETR: Apache License 2.0
- Detectron2: Apache License 2.0

Retain the original license and notice files, and document the license for any
new code before redistribution.
