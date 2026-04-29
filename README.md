# KARL — MLX + PyTorch Replication

MLX and PyTorch replication of KARL, a single-pass adaptive image tokenizer that predicts the minimum number of tokens needed to reconstruct an image by approximating its Kolmogorov Complexity.

> **Single-pass Adaptive Image Tokenization for Minimum Program Search**
> Duggal, Byun, Freeman, Torralba, Isola — MIT CSAIL
> [arXiv:2507.07995](https://arxiv.org/abs/2507.07995)

## Quick Start

```bash
./run_karl.sh                  # interactive mode menu
./run_karl.sh --mode mlx       # Apple Silicon (M1–M4)
./run_karl.sh --mode vanilla   # PyTorch + CUDA
./run_karl.sh --mode both      # both sequentially
```

The pipeline installs dependencies, downloads ~1,300 ImageNet-100 samples, trains (pretrain + finetune), and evaluates.

### Custom data / paper-scale training

```bash
./run_karl.sh --mode mlx --data_path /path/to/imagenet100 \
    --epochs_pretrain 200 --epochs_finetune 400 --batch_size 8
```

### Evaluate only

```bash
./run_karl.sh --mode mlx --skip_train
```

## MLX Training (`karl_mlx.py`)

```bash
python karl_mlx.py --stage pretrain --data_path /path/to/data --epochs 200
python karl_mlx.py --stage finetune --data_path /path/to/data \
    --finetune output_karl_mlx/checkpoint_last.safetensors --epochs 400
```

Key flags: `--model` (karl_small/karl_tiny), `--batch_size`, `--lr`, `--vqgan_ckpt`, `--grad_clip`, `--warmup_epochs`.

## Vanilla KARL (PyTorch)

The original [ShivamDuggal4/karl](https://github.com/ShivamDuggal4/karl) repo is cloned automatically when running `--mode vanilla`. Requires CUDA.

## Network-Adaptive KARL

Channel-aware compression where ε is driven by wireless SINR — useful for edge devices transmitting over bandwidth-constrained links.

```bash
./run_network_experiments.sh
```

Compares fixed-ε (vanilla) vs SINR-adaptive ε. Outputs metrics and figures to `results_network/`.

## Requirements

**MLX:** Python 3.10+, Apple Silicon — `pip install mlx numpy Pillow scikit-image matplotlib tqdm requests`

**PyTorch:** Python 3.10+, CUDA — `pip install torch torchvision timm omegaconf pytorch-lightning wandb numpy Pillow scikit-image matplotlib tqdm requests`

Dependencies are installed automatically by `run_karl.sh`.

## Cleanup

```bash
./cleanup.sh        # remove generated outputs and downloaded data
./cleanup.sh --all  # also remove cloned karl/ repo and model weights
```

## Citation

```bibtex
@article{duggal2024KARL,
  author  = {Shivam Duggal and Sanghyun Byun and William T. Freeman and Antonio Torralba and Phillip Isola},
  title   = {Single-pass Adaptive Image Tokenization for Minimum Program Search},
  journal = {arxiv},
  year    = {2025}
}
```
