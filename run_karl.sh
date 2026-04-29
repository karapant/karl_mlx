#!/usr/bin/env bash
# ===================================================================
# run_karl.sh — End-to-end KARL replication pipeline
#
# Supports three run modes:
#   1. Apple MLX Optimized  — runs karl_mlx.py on Apple Silicon
#   2. Vanilla KARL (PyTorch) — runs the original repo in karl/
#   3. Both (sequentially)  — runs Vanilla first, then MLX
#
# Usage:
#   chmod +x run_karl.sh
#   ./run_karl.sh                          # interactive mode menu
#   ./run_karl.sh --mode mlx               # skip menu, run MLX
#   ./run_karl.sh --mode vanilla            # skip menu, run Vanilla
#   ./run_karl.sh --mode both               # skip menu, run both
#   ./run_karl.sh --data_path /path/to/data # use your own data
#   ./run_karl.sh --skip_train              # eval-only
# ===================================================================
set -euo pipefail

# ---------------------------------------------------------------
# Defaults
#
# Paper settings (8×GPU, ImageNet-100 ~130k images):
#   Pretrain:  200 epochs, batch_size 64×8 GPUs
#   Finetune:  400 epochs, batch_size 18×8 GPUs
#
# Demo defaults below are scaled down for single machine.
# ---------------------------------------------------------------
DATA_PATH="${DATA_PATH:-}"
MODEL="${MODEL:-karl_small}"
EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-50}"
EPOCHS_FINETUNE="${EPOCHS_FINETUNE:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
NUM_IMAGES="${NUM_IMAGES:-1300}"
OUTPUT_DIR="${OUTPUT_DIR:-output_karl_mlx}"
RESULTS_DIR="${RESULTS_DIR:-results}"
RUN_MODE=""
SKIP_TRAIN=0
SKIP_DEPS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)             RUN_MODE="$2"; shift 2 ;;
        --data_path)        DATA_PATH="$2"; shift 2 ;;
        --model)            MODEL="$2"; shift 2 ;;
        --epochs)           EPOCHS_PRETRAIN="$2"; EPOCHS_FINETUNE="$2"; shift 2 ;;
        --epochs_pretrain)  EPOCHS_PRETRAIN="$2"; shift 2 ;;
        --epochs_finetune)  EPOCHS_FINETUNE="$2"; shift 2 ;;
        --batch_size)       BATCH_SIZE="$2"; shift 2 ;;
        --lr)               LR="$2"; shift 2 ;;
        --num_images)       NUM_IMAGES="$2"; shift 2 ;;
        --output_dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --results_dir)      RESULTS_DIR="$2"; shift 2 ;;
        --skip_train)       SKIP_TRAIN=1; shift ;;
        --skip_deps)        SKIP_DEPS=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------
# Mode selection menu
# ---------------------------------------------------------------
if [[ -z "$RUN_MODE" ]]; then
    echo ""
    echo "============================================================"
    echo " KARL — Kolmogorov-Approximating Representation Learning"
    echo "============================================================"
    echo ""
    echo " Select run mode:"
    echo ""
    echo "   1) Apple MLX Optimized"
    echo "      Runs karl_mlx.py on Apple Silicon (M1/M2/M3/M4)."
    echo "      Faithful port of the paper's architecture to MLX."
    echo ""
    echo "   2) Vanilla KARL (PyTorch)"
    echo "      Runs the original PyTorch implementation in karl/."
    echo "      Requires CUDA GPU and PyTorch + torchvision installed."
    echo ""
    echo "   3) Both (sequentially)"
    echo "      Runs Vanilla KARL first, then Apple MLX."
    echo "      Useful for comparing outputs side by side."
    echo ""
    echo "============================================================"
    echo ""
    read -rp " Enter choice [1/2/3]: " choice
    case "$choice" in
        1) RUN_MODE="mlx" ;;
        2) RUN_MODE="vanilla" ;;
        3) RUN_MODE="both" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
    echo ""
fi

# Normalize mode
RUN_MODE=$(echo "$RUN_MODE" | tr '[:upper:]' '[:lower:]')
case "$RUN_MODE" in
    mlx|1) RUN_MODE="mlx" ;;
    vanilla|pytorch|2) RUN_MODE="vanilla" ;;
    both|3) RUN_MODE="both" ;;
    *) echo "Unknown mode: $RUN_MODE (use mlx, vanilla, or both)"; exit 1 ;;
esac

# ===============================================================
# Shared: Install deps + download data
# ===============================================================

install_deps() {
    if [[ "$SKIP_DEPS" -eq 1 ]]; then
        echo ">>> Skipping dependency install (--skip_deps)."
        return
    fi
    echo ">>> Installing Python dependencies..."
    if [[ "$1" == "mlx" ]]; then
        pip3 install --quiet mlx numpy Pillow scikit-image matplotlib tqdm requests \
            2>&1 | grep -v "^ERROR: pip's dependency" || true
    elif [[ "$1" == "vanilla" ]]; then
        pip3 install --quiet torch torchvision timm omegaconf pytorch-lightning \
            numpy Pillow scikit-image matplotlib tqdm requests wandb \
            2>&1 | grep -v "^ERROR: pip's dependency" || true
    fi
    echo "    Done."
}

download_data() {
    if [[ -n "$DATA_PATH" ]]; then
        echo ">>> Using provided data path: $DATA_PATH"
        return
    fi
    echo ">>> Downloading ImageNet-100 sample images (exact paper synsets)..."
    echo "    Target: $NUM_IMAGES images (~$((NUM_IMAGES / 100)) per class)"
    python3 download_imagenet100_samples.py \
        --output_dir sample_images \
        --num_images "$NUM_IMAGES" \
        --num_per_class "$((NUM_IMAGES / 100 + 1))"
    DATA_PATH="sample_images"
    echo "    Training data: $DATA_PATH"
}

# ===============================================================
# MLX Pipeline
# ===============================================================

run_mlx() {
    local out_dir="${OUTPUT_DIR}"
    local res_dir="${RESULTS_DIR}"

    echo ""
    echo "============================================================"
    echo " Apple MLX Optimized Pipeline"
    echo "============================================================"
    echo " Model: $MODEL | Pretrain: $EPOCHS_PRETRAIN ep | Finetune: $EPOCHS_FINETUNE ep"
    echo "============================================================"

    install_deps "mlx"
    download_data

    local pretrain_ckpt="$out_dir/checkpoint_last.safetensors"
    local final_ckpt=""

    if [[ "$SKIP_TRAIN" -eq 0 ]]; then
        echo ""
        echo ">>> [MLX] Stage 1 — Latent-distillation pretrain ($EPOCHS_PRETRAIN epochs)..."
        python3 karl_mlx.py \
            --stage pretrain --model "$MODEL" \
            --data_path "$DATA_PATH" --output_dir "$out_dir" \
            --epochs "$EPOCHS_PRETRAIN" --batch_size "$BATCH_SIZE" \
            --lr "$LR" --quantize_latent --save_every 10
        echo "    Checkpoint: $pretrain_ckpt"

        echo ""
        echo ">>> [MLX] Stage 2 — Full finetuning ($EPOCHS_FINETUNE epochs)..."
        local ft_dir="${out_dir}_finetune"
        python3 karl_mlx.py \
            --stage finetune --model "$MODEL" \
            --data_path "$DATA_PATH" --output_dir "$ft_dir" \
            --finetune "$pretrain_ckpt" \
            --epochs "$EPOCHS_FINETUNE" --batch_size "$BATCH_SIZE" \
            --lr "$LR" --quantize_latent --save_every 10
        final_ckpt="$ft_dir/checkpoint_last.safetensors"
    else
        echo ">>> [MLX] Skipping training (--skip_train)."
        if [[ -f "${out_dir}_finetune/checkpoint_last.safetensors" ]]; then
            final_ckpt="${out_dir}_finetune/checkpoint_last.safetensors"
        elif [[ -f "$pretrain_ckpt" ]]; then
            final_ckpt="$pretrain_ckpt"
        else
            final_ckpt="none"
            echo "    WARNING: No checkpoint found."
        fi
    fi

    echo ""
    echo ">>> [MLX] Evaluating..."
    local eval_dir="eval_images"
    mkdir -p "$eval_dir"
    local ec
    ec=$(find "$eval_dir" -maxdepth 1 \( -name '*.jpg' -o -name '*.png' -o -name '*.JPEG' \) 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$ec" -lt 5 ]]; then
        find "$DATA_PATH" -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.JPEG' \) | head -10 | while read -r f; do
            cp "$f" "$eval_dir/" 2>/dev/null || true
        done
    fi
    python3 evaluate_karl_mlx.py \
        --checkpoint "$final_ckpt" --data_path "$eval_dir" \
        --output_dir "$res_dir" --model "$MODEL" --quantize_latent \
        --token_budgets "32,64,128,256" --epsilons "0.03,0.05,0.09"

    echo ""
    echo " [MLX] Results in $res_dir/"
}

# ===============================================================
# Vanilla KARL (PyTorch) Pipeline
# ===============================================================

run_vanilla() {
    local out_base="output_karl_vanilla"
    local res_dir="${RESULTS_DIR}_vanilla"

    echo ""
    echo "============================================================"
    echo " Vanilla KARL (PyTorch) Pipeline"
    echo "============================================================"
    echo " Model: $MODEL | Pretrain: $EPOCHS_PRETRAIN ep | Finetune: $EPOCHS_FINETUNE ep"
    echo "============================================================"

    install_deps "vanilla"
    download_data

    # Check for CUDA
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo ""
        echo "    WARNING: CUDA not available. Vanilla KARL requires a CUDA GPU."
        echo "    Attempting CPU fallback (will be very slow)..."
        DEVICE="cpu"
        NPROC=1
    else
        DEVICE="cuda"
        NPROC=$(python3 -c "import torch; print(torch.cuda.device_count())")
        echo "    Found $NPROC CUDA GPU(s)."
    fi

    # Download pretrained VQGAN weights if needed
    echo ""
    echo ">>> [Vanilla] Downloading pretrained base tokenizer weights..."
    (cd karl && python3 -c "
from kolmogorov_tokenizers.pretrained_models.download import download_all
download_all()
" 2>/dev/null) || echo "    (Download skipped or already present)"

    local pretrain_out="$out_base/pretrain"
    local finetune_out="$out_base/finetune"

    if [[ "$SKIP_TRAIN" -eq 0 ]]; then
        echo ""
        echo ">>> [Vanilla] Stage 1 — Latent-distillation pretrain ($EPOCHS_PRETRAIN epochs)..."
        (cd karl && torchrun --nproc_per_node="$NPROC" --master_port=12345 \
            main_pretrain.py \
            --batch_size "$BATCH_SIZE" \
            --num_workers 4 \
            --model karl_small \
            --base_tokenizer vqgan \
            --quantize_latent --factorize_latent \
            --epochs "$EPOCHS_PRETRAIN" \
            --warmup_epochs 40 \
            --blr 1e-3 --weight_decay 0.05 \
            --grad_clip 3.0 \
            --output_dir "../$pretrain_out" \
            --data_path "../$DATA_PATH" \
            --device "$DEVICE")
        echo "    Checkpoint: $pretrain_out/checkpoint-last.pth"

        echo ""
        echo ">>> [Vanilla] Stage 2 — Full finetuning ($EPOCHS_FINETUNE epochs)..."
        (cd karl && torchrun --nproc_per_node="$NPROC" --master_port=12345 \
            main_full_finetuning.py \
            --batch_size "$BATCH_SIZE" \
            --model karl_small \
            --base_tokenizer vqgan \
            --quantize_latent --factorize_latent \
            --epochs "$EPOCHS_FINETUNE" \
            --warmup_epochs 40 \
            --blr 1e-3 --weight_decay 0.05 \
            --grad_clip 3.0 \
            --finetune "../$pretrain_out/checkpoint-last.pth" \
            --output_dir "../$finetune_out" \
            --data_path "../$DATA_PATH" \
            --device "$DEVICE")
        echo "    Checkpoint: $finetune_out/checkpoint-last.pth"
    else
        echo ">>> [Vanilla] Skipping training (--skip_train)."
    fi

    # Evaluate
    if [[ -f "$finetune_out/checkpoint-last.pth" ]] || [[ -f "$pretrain_out/checkpoint-last.pth" ]]; then
        local ckpt
        if [[ -f "$finetune_out/checkpoint-last.pth" ]]; then
            ckpt="$finetune_out/checkpoint-last.pth"
        else
            ckpt="$pretrain_out/checkpoint-last.pth"
        fi
        echo ""
        echo ">>> [Vanilla] Evaluating..."
        (cd karl && python3 evaluate.py \
            --model karl_small \
            --base_tokenizer vqgan \
            --quantize_latent \
            --output_dir "../$res_dir" \
            --ckpt "../$ckpt" \
            --data_path "../$DATA_PATH")
        echo " [Vanilla] Results in $res_dir/"
    else
        echo "    No checkpoint found for evaluation."
    fi
}

# ===============================================================
# Dispatch
# ===============================================================

echo ""
echo "============================================================"
echo " KARL — Run Mode: $(echo "$RUN_MODE" | tr '[:lower:]' '[:upper:]')"
echo "============================================================"

case "$RUN_MODE" in
    mlx)
        run_mlx
        ;;
    vanilla)
        run_vanilla
        ;;
    both)
        run_vanilla
        echo ""
        echo "============================================================"
        echo " Vanilla complete. Now running MLX..."
        echo "============================================================"
        run_mlx
        ;;
esac

echo ""
echo "============================================================"
echo " KARL Pipeline Complete! (mode: $RUN_MODE)"
echo "============================================================"
