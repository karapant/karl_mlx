#!/usr/bin/env bash
# run_network_experiments.sh — Train KARL + run channel-adaptive experiments
# Trains KARL (FAST or SLOW), then compares vanilla vs channel-adaptive KARL
set -euo pipefail

# Defaults (override via environment or flags)
CHECKPOINT="${CHECKPOINT:-output_karl_mlx/checkpoint_last.safetensors}"
DATA_PATH="${DATA_PATH:-sample_images}"
OUTPUT_DIR="${OUTPUT_DIR:-results_network}"
BANDWIDTH_MHZ="${BANDWIDTH_MHZ:-20.0}"
SINR_MIN="${SINR_MIN:--3.0}"
SINR_MAX="${SINR_MAX:-25.0}"
SINR_STEPS="${SINR_STEPS:-8}"
FRAME_RATE="${FRAME_RATE:-30.0}"
VANILLA_EPSILON="${VANILLA_EPSILON:-0.05}"
TRAIN_SPEED=""
SKIP_TRAIN=0

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)    CHECKPOINT="$2"; shift 2;;
        --data_path)     DATA_PATH="$2"; shift 2;;
        --output_dir)    OUTPUT_DIR="$2"; shift 2;;
        --bandwidth_mhz) BANDWIDTH_MHZ="$2"; shift 2;;
        --sinr_min)      SINR_MIN="$2"; shift 2;;
        --sinr_max)      SINR_MAX="$2"; shift 2;;
        --sinr_steps)    SINR_STEPS="$2"; shift 2;;
        --frame_rate)    FRAME_RATE="$2"; shift 2;;
        --vanilla_epsilon) VANILLA_EPSILON="$2"; shift 2;;
        --fast)          TRAIN_SPEED="fast"; shift;;
        --slow)          TRAIN_SPEED="slow"; shift;;
        --skip_train)    SKIP_TRAIN=1; shift;;
        *) echo "Unknown flag: $1"; exit 1;;
    esac
done

echo "=============================================="
echo " Network-Adaptive KARL Experiments"
echo "=============================================="

# Check dependencies
python3 -c "import mlx, numpy, PIL, skimage, matplotlib" 2>/dev/null || {
    echo "[setup] Installing dependencies..."
    pip install mlx numpy Pillow scikit-image matplotlib tqdm requests
}

# Create data dir if missing
mkdir -p "$DATA_PATH"

# ---------------------------------------------------------------
# Training step
# ---------------------------------------------------------------
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    # Ask FAST or SLOW if not specified via flag
    if [[ -z "$TRAIN_SPEED" ]]; then
        echo ""
        echo " Select training speed:"
        echo ""
        echo "   1) FAST  — ~1 hour (10 pretrain + 10 finetune epochs, 100 images)"
        echo "   2) SLOW  — ~1 day  (50 pretrain + 50 finetune epochs, 1300 images)"
        echo ""
        read -rp " Enter choice [1/2]: " speed_choice
        case "$speed_choice" in
            1|fast|FAST)  TRAIN_SPEED="fast" ;;
            2|slow|SLOW)  TRAIN_SPEED="slow" ;;
            *) echo "Invalid choice, defaulting to FAST."; TRAIN_SPEED="fast" ;;
        esac
    fi

    if [[ "$TRAIN_SPEED" == "fast" ]]; then
        EPOCHS_PRETRAIN=10
        EPOCHS_FINETUNE=10
        BATCH_SIZE=8
        NUM_IMAGES=100
    else
        EPOCHS_PRETRAIN=50
        EPOCHS_FINETUNE=50
        BATCH_SIZE=8
        NUM_IMAGES=1300
    fi

    echo ""
    echo " Training mode: $(echo "$TRAIN_SPEED" | tr '[:lower:]' '[:upper:]')"
    echo " Pretrain: $EPOCHS_PRETRAIN epochs | Finetune: $EPOCHS_FINETUNE epochs"
    echo " Images: $NUM_IMAGES | Batch size: $BATCH_SIZE"
    echo "=============================================="

    # Ensure we have enough data
    img_count=$(find "$DATA_PATH" -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.JPEG' \) 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$img_count" -lt "$NUM_IMAGES" ]]; then
        echo "[setup] Need $NUM_IMAGES images, found $img_count. Downloading..."
        python3 download_imagenet100_samples.py --output_dir "$DATA_PATH" --num_images "$NUM_IMAGES"
    fi

    TRAIN_OUT="output_karl_mlx"

    echo ""
    echo ">>> Stage 1 — Latent-distillation pretrain ($EPOCHS_PRETRAIN epochs)..."
    python3 karl_mlx.py \
        --stage pretrain --model karl_small \
        --data_path "$DATA_PATH" --output_dir "$TRAIN_OUT" \
        --epochs "$EPOCHS_PRETRAIN" --batch_size "$BATCH_SIZE" \
        --quantize_latent --save_every 5

    echo ""
    echo ">>> Stage 2 — Full finetuning ($EPOCHS_FINETUNE epochs)..."
    python3 karl_mlx.py \
        --stage finetune --model karl_small \
        --data_path "$DATA_PATH" --output_dir "${TRAIN_OUT}_finetune" \
        --finetune "$TRAIN_OUT/checkpoint_last.safetensors" \
        --epochs "$EPOCHS_FINETUNE" --batch_size "$BATCH_SIZE" \
        --quantize_latent --save_every 5

    # Use finetuned checkpoint for experiments
    if [[ -f "${TRAIN_OUT}_finetune/checkpoint_last.safetensors" ]]; then
        CHECKPOINT="${TRAIN_OUT}_finetune/checkpoint_last.safetensors"
    else
        CHECKPOINT="$TRAIN_OUT/checkpoint_last.safetensors"
    fi

    echo ""
    echo ">>> Training complete. Checkpoint: $CHECKPOINT"
else
    echo " Skipping training (--skip_train)."
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "[warn] Checkpoint not found at $CHECKPOINT"
        echo "       Proceeding with random weights (results will be illustrative only)..."
    fi
fi

echo ""
echo "=============================================="
echo " Running Network-Adaptive Experiments"
echo "=============================================="
echo " Checkpoint:  $CHECKPOINT"
echo " Data:        $DATA_PATH"
echo " Output:      $OUTPUT_DIR"
echo " Bandwidth:   ${BANDWIDTH_MHZ} MHz"
echo " SINR range:  ${SINR_MIN} to ${SINR_MAX} dB (${SINR_STEPS} steps)"
echo " Frame rate:  ${FRAME_RATE} fps"
echo " Vanilla ε:   ${VANILLA_EPSILON}"
echo "=============================================="

# Run experiment
python3 network_adaptive_karl.py \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --bandwidth_mhz "$BANDWIDTH_MHZ" \
    --sinr_min "$SINR_MIN" \
    --sinr_max "$SINR_MAX" \
    --sinr_steps "$SINR_STEPS" \
    --frame_rate "$FRAME_RATE" \
    --vanilla_epsilon "$VANILLA_EPSILON"

echo ""
echo "[done] Results saved to $OUTPUT_DIR/"
echo "       - $OUTPUT_DIR/network_metrics.json"
echo "       - $OUTPUT_DIR/figures/"
