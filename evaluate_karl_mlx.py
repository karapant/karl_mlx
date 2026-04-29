"""
Evaluate a trained KARL-MLX checkpoint and produce result visualizations.

Usage:
  python evaluate_karl_mlx.py --checkpoint output_karl_mlx/checkpoint_last.safetensors \
                              --data_path sample_images --output_dir results
"""

import argparse, os, json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import mlx.core as mx
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

import karl_mlx


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_image_np(path: str, size: int = 256) -> np.ndarray:
    """Load image as float32 (H, W, 3) in [0, 1]."""
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(model, image_paths, token_budgets, epsilons, output_dir):
    """Run inference at multiple token budgets / quality thresholds.

    Saves per-image reconstructions and a metrics JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = []

    for img_path in tqdm(image_paths, desc="Evaluating", unit="img"):
        img_np = load_image_np(img_path)
        img_mx = mx.expand_dims(mx.array(img_np), 0)  # (1, H, W, 3)
        fname = Path(img_path).stem

        for T in token_budgets:
            for eps in epsilons:
                recon, active = model.encode(img_mx, token_budget=T, desired_quality=eps)
                mx.eval(recon, active)

                recon_np = np.array(recon[0])
                recon_np = np.clip(recon_np, 0, 1)
                active_count = int(active[0].item())

                # Compute metrics
                l1 = float(np.mean(np.abs(img_np - recon_np)))
                mse = float(np.mean((img_np - recon_np) ** 2))
                psnr = float(psnr_fn(img_np, recon_np, data_range=1.0))
                ssim = float(ssim_fn(img_np, recon_np, data_range=1.0, channel_axis=2))

                tag = f"{fname}_T{T}_eps{eps}"
                metrics.append(dict(
                    image=fname, token_budget=T, epsilon=eps,
                    active_tokens=active_count, l1=l1, mse=mse, psnr=psnr, ssim=ssim,
                ))

                # Save reconstruction
                recon_dir = os.path.join(output_dir, "reconstructions")
                os.makedirs(recon_dir, exist_ok=True)
                Image.fromarray(to_uint8(recon_np)).save(
                    os.path.join(recon_dir, f"{tag}.png"))

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Saved {len(metrics)} metric entries → {metrics_path}")
    return metrics


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

def make_visualizations(metrics, image_paths, output_dir):
    """Generate summary plots from evaluation metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_dir = os.path.join(output_dir, "figures")
    os.makedirs(vis_dir, exist_ok=True)

    # --- 1. Side-by-side grid: original vs reconstructions at different token budgets ---
    recon_dir = os.path.join(output_dir, "reconstructions")
    budgets = sorted(set(m["token_budget"] for m in metrics))
    eps_vals = sorted(set(m["epsilon"] for m in metrics))
    default_eps = eps_vals[len(eps_vals) // 2] if eps_vals else 0.05

    for img_path in image_paths:
        fname = Path(img_path).stem
        orig = load_image_np(img_path)
        row_imgs = [orig]
        row_labels = ["Original"]
        for T in budgets:
            tag = f"{fname}_T{T}_eps{default_eps}"
            rp = os.path.join(recon_dir, f"{tag}.png")
            if os.path.exists(rp):
                row_imgs.append(load_image_np(rp))
                entry = next((m for m in metrics
                              if m["image"] == fname and m["token_budget"] == T
                              and m["epsilon"] == default_eps), None)
                active = entry["active_tokens"] if entry else "?"
                row_labels.append(f"T={T}\n(active={active})")

        ncols = len(row_imgs)
        fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.5))
        if ncols == 1:
            axes = [axes]
        for ax, im, lbl in zip(axes, row_imgs, row_labels):
            ax.imshow(im)
            ax.set_title(lbl, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"KARL Adaptive Tokenization — {fname}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f"grid_{fname}.png"), dpi=150)
        plt.close(fig)

    # --- 2. Metrics vs token budget (averaged over images) ---
    for metric_name in ("l1", "psnr", "ssim"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for eps in eps_vals:
            xs, ys = [], []
            for T in budgets:
                vals = [m[metric_name] for m in metrics
                        if m["token_budget"] == T and m["epsilon"] == eps]
                if vals:
                    xs.append(T)
                    ys.append(np.mean(vals))
            ax.plot(xs, ys, "o-", label=f"ε={eps}")
        ax.set_xlabel("Token budget")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()} vs Token Budget")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f"{metric_name}_vs_tokens.png"), dpi=150)
        plt.close(fig)

    # --- 3. Active tokens (approx KC) per image bar chart ---
    fig, ax = plt.subplots(figsize=(max(6, len(image_paths) * 0.8), 4))
    names, counts = [], []
    for img_path in image_paths:
        fname = Path(img_path).stem
        entry = next((m for m in metrics
                      if m["image"] == fname and m["token_budget"] == max(budgets)
                      and m["epsilon"] == default_eps), None)
        if entry:
            names.append(fname)
            counts.append(entry["active_tokens"])
    ax.bar(names, counts, color="steelblue")
    ax.set_ylabel("Active tokens (≈ KC)")
    ax.set_title(f"Approx. Kolmogorov Complexity (ε={default_eps})")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "approx_kc_bar.png"), dpi=150)
    plt.close(fig)

    print(f"[vis] Saved figures → {vis_dir}/")


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

def print_summary(metrics):
    """Print a compact results table to stdout."""
    budgets = sorted(set(m["token_budget"] for m in metrics))
    epsilons = sorted(set(m["epsilon"] for m in metrics))
    print("\n" + "=" * 72)
    print("KARL-MLX Evaluation Summary")
    print("=" * 72)
    print(f"{'T':>5} {'ε':>6} {'Active':>7} {'L1':>8} {'PSNR':>8} {'SSIM':>8}")
    print("-" * 72)
    for T in budgets:
        for eps in epsilons:
            subset = [m for m in metrics if m["token_budget"] == T and m["epsilon"] == eps]
            if not subset:
                continue
            avg = lambda k: np.mean([m[k] for m in subset])
            print(f"{T:>5} {eps:>6.3f} {avg('active_tokens'):>7.1f} "
                  f"{avg('l1'):>8.4f} {avg('psnr'):>8.2f} {avg('ssim'):>8.4f}")
    print("=" * 72 + "\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate KARL-MLX and produce visualizations")
    p.add_argument("--checkpoint", required=True, help="Path to .safetensors checkpoint")
    p.add_argument("--data_path", required=True, help="Directory of evaluation images")
    p.add_argument("--output_dir", default="results", help="Where to write outputs")
    p.add_argument("--model", default="karl_small", choices=list(karl_mlx.KARL_CONFIGS.keys()))
    p.add_argument("--quantize_latent", action="store_true", default=True)
    p.add_argument("--no_quantize_latent", dest="quantize_latent", action="store_false")
    p.add_argument("--token_budgets", default="32,64,128,256", help="Comma-separated token budgets")
    p.add_argument("--epsilons", default="0.03,0.05,0.09", help="Comma-separated quality thresholds")
    args = p.parse_args()

    token_budgets = [int(x) for x in args.token_budgets.split(",")]
    epsilons = [float(x) for x in args.epsilons.split(",")]

    # Collect images
    image_paths = karl_mlx.image_folder_paths(args.data_path)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.data_path}")
    print(f"[eval] Found {len(image_paths)} images")

    # Load model
    model = karl_mlx.KARLTokenizer(cfg_name=args.model, quantize_latent=args.quantize_latent)
    if os.path.exists(args.checkpoint):
        model.load_weights(args.checkpoint)
        print(f"[eval] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[eval] WARNING: checkpoint not found at {args.checkpoint}, using random weights")

    # Run
    metrics = evaluate(model, image_paths, token_budgets, epsilons, args.output_dir)
    print_summary(metrics)
    make_visualizations(metrics, image_paths, args.output_dir)
    print(f"[eval] Done. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
