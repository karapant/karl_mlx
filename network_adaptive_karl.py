"""
Network-Adaptive KARL: Channel-Aware Compression Experiments
=============================================================
Simulates edge-to-cloud transmission over a wireless channel with fixed bandwidth
and varying SINR. Compares:
  - Vanilla KARL: fixed ε regardless of channel state
  - Channel-Adaptive KARL: ε driven by SINR → Shannon capacity → token budget

Usage:
  python network_adaptive_karl.py \
      --checkpoint output_karl_mlx/checkpoint_last.safetensors \
      --data_path sample_images --output_dir results_network
"""

import argparse, json, math, os
from pathlib import Path

import numpy as np
import mlx.core as mx
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from tqdm import tqdm

import karl_mlx

# ------------------------------------------------------------------
# Channel model
# ------------------------------------------------------------------

BITS_PER_TOKEN = 12  # log2(4096) codebook entries

def shannon_capacity(bandwidth_hz: float, sinr_linear: float) -> float:
    """Shannon capacity in bits/sec."""
    return bandwidth_hz * math.log2(1 + sinr_linear)

def sinr_db_to_linear(sinr_db: float) -> float:
    return 10 ** (sinr_db / 10)

def max_tokens_per_frame(sinr_db: float, bandwidth_hz: float,
                         frame_rate: float = 30.0, efficiency: float = 0.7) -> int:
    """Compute max transmittable tokens given channel state."""
    sinr_lin = sinr_db_to_linear(sinr_db)
    cap = shannon_capacity(bandwidth_hz, sinr_lin)
    r_eff = efficiency * cap
    bits_per_frame = r_eff / frame_rate
    return int(bits_per_frame / BITS_PER_TOKEN)

def token_budget_to_epsilon(t_max: int) -> float:
    """Map available token budget → ε. Learned relationship from KARL training.
    Uses a simple piecewise-linear approximation of the ε↔token-count curve."""
    # Clamp to KARL's operating range
    t_max = max(16, min(t_max, 256))
    # Approximate inverse: more tokens → lower ε
    # Based on KARL's 19 bins: [0.005 ... 0.38]
    eps = 0.38 * (1 - (t_max - 16) / (256 - 16)) ** 1.5 + 0.005
    return round(min(max(eps, 0.005), 0.38), 4)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_image_np(path: str, size: int = 256) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

def compute_metrics(orig: np.ndarray, recon: np.ndarray) -> dict:
    recon = np.clip(recon, 0, 1)
    return dict(
        l1=float(np.mean(np.abs(orig - recon))),
        mse=float(np.mean((orig - recon) ** 2)),
        psnr=float(psnr_fn(orig, recon, data_range=1.0)),
        ssim=float(ssim_fn(orig, recon, data_range=1.0, channel_axis=2)),
    )

def bits_transmitted(active_tokens: int) -> int:
    return active_tokens * BITS_PER_TOKEN

# ------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------

def run_experiment(model, image_paths, sinr_values_db, bandwidth_hz,
                   vanilla_epsilon, frame_rate, efficiency, output_dir):
    """Run vanilla vs adaptive KARL across SINR conditions."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for sinr_db in sinr_values_db:
        t_max = max_tokens_per_frame(sinr_db, bandwidth_hz, frame_rate, efficiency)
        adaptive_eps = token_budget_to_epsilon(t_max)
        # Clamp token budget to KARL's max
        t_max_clamped = min(t_max, 256)

        print(f"\n--- SINR={sinr_db} dB | T_max={t_max} | adaptive ε={adaptive_eps:.4f} ---")

        for img_path in tqdm(image_paths, desc=f"SINR={sinr_db}dB", leave=False):
            img_np = load_image_np(img_path)
            img_mx = mx.expand_dims(mx.array(img_np), 0)
            fname = Path(img_path).stem

            # --- Vanilla KARL: fixed ε, full token budget ---
            recon_v, active_v = model.encode(img_mx, token_budget=256,
                                             desired_quality=vanilla_epsilon)
            mx.eval(recon_v, active_v)
            active_v_int = int(active_v[0].item())
            metrics_v = compute_metrics(img_np, np.array(recon_v[0]))

            # Vanilla transmits all active tokens — may exceed channel capacity
            vanilla_bits = bits_transmitted(active_v_int)
            vanilla_deliverable = active_v_int <= t_max_clamped
            # If exceeds capacity: frame is delayed/dropped
            vanilla_effective_tokens = min(active_v_int, t_max_clamped)

            # --- Adaptive KARL: ε set by channel state ---
            recon_a, active_a = model.encode(img_mx, token_budget=t_max_clamped,
                                             desired_quality=adaptive_eps)
            mx.eval(recon_a, active_a)
            active_a_int = int(active_a[0].item())
            metrics_a = compute_metrics(img_np, np.array(recon_a[0]))

            adaptive_bits = bits_transmitted(active_a_int)
            adaptive_deliverable = True  # Always fits by design

            results.append(dict(
                image=fname, sinr_db=sinr_db, t_max=t_max, t_max_clamped=t_max_clamped,
                bandwidth_mhz=bandwidth_hz / 1e6,
                # Vanilla
                vanilla_epsilon=vanilla_epsilon,
                vanilla_active_tokens=active_v_int,
                vanilla_bits=vanilla_bits,
                vanilla_deliverable=vanilla_deliverable,
                vanilla_effective_tokens=vanilla_effective_tokens,
                **{f"vanilla_{k}": v for k, v in metrics_v.items()},
                # Adaptive
                adaptive_epsilon=adaptive_eps,
                adaptive_active_tokens=active_a_int,
                adaptive_bits=adaptive_bits,
                adaptive_deliverable=adaptive_deliverable,
                **{f"adaptive_{k}": v for k, v in metrics_a.items()},
            ))

    # Save results
    results_path = os.path.join(output_dir, "network_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[network] Saved {len(results)} entries → {results_path}")
    return results

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

def make_plots(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sinr_values = sorted(set(r["sinr_db"] for r in results))

    # --- 1. PSNR vs SINR ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vanilla_psnr = [np.mean([r["vanilla_psnr"] for r in results if r["sinr_db"] == s])
                    for s in sinr_values]
    adaptive_psnr = [np.mean([r["adaptive_psnr"] for r in results if r["sinr_db"] == s])
                     for s in sinr_values]
    ax.plot(sinr_values, vanilla_psnr, "o--", label="Vanilla KARL (fixed ε)", color="tab:red")
    ax.plot(sinr_values, adaptive_psnr, "s-", label="Channel-Adaptive KARL", color="tab:blue")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Reconstruction Quality vs Channel Condition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "psnr_vs_sinr.png"), dpi=150)
    plt.close(fig)

    # --- 2. Active tokens vs SINR ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vanilla_tokens = [np.mean([r["vanilla_active_tokens"] for r in results if r["sinr_db"] == s])
                      for s in sinr_values]
    adaptive_tokens = [np.mean([r["adaptive_active_tokens"] for r in results if r["sinr_db"] == s])
                       for s in sinr_values]
    t_max_vals = [next(r["t_max_clamped"] for r in results if r["sinr_db"] == s)
                  for s in sinr_values]
    ax.plot(sinr_values, vanilla_tokens, "o--", label="Vanilla KARL tokens", color="tab:red")
    ax.plot(sinr_values, adaptive_tokens, "s-", label="Adaptive KARL tokens", color="tab:blue")
    ax.plot(sinr_values, t_max_vals, "k--", alpha=0.5, label="Channel capacity (T_max)")
    ax.fill_between(sinr_values, t_max_vals, [max(vanilla_tokens)] * len(sinr_values),
                    alpha=0.1, color="red", label="Overflow region (vanilla)")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("Tokens per frame")
    ax.set_title("Token Count vs Channel Condition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "tokens_vs_sinr.png"), dpi=150)
    plt.close(fig)

    # --- 3. Delivery success rate ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vanilla_delivery = [np.mean([r["vanilla_deliverable"] for r in results if r["sinr_db"] == s]) * 100
                        for s in sinr_values]
    adaptive_delivery = [100.0 for _ in sinr_values]  # Always 100% by design
    ax.plot(sinr_values, vanilla_delivery, "o--", label="Vanilla KARL", color="tab:red")
    ax.plot(sinr_values, adaptive_delivery, "s-", label="Adaptive KARL", color="tab:blue")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("Frame delivery rate (%)")
    ax.set_title("Frames Deliverable Within Channel Capacity")
    ax.set_ylim(-5, 110)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "delivery_vs_sinr.png"), dpi=150)
    plt.close(fig)

    # --- 4. Bits transmitted vs SINR ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vanilla_bits = [np.mean([r["vanilla_bits"] for r in results if r["sinr_db"] == s])
                    for s in sinr_values]
    adaptive_bits = [np.mean([r["adaptive_bits"] for r in results if r["sinr_db"] == s])
                     for s in sinr_values]
    capacity_bits = [t * BITS_PER_TOKEN for t in t_max_vals]
    ax.plot(sinr_values, vanilla_bits, "o--", label="Vanilla KARL (attempted)", color="tab:red")
    ax.plot(sinr_values, adaptive_bits, "s-", label="Adaptive KARL", color="tab:blue")
    ax.plot(sinr_values, capacity_bits, "k--", alpha=0.5, label="Channel capacity")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("Bits per frame")
    ax.set_title("Transmission Load vs Channel Capacity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bits_vs_sinr.png"), dpi=150)
    plt.close(fig)

    # --- 5. Summary table ---
    print("\n" + "=" * 90)
    print("Network-Adaptive KARL — Experiment Summary")
    print("=" * 90)
    print(f"{'SINR(dB)':>8} {'T_max':>6} {'ε_adapt':>8} | "
          f"{'V_tokens':>8} {'V_PSNR':>7} {'V_deliv':>7} | "
          f"{'A_tokens':>8} {'A_PSNR':>7} {'A_deliv':>7}")
    print("-" * 90)
    for i, s in enumerate(sinr_values):
        print(f"{s:>8.1f} {t_max_vals[i]:>6} {token_budget_to_epsilon(t_max_vals[i]):>8.4f} | "
              f"{vanilla_tokens[i]:>8.1f} {vanilla_psnr[i]:>7.2f} {vanilla_delivery[i]:>6.1f}% | "
              f"{adaptive_tokens[i]:>8.1f} {adaptive_psnr[i]:>7.2f} {adaptive_delivery[i]:>6.1f}%")
    print("=" * 90)

    print(f"\n[network] Figures saved → {fig_dir}/")

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Network-Adaptive KARL experiments")
    p.add_argument("--checkpoint", required=True, help="KARL checkpoint (.safetensors)")
    p.add_argument("--data_path", required=True, help="Directory of evaluation images")
    p.add_argument("--output_dir", default="results_network", help="Output directory")
    p.add_argument("--model", default="karl_small", choices=list(karl_mlx.KARL_CONFIGS.keys()))
    p.add_argument("--quantize_latent", action="store_true", default=True)
    p.add_argument("--no_quantize_latent", dest="quantize_latent", action="store_false")
    # Channel parameters
    p.add_argument("--bandwidth_mhz", type=float, default=20.0,
                   help="Channel bandwidth in MHz (fixed across experiment)")
    p.add_argument("--sinr_min", type=float, default=-3.0, help="Min SINR in dB")
    p.add_argument("--sinr_max", type=float, default=25.0, help="Max SINR in dB")
    p.add_argument("--sinr_steps", type=int, default=8, help="Number of SINR points")
    p.add_argument("--frame_rate", type=float, default=30.0, help="Camera frame rate (fps)")
    p.add_argument("--efficiency", type=float, default=0.7, help="Channel efficiency η")
    # Vanilla baseline
    p.add_argument("--vanilla_epsilon", type=float, default=0.05,
                   help="Fixed ε for vanilla KARL baseline")
    args = p.parse_args()

    bandwidth_hz = args.bandwidth_mhz * 1e6
    sinr_values = np.linspace(args.sinr_min, args.sinr_max, args.sinr_steps).tolist()

    # Print channel parameters
    print("=" * 60)
    print("Network-Adaptive KARL Experiment")
    print("=" * 60)
    print(f"  Bandwidth:       {args.bandwidth_mhz} MHz")
    print(f"  SINR range:      {args.sinr_min} to {args.sinr_max} dB ({args.sinr_steps} steps)")
    print(f"  Frame rate:      {args.frame_rate} fps")
    print(f"  Efficiency (η):  {args.efficiency}")
    print(f"  Vanilla ε:       {args.vanilla_epsilon}")
    print(f"  Bits/token:      {BITS_PER_TOKEN}")
    print()
    print("  SINR → T_max → ε mapping:")
    for s in sinr_values:
        t = max_tokens_per_frame(s, bandwidth_hz, args.frame_rate, args.efficiency)
        t_c = min(t, 256)
        eps = token_budget_to_epsilon(t_c)
        print(f"    {s:>6.1f} dB → T_max={t:>6} → clamped={t_c:>3} → ε={eps:.4f}")
    print("=" * 60)

    # Load images
    image_paths = karl_mlx.image_folder_paths(args.data_path)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.data_path}")
    print(f"\n[network] Found {len(image_paths)} images")

    # Load model
    model = karl_mlx.KARLTokenizer(cfg_name=args.model, quantize_latent=args.quantize_latent)
    if os.path.exists(args.checkpoint):
        model.load_weights(args.checkpoint)
        print(f"[network] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[network] WARNING: checkpoint not found, using random weights")

    # Run
    results = run_experiment(model, image_paths, sinr_values, bandwidth_hz,
                             args.vanilla_epsilon, args.frame_rate, args.efficiency,
                             args.output_dir)
    make_plots(results, args.output_dir)


if __name__ == "__main__":
    main()
