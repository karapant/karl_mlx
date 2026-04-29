#!/usr/bin/env python3
"""Download sample images from the exact ImageNet-100 synsets used in the KARL paper.

Tries two sources in order:
  1. image-net.org synset image list API (original ImageNet URLs)
  2. Hugging Face clane9/imagenet-100 dataset (fallback)

Usage:
    python download_imagenet100_samples.py --output_dir sample_images --num_per_class 1
"""

import argparse
import os
import random
import urllib.request
import ssl
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Exact 100 synsets from karl/run_scripts/create_imagenet100.py
SYNSETS = [
    "n01558993", "n02085620", "n02106550", "n02259212", "n03032252", "n03764736", "n04099969", "n04589890",
    "n01692333", "n02086240", "n02107142", "n02326432", "n03062245", "n03775546", "n04111531", "n04592741",
    "n01729322", "n02086910", "n02108089", "n02396427", "n03085013", "n03777754", "n04127249", "n07714571",
    "n01735189", "n02087046", "n02109047", "n02483362", "n03259280", "n03785016", "n04136333", "n07715103",
    "n01749939", "n02089867", "n02113799", "n02488291", "n03379051", "n03787032", "n04229816", "n07753275",
    "n01773797", "n02089973", "n02113978", "n02701002", "n03424325", "n03794056", "n04238763", "n07831146",
    "n01820546", "n02090622", "n02114855", "n02788148", "n03492542", "n03837869", "n04336792", "n07836838",
    "n01855672", "n02091831", "n02116738", "n02804414", "n03494278", "n03891251", "n04418357", "n13037406",
    "n01978455", "n02093428", "n02119022", "n02859443", "n03530642", "n03903868", "n04429376", "n13040303",
    "n01980166", "n02099849", "n02123045", "n02869837", "n03584829", "n03930630", "n04435653",
    "n01983481", "n02100583", "n02138441", "n02877765", "n03594734", "n03947888", "n04485082",
    "n02009229", "n02104029", "n02172182", "n02974003", "n03637318", "n04026417", "n04493381",
    "n02018207", "n02105505", "n02231487", "n03017168", "n03642806", "n04067472", "n04517823",
]

# Unverified SSL context for image-net.org
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _download_file(url, dest, timeout=10):
    """Download a single URL to dest. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            data = resp.read()
            if len(data) < 1000:  # skip tiny/broken files
                return False
            with open(dest, "wb") as f:
                f.write(data)
        return True
    except Exception:
        return False


def download_via_imagenet_api(output_dir, num_per_class, total_target):
    """Try downloading from image-net.org synset image URL lists."""
    print("    Source 1: image-net.org synset image lists...")
    train_dir = os.path.join(output_dir, "train")
    downloaded = 0
    synset_list = list(SYNSETS)
    random.shuffle(synset_list)

    for synset in synset_list:
        if downloaded >= total_target:
            break
        class_dir = os.path.join(train_dir, synset)
        os.makedirs(class_dir, exist_ok=True)

        # Check if we already have enough for this class
        existing = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))])
        if existing >= num_per_class:
            downloaded += existing
            continue

        # Fetch URL list for this synset
        api_url = f"https://image-net.org/api/imagenet.synset.geturls?wnid={synset}"
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                urls = resp.read().decode("utf-8", errors="ignore").strip().split("\n")
            urls = [u.strip() for u in urls if u.strip().startswith("http")]
            random.shuffle(urls)
        except Exception:
            continue

        count = 0
        for url in urls[:num_per_class * 5]:  # try up to 5x to get enough
            if count >= num_per_class:
                break
            dest = os.path.join(class_dir, f"{synset}_{count:03d}.jpg")
            if _download_file(url, dest):
                count += 1
                downloaded += 1
                print(f"\r    Downloaded {downloaded}/{total_target} images", end="", flush=True)

    print()
    return downloaded


def download_via_huggingface(output_dir, num_per_class, total_target, already_have):
    """Fallback: download from Hugging Face clane9/imagenet-100 dataset."""
    print("    Source 2: Hugging Face clane9/imagenet-100 dataset...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("    Installing datasets library...")
        import subprocess
        subprocess.check_call(["pip3", "install", "--quiet", "datasets"])
        from datasets import load_dataset

    train_dir = os.path.join(output_dir, "train")
    ds = load_dataset("clane9/imagenet-100", split="train", streaming=True)

    # The HF dataset uses integer labels; we need to map them back.
    # Download images and organize by integer label, then we'll have
    # a valid ImageFolder structure regardless of synset mapping.
    downloaded = already_have
    class_counts = {}

    for sample in ds:
        if downloaded >= total_target:
            break
        label = sample["label"]
        class_name = f"class_{label:04d}"
        cc = class_counts.get(class_name, 0)
        if cc >= num_per_class:
            continue

        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        dest = os.path.join(class_dir, f"{class_name}_{cc:03d}.jpg")
        img = sample["image"]
        img = img.convert("RGB")
        img.save(dest, "JPEG", quality=95)
        class_counts[class_name] = cc + 1
        downloaded += 1
        print(f"\r    Downloaded {downloaded}/{total_target} images", end="", flush=True)

    print()
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download ImageNet-100 sample images for KARL demo")
    parser.add_argument("--output_dir", type=str, default="sample_images")
    parser.add_argument("--num_images", type=int, default=50, help="Total number of images to download")
    parser.add_argument("--num_per_class", type=int, default=1, help="Max images per class")
    args = parser.parse_args()

    total_target = args.num_images
    train_dir = os.path.join(args.output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    # Count existing images
    existing = sum(
        1 for root, _, files in os.walk(train_dir)
        for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if existing >= total_target:
        print(f"    Already have {existing} images in {train_dir}, skipping download.")
        return

    # Try image-net.org first
    downloaded = download_via_imagenet_api(args.output_dir, args.num_per_class, total_target)

    # Fall back to HF if we didn't get enough
    if downloaded < total_target:
        print(f"    Got {downloaded}/{total_target} from image-net.org, trying Hugging Face fallback...")
        downloaded = download_via_huggingface(args.output_dir, args.num_per_class, total_target, downloaded)

    print(f"    Done. {downloaded} images saved to {train_dir}/")


if __name__ == "__main__":
    main()
