import json
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from eval import get_eval_fn, Evaluator


def load_image(path, resolution=256, device='cuda'):
    """Load and preprocess image to [-1, 1] range (same as data.py)"""
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution)
    ])
    img = (trans(Image.open(path)) * 2 - 1).to(device)
    if img.shape[0] == 1:
        img = torch.cat([img] * 3, dim=0)
    return img


def main():
    parser = argparse.ArgumentParser(description="Recalculate metrics from saved samples")
    parser.add_argument("--samples", type=str, required=True, help="Path to samples folder")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth dataset folder")
    parser.add_argument("--output", type=str, default=None, help="Output path for metrics.json (default: samples/../metrics.json)")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--device", type=str, default='cuda')

    args = parser.parse_args()

    samples_path = Path(args.samples)
    gt_path = Path(args.gt)

    if args.output is None:
        output_path = samples_path.parent / 'metrics.json'
    else:
        output_path = Path(args.output)

    # Detect num_runs from sample files
    sample_files = sorted(samples_path.glob('*.png'))
    if not sample_files:
        print(f"No samples found in {samples_path}")
        return

    # Parse run IDs from filenames (format: 00000_run0000.png)
    run_ids = set()
    for f in sample_files:
        parts = f.stem.split('_run')
        if len(parts) == 2:
            run_ids.add(int(parts[1]))
    num_runs = len(run_ids)
    print(f"Detected {num_runs} run(s)")

    # Load GT images
    print("Loading ground truth images...")
    gt_files = sorted([f for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
                       for f in gt_path.rglob(ext)])[:args.num_images]
    gt_images = torch.stack([load_image(f, args.resolution, args.device) for f in gt_files])
    print(f"Loaded {len(gt_images)} GT images")

    # Load sample images for each run
    all_samples = []
    for run_id in sorted(run_ids):
        print(f"Loading samples for run {run_id}...")
        run_samples = []
        for idx in range(args.num_images):
            sample_file = samples_path / f'{idx:05d}_run{run_id:04d}.png'
            if sample_file.exists():
                run_samples.append(load_image(sample_file, args.resolution, args.device))
            else:
                print(f"Warning: {sample_file} not found")
        all_samples.append(torch.stack(run_samples))

    all_samples = torch.stack(all_samples)  # [num_runs, num_images, C, H, W]
    print(f"Sample shape: {all_samples.shape}")

    # Create evaluator using eval.py (same as posterior_sample.py)
    eval_fn_list = [get_eval_fn('psnr'), get_eval_fn('ssim'), get_eval_fn('lpips')]
    evaluator = Evaluator(eval_fn_list)

    # Compute metrics using evaluator.report() (same as posterior_sample.py)
    # evaluator.report expects: gt [B, C, H, W], measurement (unused for metrics), samples [N, B, C, H, W]
    print("Computing metrics...")
    dummy_measurement = torch.zeros_like(gt_images)  # measurement not used for psnr/ssim/lpips
    results = evaluator.report(gt_images, dummy_measurement, all_samples)

    # Save (same as posterior_sample.py:248-252)
    markdown_text = evaluator.display(results)
    eval_md_path = output_path.parent / 'eval.md'
    with open(eval_md_path, 'w') as f:
        f.write(markdown_text)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(markdown_text)
    print(f"\nSaved metrics to {output_path}")
    print(f"Saved eval to {eval_md_path}")


if __name__ == "__main__":
    main()
