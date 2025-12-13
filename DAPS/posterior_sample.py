import json
import yaml
import torch
from torchvision.utils import save_image
from forward_operator import get_operator
from data import get_dataset
from sampler import get_sampler, Trajectory
from model import get_model
from eval import get_eval_fn, get_eval_fn_cmp, Evaluator
from torch.nn.functional import interpolate
from pathlib import Path
from omegaconf import OmegaConf
from evaluate_fid import calculate_fid
from torch.utils.data import DataLoader
import hydra
import wandb
import setproctitle
from PIL import Image
import numpy as np
import imageio

import os


def resize(y, x, task_name):
    """
        Visualization Only: resize measurement y according to original signal image x
    """
    if y.shape != x.shape:
        ry = interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ry = y
    if task_name == 'phase_retrieval':
        def norm_01(y):
            tmp = (y - y.mean()) / y.std()
            tmp = tmp.clip(-0.5, 0.5) * 3
            return tmp

        ry = norm_01(ry) * 2 - 1
    return ry


def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir()
    return Path(dir)


def norm(x):
    """
        normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)


def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images
    """
    pils = []
    for x_ in x:
        np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils


def tensor_to_numpy(x):
    """
        [B, C, H, W] tensor -> [B, C, H, W] numpy
    """
    np_images = norm(x).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np_images.astype(np.uint8)


def save_mp4_video(gt, y, x0hat_traj, x0y_traj, xt_traj, output_path, fps=24, sec=5, space=4):
    """
        stack and save trajectory as mp4 video
    """
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    ix, iy = x0hat_traj.shape[-2:]
    reindex = np.linspace(0, len(xt_traj) - 1, sec * fps).astype(int)
    np_x0hat_traj = tensor_to_numpy(x0hat_traj[reindex])
    np_x0y_traj = tensor_to_numpy(x0y_traj[reindex])
    np_xt_traj = tensor_to_numpy(xt_traj[reindex])
    np_y = tensor_to_numpy(y[None])[0]
    np_gt = tensor_to_numpy(gt[None])[0]
    for x0hat, x0y, xt in zip(np_x0hat_traj, np_x0y_traj, np_xt_traj):
        canvas = np.ones((ix, 5 * iy + 4 * space, 3), dtype=np.uint8) * 255
        cx = cy = 0
        canvas[cx:cx + ix, cy:cy + iy] = np_y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = np_gt

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0hat

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = xt
        writer.append_data(canvas)
    writer.close()


def sample_per_image(sampler, model, operator, evaluator, image, measurement, num_samples, args, root, image_idx):
    """
    Generate num_samples samples for a single image.

    Args:
        image: [C, H, W] - single ground truth image
        measurement: [...] - single measurement
        num_samples: number of samples to generate (default: 4)

    Returns:
        samples: [num_samples, C, H, W]
        per_image_result: dict with evaluation metrics
        trajs: trajectory data (if recorded)
    """
    # Replicate image and measurement for batch processing
    image_batch = image.unsqueeze(0).expand(num_samples, -1, -1, -1)  # [num_samples, C, H, W]
    y_batch = measurement.unsqueeze(0).expand(num_samples, *measurement.shape)  # [num_samples, ...]

    # Generate starting noise
    x_start = sampler.get_start(num_samples, model)  # [num_samples, C, H, W]

    # Sample
    record = args.save_traj
    samples = sampler.sample(model, x_start, operator, y_batch, evaluator,
                            verbose=True, record=record, gt=image_batch)

    trajs = None
    if record:
        trajs = sampler.trajectory.compile()

    # Evaluate this image's samples
    per_image_result = evaluator.report_per_image(image, measurement, samples)

    # Save individual samples
    if args.save_samples:
        pil_image_list = tensor_to_pils(samples)
        image_dir = safe_dir(root / 'samples')
        for sample_idx, pil_img in enumerate(pil_image_list):
            image_path = image_dir / '{:05d}_sample{:02d}.png'.format(image_idx, sample_idx)
            pil_img.save(str(image_path))

    # Save trajectory
    if args.save_traj and trajs is not None:
        traj_dir = safe_dir(root / 'trajectory')
        x0hat_traj = trajs.tensor_data['x0hat']
        x0y_traj = trajs.tensor_data['x0y']
        xt_traj = trajs.tensor_data['xt']
        cur_resized_y = resize(y_batch, samples, args.task[args.task_group].operator.name)
        slices = np.linspace(0, len(x0hat_traj)-1, 10).astype(int)
        slices = np.unique(slices)

        for sample_idx in range(num_samples):
            if args.save_traj_video:
                video_path = str(traj_dir / '{:05d}_sample{:02d}.mp4'.format(image_idx, sample_idx))
                save_mp4_video(samples[sample_idx], cur_resized_y[sample_idx],
                              x0hat_traj[:, sample_idx], x0y_traj[:, sample_idx],
                              xt_traj[:, sample_idx], video_path)
            # save long grid images
            selected_traj_grid = torch.cat([x0y_traj[slices, sample_idx],
                                           x0hat_traj[slices, sample_idx],
                                           xt_traj[slices, sample_idx]], dim=0)
            traj_grid_path = str(traj_dir / '{:05d}_sample{:02d}.png'.format(image_idx, sample_idx))
            save_image(selected_traj_grid * 0.5 + 0.5, fp=traj_grid_path, nrow=len(slices))

    return samples, per_image_result, trajs


@hydra.main(version_base='1.3', config_path='configs', config_name='default.yaml')
def main(args):
    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device('cuda:{}'.format(args.gpu))

    setproctitle.setproctitle(args.name)
    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # get data
    dataset = get_dataset(**args.data)
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # get operator & measurement
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # get sampler
    sampler = get_sampler(**args.sampler, mcmc_sampler_config=task_group.mcmc_sampler_config)

    # get model
    model = get_model(**args.model)

    # get evaluator
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    # log hyperparameters and configurations
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = safe_dir(Path(args.save_dir))
    root = safe_dir(save_dir / args.name)
    with open(str(root / 'config.yaml'), 'w') as file:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), file, default_flow_style=False, allow_unicode=True)

    # logging to wandb
    if args.wandb:
        wandb.init(
            project=args.project_name,
            name=args.name,
            config=OmegaConf.to_container(args, resolve=True)
        )

    # +++++++++++++++++++++++++++++++++++
    # main sampling process (per-image with num_samples)
    num_samples = args.num_samples  # default: 4
    all_samples = []  # [N, num_samples, C, H, W]
    all_trajs = []
    per_image_results = []

    for img_idx in range(total_number):
        print(f'Processing image {img_idx + 1}/{total_number}')

        samples, per_image_result, trajs = sample_per_image(
            sampler, model, operator, evaluator,
            image=images[img_idx],
            measurement=y[img_idx],
            num_samples=num_samples,
            args=args,
            root=root,
            image_idx=img_idx
        )

        all_samples.append(samples)  # [num_samples, C, H, W]
        per_image_results.append(per_image_result)
        if trajs is not None:
            all_trajs.append(trajs)

        # Log per-image metrics
        main_metric = evaluator.main_eval_fn_name
        print(f'  Image {img_idx}: {main_metric} best={per_image_result[main_metric]["best"]:.3f}, '
              f'mean={per_image_result[main_metric]["mean"]:.3f}, std={per_image_result[main_metric]["std"]:.3f}')

        if args.wandb:
            wandb.log({
                f'{main_metric}_best': per_image_result[main_metric]['best'],
                f'{main_metric}_mean': per_image_result[main_metric]['mean'],
                'image_idx': img_idx
            })

    # Stack all samples: [N, num_samples, C, H, W]
    all_samples = torch.stack(all_samples, dim=0)

    # Aggregate results
    results = evaluator.aggregate_results(per_image_results)

    # Display and save metrics
    markdown_text = evaluator.display_aggregated(results)
    with open(str(root / 'eval.md'), 'w') as file:
        file.write(markdown_text)
    json.dump(results, open(str(root / 'metrics.json'), 'w'), indent=4)
    print(markdown_text)

    if args.wandb:
        evaluator.log_wandb_overall(results)

    # log grid results: [gt, y, sample0, sample1, sample2, sample3] for each image
    resized_y = resize(y, images, args.task[args.task_group].operator.name)
    # Interleave: for each image, show gt, y, then all samples
    grid_rows = []
    for img_idx in range(total_number):
        grid_rows.append(images[img_idx])
        grid_rows.append(resized_y[img_idx])
        for s in range(num_samples):
            grid_rows.append(all_samples[img_idx, s])
    stack = torch.stack(grid_rows, dim=0)
    save_image(stack * 0.5 + 0.5, fp=str(root / 'grid_results.png'), nrow=2 + num_samples)

    # save raw trajectory data
    if args.save_traj_raw_data and len(all_trajs) > 0:
        traj_dir = safe_dir(root / 'trajectory')
        traj_raw_data = safe_dir(traj_dir / 'raw')
        for img_idx, traj in enumerate(all_trajs):
            print(f'saving trajectory for image {img_idx}...')
            torch.save(traj, str(traj_raw_data / 'trajectory_img{:05d}.pth'.format(img_idx)))

    # evaluate FID score
    if args.eval_fid:
        print('Calculating FID...')
        fid_dir = safe_dir(root / 'fid')

        # Select best samples for each image based on metrics
        best_samples = []
        for img_idx in range(total_number):
            best_idx = per_image_results[img_idx][evaluator.main_eval_fn_name]['best_idx']
            best_samples.append(all_samples[img_idx, best_idx])
        best_samples = torch.stack(best_samples, dim=0)  # [N, C, H, W]

        # save the best samples
        best_sample_dir = safe_dir(fid_dir / 'best_sample')
        pil_image_list = tensor_to_pils(best_samples)
        for idx in range(len(pil_image_list)):
            image_path = best_sample_dir / '{:05d}.png'.format(idx)
            pil_image_list[idx].save(str(image_path))

        fake_dataset = get_dataset(args.data.name, resolution=args.data.resolution, root=str(best_sample_dir))
        real_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        fake_loader = DataLoader(fake_dataset, batch_size=100, shuffle=False)

        fid_score = calculate_fid(real_loader, fake_loader)
        print(f'FID Score: {fid_score.item():.4f}')
        with open(str(fid_dir / 'fid.txt'), 'w') as file:
            file.write(f'FID Score: {fid_score.item():.4f}')
        if args.wandb:
            wandb.log({'FID': fid_score.item()})

    print(f'finish {args.name}!')


if __name__ == '__main__':
    main()
