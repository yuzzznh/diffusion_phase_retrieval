from abc import ABC, abstractmethod
from piq import psnr, ssim, LPIPS
import prettytable
import torch
import torch.nn as nn
import wandb
import numpy as np
import warnings


class Evaluator:
    """
        Evaluation module for computing evaluation metrics.
    """

    def __init__(self, eval_fn_list):
        """
            Initializes the evaluator with the ground truth and measurement.

            Parameters:
                eval_fn_list (tuple): List of evaluation functions to use.
        """
        super().__init__()
        self.eval_fn = {}
        for eval_fn in eval_fn_list:
            self.eval_fn[eval_fn.name] = eval_fn
        self.main_eval_fn_name = eval_fn_list[0].name

    def get_main_eval_fn(self):
        """
            return the first eval_fn by default
        """
        return self.eval_fn[self.main_eval_fn_name]

    def __call__(self, gt, measurement, x, reduction='mean'):
        """
            Computes evaluation metrics for the given input.

            Parameters:
                x (torch.Tensor): Input tensor.
                reduction (str): Reduction method ('mean' or 'none').

            Returns:
                dict: Dictionary of evaluation results.
        """
        results = {}
        for eval_fn_name, eval_fn in self.eval_fn.items():
            results[eval_fn_name] = eval_fn(gt, measurement, x, reduction)
        return results

    def to_list(self, x):
        return x.cpu().detach().tolist()

    def report(self, gt, measurement, x):
        '''x: [N, B, C, H, W] or [B, C, H, W]'''
        if len(x.shape) == 4:
            x = x[None]
        result_dicts = {}

        # eval function
        broadcasted_shape = torch.broadcast_shapes(x.shape, gt.shape)
        x0_flatten = gt.expand(broadcasted_shape).flatten(0, 1)
        x_flatten = x.expand(broadcasted_shape).flatten(0, 1)
        y_flatten = measurement.expand((broadcasted_shape[0], *measurement.shape)).flatten(0, 1)

        for key, fn in self.eval_fn.items():
            value = fn(x0_flatten, y_flatten, x_flatten, reduction='none').reshape(broadcasted_shape[0], -1)
            result_dicts[key] = {
                'sample': self.to_list(value.permute(1, 0)),
                'mean': self.to_list(value.mean(0)),
                'std': self.to_list(value.std(0) if value.shape[0] != 1 else torch.zeros_like(value.mean(0))),
                'max': self.to_list(value.max(0)[0]),
                'min': self.to_list(value.min(0)[0]),
            }
        return result_dicts

    def report_per_image(self, gt, measurement, samples):
        '''
        Evaluate a single image with multiple samples.

        Args:
            gt: [C, H, W] - single ground truth image
            measurement: [...] - single measurement
            samples: [num_samples, C, H, W] - multiple samples for this image

        Returns:
            dict: per-image evaluation results with statistics over samples
        '''
        num_samples = samples.shape[0]
        gt_expanded = gt.unsqueeze(0).expand(num_samples, -1, -1, -1)  # [num_samples, C, H, W]

        result_dict = {}
        for key, fn in self.eval_fn.items():
            values = fn(gt_expanded, measurement.unsqueeze(0).expand(num_samples, *measurement.shape),
                       samples, reduction='none')  # [num_samples]
            cmp = get_eval_fn_cmp(key)
            best_idx = values.argmax() if cmp == 'max' else values.argmin()
            result_dict[key] = {
                'samples': self.to_list(values),
                'mean': float(values.mean()),
                'std': float(values.std()) if num_samples > 1 else 0.0,
                'max': float(values.max()),
                'min': float(values.min()),
                'best': float(values[best_idx]),
                'best_idx': int(best_idx),
            }
        return result_dict

    def aggregate_results(self, per_image_results):
        '''
        Aggregate per-image results into overall statistics.

        Args:
            per_image_results: list of dicts from report_per_image

        Returns:
            dict: aggregated results with per_image and overall statistics
        '''
        result_dicts = {}

        for key in self.eval_fn.keys():
            per_image_data = [r[key] for r in per_image_results]
            best_values = [r['best'] for r in per_image_data]
            mean_values = [r['mean'] for r in per_image_data]

            result_dicts[key] = {
                'per_image': per_image_data,
                'overall': {
                    'best_mean': float(np.mean(best_values)),
                    'best_std': float(np.std(best_values)),
                    'mean_of_means': float(np.mean(mean_values)),
                    'std_of_means': float(np.std(mean_values)),
                }
            }
        return result_dicts

    def display(self, result_dicts):
        table = Table('results')
        average, std = {}, {}
        for key in result_dicts.keys():
            value = ['{:.3f}'.format(v) for v in result_dicts[key][get_eval_fn_cmp(key)]]
            table.add_column(key, value)
            average[key] = '{:.3f}'.format(np.mean(result_dicts[key][get_eval_fn_cmp(key)]))
            std[key] = '{:.3f}'.format(np.std(result_dicts[key][get_eval_fn_cmp(key)]))
        # for average
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['mean' for _ in result_dicts.keys()])
        table.add_row(average.values())
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['std' for _ in result_dicts.keys()])
        table.add_row(std.values())

        return table.get_string()

    def display_aggregated(self, result_dicts):
        '''Display aggregated results from aggregate_results().'''
        table = Table('results')

        # Get best values for each image
        for key in result_dicts.keys():
            per_image = result_dicts[key]['per_image']
            best_values = ['{:.3f}'.format(r['best']) for r in per_image]
            table.add_column(key, best_values)

        # Add overall statistics
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['best_mean' for _ in result_dicts.keys()])
        table.add_row(['{:.3f}'.format(result_dicts[key]['overall']['best_mean']) for key in result_dicts.keys()])
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['best_std' for _ in result_dicts.keys()])
        table.add_row(['{:.3f}'.format(result_dicts[key]['overall']['best_std']) for key in result_dicts.keys()])

        return table.get_string()

    def log_wandb(self, result_dicts, batch_size):
        for s in range(batch_size):
            log_dict = {key: result_dicts[key][get_eval_fn_cmp(key)][s] for key in result_dicts.keys()}
            wandb.log(log_dict)
        log_dict = {key: np.mean(result_dicts[key][get_eval_fn_cmp(key)]) for key in result_dicts.keys()}
        new_log_dict = {key + '_all': value for key, value in log_dict.items()}
        wandb.log(new_log_dict)
        return

    def log_wandb_aggregated(self, result_dicts, image_idx):
        '''Log per-image results to wandb.'''
        for key in result_dicts.keys():
            per_image = result_dicts[key]['per_image'][image_idx]
            log_dict = {
                f'{key}_best': per_image['best'],
                f'{key}_mean': per_image['mean'],
                f'{key}_std': per_image['std'],
            }
            wandb.log(log_dict)

    def log_wandb_overall(self, result_dicts):
        '''Log overall statistics to wandb.'''
        log_dict = {}
        for key in result_dicts.keys():
            overall = result_dicts[key]['overall']
            log_dict[f'{key}_best_mean'] = overall['best_mean']
            log_dict[f'{key}_best_std'] = overall['best_std']
        wandb.log(log_dict)


class Table(object):
    def __init__(self, title=None, field_names=None):
        """
            title:          str
            field_names:    list of field names
        """
        self.table = prettytable.PrettyTable(title=title, field_names=field_names)

    def add_rows(self, rows):
        """
            rows: list of tuples
        """
        self.table.add_rows(rows)

    def add_row(self, row):
        self.table.add_row(row)

    def add_column(self, fieldname, column):
        self.table.add_column(fieldname=fieldname, column=column)

    def get_string(self):
        """
            a markdown format table
        """
        _junc = self.table.junction_char
        if _junc != "|":
            self.table.junction_char = "|"
        markdown = [row for row in self.table.get_string().split("\n")[1:-1]]
        self.table.junction_char = _junc
        return "\n" + "\n".join(markdown)

    def get_latex_string(self):
        # TODO: to be done in future
        pass


__EVAL_FN__ = {}
__EVAL_FN_CMP__ = {}


def register_eval_fn(name: str):
    def wrapper(cls):
        if __EVAL_FN__.get(name, None):
            if __EVAL_FN__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __EVAL_FN__[name] = cls
        __EVAL_FN_CMP__[name] = cls.cmp
        cls.name = name
        return cls

    return wrapper


def get_eval_fn(name: str, **kwargs):
    if __EVAL_FN__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __EVAL_FN__[name](**kwargs)


def get_eval_fn_cmp(name: str):
    return __EVAL_FN_CMP__[name]


class EvalFn(ABC):
    def norm(self, x):
        return (x * 0.5 + 0.5).clip(0, 1)

    @abstractmethod
    def __call__(self, gt, measurement, sample, reduction='none'):
        pass


@register_eval_fn('psnr')
class PeakSignalNoiseRatio(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none'):
        return psnr(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('ssim')
class StructuralSimilarityIndexMeasure(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none'):
        return ssim(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('lpips')
class LearnedPerceptualImagePatchSimilarity(EvalFn):
    cmp = 'min'  # the higher, the better

    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.lpips_fn = LPIPS(replace_pooling=True, reduction='none')

    def evaluate_in_batch(self, gt, pred):
        batch_size = self.batch_size
        results = []
        for start in range(0, gt.shape[0], batch_size):
            res = self.lpips_fn(self.norm(gt[start:start+batch_size]), self.norm(pred[start:start+batch_size]))
            results.append(res)
        results = torch.cat(results, dim=0)
        return results

    def __call__(self, gt, measurement, sample, reduction='none'):
        res = self.evaluate_in_batch(gt, sample)
        if reduction == 'mean':
            res = res.mean()
        return res