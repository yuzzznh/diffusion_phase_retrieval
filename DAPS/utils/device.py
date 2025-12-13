"""
Device utilities for TPU/CUDA abstraction.

TPU (PyTorch XLA) vs CUDA를 통합 관리하는 유틸리티 모듈.
use_tpu flag에 따라 적절한 device와 관련 함수들을 제공합니다.
"""

import torch
import numpy as np

# TPU 관련 모듈은 조건부 import (CUDA 환경에서는 torch_xla가 없을 수 있음)
_xm = None
_xla_available = False

def _lazy_import_xla():
    """Lazy import for torch_xla to avoid import errors on CUDA-only systems."""
    global _xm, _xla_available
    if _xm is None:
        try:
            import torch_xla.core.xla_model as xm
            _xm = xm
            _xla_available = True
        except ImportError:
            _xla_available = False
    return _xm, _xla_available


def get_device(use_tpu: bool, gpu_id: int = 0) -> torch.device:
    """
    Get the appropriate device based on use_tpu flag.

    Args:
        use_tpu: True면 TPU device, False면 CUDA device 반환
        gpu_id: CUDA 사용 시 GPU ID (default: 0)

    Returns:
        torch.device: TPU 또는 CUDA device
    """
    if use_tpu:
        xm, available = _lazy_import_xla()
        if not available:
            raise RuntimeError("use_tpu=True but torch_xla is not installed. "
                             "Install with: pip install torch-xla")
        return xm.xla_device()
    else:
        return torch.device(f'cuda:{gpu_id}')


def setup_device(use_tpu: bool, gpu_id: int = 0, seed: int = 42) -> torch.device:
    """
    Initialize device and set random seeds.

    Args:
        use_tpu: True면 TPU, False면 CUDA
        gpu_id: CUDA 사용 시 GPU ID
        seed: Random seed for reproducibility

    Returns:
        torch.device: 설정된 device
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_tpu:
        xm, available = _lazy_import_xla()
        if not available:
            raise RuntimeError("use_tpu=True but torch_xla is not installed.")

        # TPU-specific seed setting
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)

        device = xm.xla_device()
        print(f"Using TPU device: {device}")
    else:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(f'cuda:{gpu_id}')
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using CUDA device: {device}")

    return device


def reset_memory_stats(use_tpu: bool) -> None:
    """
    Reset memory statistics for tracking peak usage.

    Args:
        use_tpu: True면 TPU, False면 CUDA
    """
    if use_tpu:
        # TPU: xm.get_memory_info()는 누적이 아니라 현재 상태를 반환하므로
        # 별도의 reset이 필요 없음. 하지만 명시적으로 호출 가능.
        pass
    else:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def get_memory_stats(use_tpu: bool, device: torch.device = None) -> dict:
    """
    Get memory usage statistics.

    Args:
        use_tpu: True면 TPU, False면 CUDA
        device: TPU 사용 시 device 객체 필요

    Returns:
        dict: Memory statistics with keys like 'peak_mb', 'used_mb', etc.
    """
    if use_tpu:
        xm, available = _lazy_import_xla()
        if not available or device is None:
            return {'peak_mb': 0.0, 'used_mb': 0.0, 'available': False}

        try:
            # xm.get_memory_info() returns dict with 'bytes_used', 'bytes_limit', etc.
            info = xm.get_memory_info(device)
            return {
                'peak_mb': info.get('peak_bytes_used', 0) / (1024 ** 2),
                'used_mb': info.get('bytes_used', 0) / (1024 ** 2),
                'limit_mb': info.get('bytes_limit', 0) / (1024 ** 2),
                'available': True,
                'device_type': 'tpu'
            }
        except Exception as e:
            print(f"Warning: Could not get TPU memory info: {e}")
            return {'peak_mb': 0.0, 'used_mb': 0.0, 'available': False}
    else:
        if torch.cuda.is_available():
            return {
                'peak_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
                'used_mb': torch.cuda.memory_allocated() / (1024 ** 2),
                'reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
                'available': True,
                'device_type': 'cuda'
            }
        else:
            return {'peak_mb': 0.0, 'used_mb': 0.0, 'available': False}


def mark_step(use_tpu: bool) -> None:
    """
    Mark a step for TPU lazy execution.

    TPU는 lazy execution을 사용하므로 sampling loop의 매 step 끝에서
    이 함수를 호출해야 합니다. 호출하지 않으면 그래프가 무한히 커지다가
    메모리가 터집니다.

    CUDA에서는 no-op입니다.

    Args:
        use_tpu: True면 xm.mark_step() 호출, False면 아무것도 안 함
    """
    if use_tpu:
        xm, available = _lazy_import_xla()
        if available:
            _xm.mark_step()


def synchronize(use_tpu: bool) -> None:
    """
    Synchronize device (wait for all operations to complete).

    Args:
        use_tpu: True면 TPU sync, False면 CUDA sync
    """
    if use_tpu:
        xm, available = _lazy_import_xla()
        if available:
            _xm.mark_step()
            _xm.wait_device_ops()
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def is_tpu_available() -> bool:
    """Check if TPU (torch_xla) is available."""
    _, available = _lazy_import_xla()
    return available


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()
