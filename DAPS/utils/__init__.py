from .device import (
    get_device,
    setup_device,
    reset_memory_stats,
    get_memory_stats,
    mark_step,
    synchronize,
    is_tpu_available,
    is_cuda_available,
)

__all__ = [
    'get_device',
    'setup_device',
    'reset_memory_stats',
    'get_memory_stats',
    'mark_step',
    'synchronize',
    'is_tpu_available',
    'is_cuda_available',
]
