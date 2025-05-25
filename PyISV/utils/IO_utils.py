
import re, os, logging, numpy as np
from pathlib import Path
from typing import Optional
import torch
from torch.utils.tensorboard.writer import SummaryWriter

def setup_tensorboard_writer(logs_dir: Path | str, run_id: str | int) -> tuple[SummaryWriter, str | Path]:
    from torch.utils.tensorboard import SummaryWriter
    tb_log_dir = f"{logs_dir}/tensorboard_{run_id}"
    writer = SummaryWriter(log_dir=tb_log_dir)
    return writer, tb_log_dir

def setup_logging(log_file: Optional[str] = None, 
                  log_level: int = logging.INFO, 
                  log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
    """Set up logging configuration. If log_file is provided, logs will be written to that file.
    Removes all existing handlers to ensure this config takes effect."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename="train.log" if log_file is None else log_file,
        level=log_level,
        format=log_format
    )

def is_main_process() -> bool:
    # For torch.distributed, rank 0 is the main process
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # For SLURM or other multi-process setups, check environment variables
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"]) == 0
    # Fallback: single process
    return True

def load_tensor(file_path: str) -> torch.Tensor:
    if file_path.endswith('.npy'):
        arr = np.load(file_path)
        tensor = torch.from_numpy(arr).float()
    else:
        tensor = torch.load(file_path).float()
    # Only unsqueeze if 2D
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    return tensor

def log_gpu_memory_usage(device: torch.device | None = None, prefix: str = "") -> None:
    """Logs current, reserved, and max allocated GPU memory for the given device (default: current device)."""
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.cuda.current_device() # type: ignore
    else:
        device = device.index if hasattr(device, 'index') else device # type: ignore
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    msg = (
        f"{prefix}GPU Memory (device {device}): "
        f"allocated={allocated:.2f}MB, reserved={reserved:.2f}MB, max_allocated={max_allocated:.2f}MB"
    )
    logging.info(msg)

class RegexFilter(logging.Filter):
    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = re.compile(pattern)
    def filter(self, record: logging.LogRecord) -> bool:
        return not self.pattern.search(record.getMessage())
