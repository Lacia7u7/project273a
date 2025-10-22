import logging
from torch.utils.tensorboard import SummaryWriter
import os

def init_logging(log_dir: str):
    """Initialize logging and TensorBoard writer."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s:%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    writer = SummaryWriter(log_dir)
    return logging.getLogger(), writer

def log_metrics(logger, writer, metrics: dict, step: int, prefix: str = ""):
    """Log metrics to console and TensorBoard."""
    for name, value in metrics.items():
        tag = f"{prefix}{name}" if prefix else name
        logger.info(f"{tag}: {value:.4f}")
        writer.add_scalar(tag, value, step)
