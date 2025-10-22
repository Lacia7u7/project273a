import os
import sys

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.config import Config


def get_model_class(config):
    arch = config.model.arch.upper()
    if arch == "HGT":
        from models.hgt import HGTModel as ModelClass
    elif arch == "RGCN":
        from models.rgcn import RGCNModel as ModelClass
    elif arch == "GRAPHSAGE":
        from models.sage_hetero import GraphSAGEModel as ModelClass
    else:
        raise ValueError(f"Unknown GNN architecture '{config.model.arch}' in config.model.arch")
    return ModelClass

def compile(model: nn.Module, config:Config, logger):
        # Only attempt if user actually requested a compile mode
        want_compile = bool(config.system.cuda.compile_mode)
        if not want_compile:
            return model

        # Skip on unsupported platforms (your error shows this case)
        unsupported_reasons = []
        if sys.platform == "win32":
            unsupported_reasons.append("Windows")
        if sys.version_info >= (3, 12):
            unsupported_reasons.append(f"Python {sys.version_info.major}.{sys.version_info.minor}")

        if unsupported_reasons:
            if logger:
                logger.info("Skipping torch.compile (%s not supported). "
                            "Set system.cuda.compile_mode=None to silence this permanently.",
                            ", ".join(unsupported_reasons))
            return model

        # Try to compile; fall back cleanly if anything goes wrong
        try:
            model = torch.compile(
                model,
                mode=config.system.cuda.compile_mode,
                fullgraph=config.system.cuda.compile_fullgraph,
            )
            if logger:
                logger.info("torch.compile enabled (mode=%s, fullgraph=%s).",
                            config.system.cuda.compile_mode, config.system.cuda.compile_fullgraph)
        except Exception as e:
            if logger:
                logger.info("torch.compile failed (%s). Continuing without it.", repr(e))
        return model

def setup_and_compile_model(model: nn.Module, config: Config, logger):
    # Optional: torch.compile for faster eager
    model = compile(model, config, logger)
    # Optional: single-node multi-GPU with DDP
    if config.system.ddp.enabled:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=config.system.ddp.backend)
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=config.system.ddp.find_unused_parameters,
            gradient_as_bucket_view=config.system.ddp.gradient_as_bucket_view,
            broadcast_buffers=config.system.ddp.broadcast_buffers,
            static_graph=config.system.ddp.static_graph
        )
    return model