from torch.optim import Adam, AdamW, SGD
import torch

def make_optimizer(params, cfg):
    opt_name = getattr(cfg.train.optimizer, "name", "AdamW")
    lr = getattr(cfg.train.optimizer, "lr", getattr(cfg.train, "learning_rate", 1e-3))
    wd = getattr(cfg.train.optimizer, "weight_decay", getattr(cfg.train, "weight_decay", 0.0))

    name = str(opt_name).lower()
    if "adamw" in name:
        opt_cls = AdamW
    elif "adam" in name:
        opt_cls = Adam
    elif "sgd" in name:
        opt_cls = SGD
    else:
        raise ValueError(f"Unsupported optimizer '{opt_name}'")

    if opt_cls is SGD:
        return opt_cls(params, lr=lr, weight_decay=wd, momentum=0.9)
    return opt_cls(params, lr=lr, weight_decay=wd)

def make_scheduler(optimizer, cfg):
    name = getattr(cfg.train.scheduler, "name", None)
    if not name:
        return None
    n = str(name).lower()
    if n == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    if n == "steplr":
        step_size = max(1, getattr(cfg.train.scheduler, "step_size", 10))
        gamma = getattr(cfg.train.scheduler, "gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unsupported scheduler '{name}'")
