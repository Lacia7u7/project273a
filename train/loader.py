import os
from typing import Any

from train.sampling import build_num_neighbors


def make_loader(ds, config, train=True, shuffle=True, sampler=None):
    from torch.utils.data import DataLoader
    io = config.system.dataloader
    ncpu = os.cpu_count() or 4
    num_workers = io.num_workers if io.num_workers is not None else max(2, ncpu - 1)
    return DataLoader(
        ds,
        batch_size=config.train.batching.batch_size_encounters if train else 2*config.train.batching.batch_size_encounters,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=io.prefetch_factor,
        persistent_workers=io.persistent_workers,
        pin_memory=io.pin_memory,
        pin_memory_device=io.pin_memory_device
    )


def _resolve_num_neighbors(
    gdata: Any,
    config,
    num_neighbors: Any,
) -> Any:
    """
    Determine the `num_neighbors` argument for ``NeighborLoader``.

    ``config.graph.fanouts`` stores per-edge fanouts as
    ``{"src__rel__dst": [fanout_l1, fanout_l2, ...]}``.  Historically the
    caller had to translate that mapping manually into PyG's expected
    ``{(src, rel, dst): [...]}`` format.  If ``num_neighbors`` is ``None`` we
    now perform that conversion automatically so the config values are always
    honoured.

    ``build_num_neighbors`` also handles padding/truncation to match the
    configured number of layers and fills in ``-1`` for edge types without an
    explicit fanout.  When ``train`` is ``False`` we still respect the config
    values; the caller can override by passing ``num_neighbors`` explicitly.
    """

    if num_neighbors is not None:
        return num_neighbors

    # ``num_layers`` defaults to 1 if the model does not expose it (e.g. MLPs).
    num_layers = getattr(getattr(config, "model", object()), "num_layers", 1)
    return build_num_neighbors(gdata, config, num_layers)


def make_neighbor_loader(gdata, input_nodes, num_neighbors, config, train=True, shuffle=True):
    from torch_geometric.loader import NeighborLoader

    resolved_num_neighbors = _resolve_num_neighbors(gdata, config, num_neighbors)

    io = config.system.dataloader
    ncpu = os.cpu_count() or 4
    num_workers = io.num_workers if io.num_workers is not None else max(2, ncpu - 1)
    return NeighborLoader(
        gdata,
        input_nodes=input_nodes,
        num_neighbors=resolved_num_neighbors,
        batch_size=config.train.batching.batch_size_encounters,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=io.prefetch_factor,
        persistent_workers=io.persistent_workers,
        pin_memory=io.pin_memory,
        pin_memory_device=io.pin_memory_device
    )


def make_dgl_loader(g, train_nids, fanouts, config):
    from dgl.dataloading import NodeDataLoader, MultiLayerNeighborSampler
    io = config.system.dataloader
    ncpu = os.cpu_count() or 4
    num_workers = io.num_workers if io.num_workers is not None else max(2, ncpu - 1)
    sampler = MultiLayerNeighborSampler(fanouts)
    return NodeDataLoader(
        g, train_nids, sampler,
        batch_size=config.train.batching.batch_size_encounters,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=io.persistent_workers,
        use_uva=config.system.cuda.uva
    )
