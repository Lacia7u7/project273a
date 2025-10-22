import os

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


def make_neighbor_loader(gdata, input_nodes, num_neighbors, config, train=True, shuffle=True):
    from torch_geometric.loader import NeighborLoader

    io = config.system.dataloader
    ncpu = os.cpu_count() or 4
    num_workers = io.num_workers if io.num_workers is not None else max(2, ncpu - 1)
    return NeighborLoader(
        gdata, input_nodes=input_nodes, num_neighbors=num_neighbors,
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
