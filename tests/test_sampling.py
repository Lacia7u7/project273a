from types import SimpleNamespace
import sys


def _make_config(fanouts, *, num_layers=2):
    dataloader_cfg = SimpleNamespace(
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        pin_memory_device="cpu",
        non_blocking=False,
    )
    return SimpleNamespace(
        model=SimpleNamespace(num_layers=num_layers),
        graph=SimpleNamespace(fanouts=fanouts),
        system=SimpleNamespace(dataloader=dataloader_cfg, cuda=SimpleNamespace(uva=False)),
        train=SimpleNamespace(batching=SimpleNamespace(batch_size_encounters=4)),
    )


def test_build_num_neighbors_respects_config(monkeypatch):
    from data import sampling

    class DummyGraph:
        edge_types = [
            ("encounter", "has_icd", "icd"),
            ("icd", "is_a", "icd_group"),
        ]

    cfg = _make_config(
        {
            "encounter__has_icd__icd": [10, 5],
            "reverse_edges": [2, 2],
        },
        num_layers=2,
    )

    out = sampling.build_num_neighbors(DummyGraph(), cfg, num_layers=2)

    assert out[("encounter", "has_icd", "icd")] == [10, 5]
    # reverse_edges entry should be ignored and fall back to default -1 fanouts
    assert out[("icd", "is_a", "icd_group")] == [-1, -1]


def test_make_neighbor_loader_uses_config_fanouts(monkeypatch):
    # Stub NeighborLoader so we do not depend on torch-geometric during tests
    captured_kwargs = {}

    class DummyLoader:
        def __init__(self, data, *, input_nodes, num_neighbors, **kwargs):
            captured_kwargs["data"] = data
            captured_kwargs["input_nodes"] = input_nodes
            captured_kwargs["num_neighbors"] = num_neighbors
            captured_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "torch_geometric.loader", SimpleNamespace(NeighborLoader=DummyLoader))

    from train.loader import make_neighbor_loader

    class DummyGraph:
        edge_types = [("encounter", "rel", "drug")]

    cfg = _make_config({"encounter__rel__drug": [3, 1]}, num_layers=2)
    graph = DummyGraph()
    input_nodes = ("encounter", [0, 1, 2])

    loader = make_neighbor_loader(graph, input_nodes, num_neighbors=None, config=cfg, shuffle=False)

    assert captured_kwargs["num_neighbors"] == {("encounter", "rel", "drug"): [3, 1]}
    # The returned object should be our DummyLoader instance
    assert isinstance(loader, DummyLoader)
