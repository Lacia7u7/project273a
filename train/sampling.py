def build_num_neighbors(graph, cfg, num_layers: int):
    """
    Convert config.graph.fanouts into NeighborLoader's mapping:
      { (src, rel, dst): [k_l1, k_l2, ...] }
    - Ignores non-edge keys (e.g., 'reverse_edges')
    - Pads/truncates per-edge lists to num_layers
    - Defaults to [-1]*num_layers for missing edge types
    """
    raw = getattr(cfg.graph, "fanouts", {}) or {}

    def pad_to_L(v, L):
        lst = list(v)
        if len(lst) < L:
            lst = lst + [lst[-1] if lst else -1] * (L - len(lst))
        elif len(lst) > L:
            lst = lst[:L]
        return lst

    base = {}
    for k, v in raw.items():
        parts = k.split("__")
        if len(parts) == 3:
            base[(parts[0], parts[1], parts[2])] = pad_to_L(v, num_layers)

    default = [-1] * num_layers
    final = {}
    for et in graph.edge_types:
        final[et] = base.get(et, default)
    return final
