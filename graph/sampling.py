from torch_geometric.loader import NeighborLoader

def get_neighbor_loader(data, input_nodes, fanouts_per_layer: dict, batch_size: int, num_workers: int = 0):
    # Prepare num_neighbors in required format for NeighborLoader
    num_neighbors = {}
    for edge_type, fanouts in fanouts_per_layer.items():
        # edge_type is in format "src__rel__dst"
        parts = edge_type.split("__")
        if len(parts) == 3:
            et = (parts[0], parts[1], parts[2])
            num_neighbors[et] = fanouts
    loader = NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True
    )
    return loader
