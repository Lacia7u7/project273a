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
