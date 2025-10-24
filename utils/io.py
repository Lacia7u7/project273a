import json
import pickle
import os

import pandas as pd
import torch

def save_config(config, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o))

def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f)

def save_vocab(vocab: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)

def load_vocab(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path: str, map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model

def save_scaler(scaler, path: str):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)

def load_scaler(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load the dataset CSV into a pandas DataFrame"""
    df = pd.read_csv(csv_path)
    return df
