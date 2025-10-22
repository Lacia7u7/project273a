import pandas as pd
import torch
from src.graph.inductive import build_star_graph_for_row

def batch_predict(csv_path: str, model, vocabs, config):
    df_new = pd.read_csv(csv_path)
    outputs = []
    model.eval()
    for _, row in df_new.iterrows():
        star_graph = build_star_graph_for_row(row, vocabs, config)
        # move to device
        x_dict = {k: v.to(next(model.parameters()).device) for k,v in star_graph.x_dict.items()}
        edge_index_dict = {k: v.to(next(model.parameters()).device) for k,v in star_graph.edge_index_dict.items()}
        with torch.no_grad():
            logit = model(x_dict, edge_index_dict)
            prob = torch.sigmoid(logit).item()
            y_hat = 1 if prob >= 0.5 else 0
            outputs.append((row[config.data.identifier_cols.encounter_id], prob, y_hat))
    out_df = pd.DataFrame(outputs, columns=["encounter_id","pred_prob","pred_label"])
    out_df.to_csv(config.inference.output_predictions_path, index=False)
    return out_df
