import torch
import numpy as np
from tqdm import tqdm

def train_gnn(model, train_loader, val_data, optimizer, criterion, config, writer):
    best_metric = -np.inf
    best_state = None
    patience = config.train.early_stopping_patience
    patience_counter = 0
    for epoch in range(1, config.train.epochs+1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(next(model.parameters()).device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            batch_size = batch['encounter'].batch_size
            loss = criterion(out[:batch_size], batch['encounter'].y[:batch_size].float())
            loss.backward()
            if config.train.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
            optimizer.step()
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / len(train_loader.dataset)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        # Validation
        if epoch % config.train.val_every == 0:
            model.eval()
            with torch.no_grad():
                x_dict = {k: v.to(next(model.parameters()).device) for k,v in val_data.x_dict.items()}
                edge_index_dict = {k: v.to(next(model.parameters()).device) for k,v in val_data.edge_index_dict.items()}
                out = model(x_dict, edge_index_dict)
                preds = torch.sigmoid(out).cpu().numpy()
                labels = val_data['encounter'].y.cpu().numpy()
                from sklearn.metrics import average_precision_score
                auprc = average_precision_score(labels, preds)
                writer.add_scalar("Val/auprc", auprc, epoch)
                # Early stopping on AUPRC
                if auprc > best_metric:
                    best_metric = auprc
                    best_state = { 'model': model.state_dict(), 'epoch': epoch, 'metric': auprc }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
    # Load best state
    if best_state:
        model.load_state_dict(best_state['model'])
    return model, best_state
