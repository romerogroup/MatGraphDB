import copy

import numpy as np
import pandas as pd
import torch


class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn, device, optimizer_kwargs=None):
        self.model = model.to(device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        self.loss_fn = loss_fn
        self.device = device

        # Best model tracking
        self.best_loss = float("inf")
        self.best_model_state = None
        self.best_epoch = 0

    def train_step(self, loader):
        self.model.train()
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.model.compute_loss(data, outputs, self.loss_fn)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def eval_step(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                outputs = self.model(data)
                loss = self.model.compute_loss(data, outputs, self.loss_fn)
                total_loss += loss.item()
        return total_loss / len(loader)

    def _update_best_model(self, current_loss, epoch):
        """Internal method to track best model"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(self.model.state_dict())

    def restore_best_model(self):
        """Load the best performing model weights"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        else:
            raise RuntimeError("No best model available yet")

    def predict(self, loader, apply_inverse_transform=False):
        """Generate predictions and return DataFrame with task-specific outputs"""
        self.model.eval()
        predictions = []
        actuals = []
        task_type = getattr(self.model, "task_type", "regression")

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                outputs = self.model(data)

                if task_type == "regression":
                    if apply_inverse_transform:
                        outputs = torch.exp(outputs)
                        targets = torch.exp(data.y)
                    else:
                        targets = data.y
                else:
                    # Handle classification outputs
                    if task_type == "binary":
                        outputs = torch.sigmoid(outputs)
                        targets = data.y.long()
                    else:  # multiclass
                        outputs = torch.softmax(outputs, dim=1)
                        targets = data.y.long()

                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())

        return pd.DataFrame(
            {
                "predicted": np.concatenate(predictions).squeeze(),
                "actual": np.concatenate(actuals).squeeze(),
            }
        )
