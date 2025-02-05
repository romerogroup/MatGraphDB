import os

import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from matgraphdb.pyg.models.heterograph_encoder.metrics import (
    LearningCurve,
    ROCCurve,
    plot_pca,
)
from matgraphdb.pyg.models.trainer import BaseTrainer, callbacks


class Trainer(BaseTrainer):

    def train_step(self, data_batch, **kwargs) -> float:
        """
        Performs a validation step.

        Args:
            data_batch: A batch of data from the validation DataLoader.

        This method should:
            - Iterate over self.train_loader.
            - Compute the model output and loss using self.loss_fn.
            - Perform backpropagation and an optimizer step using self.optimizer.
            - Return an aggregated training loss (e.g., the average loss over the epoch).
        Returns:
            float: The training loss for the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(
            data_batch.x_dict,
            data_batch.edge_index_dict,
            data_batch["materials", "elements"].edge_label_index,
            node_ids={
                "materials": data_batch["materials"].node_ids,
                "elements": data_batch["elements"].node_id,
            },
        )
        target = data_batch["materials", "elements"].edge_label

        loss = self.loss_fn(pred, target)

        loss.backward()
        self.optimizer.step()
        return float(loss.cpu().float())

    def validation_step(self, data_batch, **kwargs):
        """
        Performs a validation step.
        Args:
            data_batch: A batch of data from the validation DataLoader.

        This method should:
            - Use the provided data_batch from a validation DataLoader.
            - Compute the model output and loss (without performing backpropagation).
            - Return an aggregated validation loss.
        Returns:
            float: The validation loss.
            torch.Tensor: The validation output.
            torch.Tensor: The validation target.
        """
        self.model.eval()
        pred = self.model(
            data_batch.x_dict,
            data_batch.edge_index_dict,
            data_batch["materials", "elements"].edge_label_index,
            node_ids={
                "materials": data_batch["materials"].node_ids,
                "elements": data_batch["elements"].node_id,
            },
        )
        target = data_batch["materials", "elements"].edge_label
        loss = self.loss_fn(pred, target)
        return float(loss.cpu().float()), pred, target

    def eval_metrics(self, preds, targets, **kwargs):
        # Calculate metrics
        pred_binary = (preds > 0.5).float()
        accuracy = (pred_binary == targets).float().mean()

        # Calculate precision, recall, f1
        true_positives = (pred_binary * targets).sum()
        true_negatives = ((1 - pred_binary) * (1 - targets)).sum()
        false_positives = (pred_binary * (1 - targets)).sum()
        false_negatives = ((1 - pred_binary) * targets).sum()

        predicted_positives = pred_binary.sum()
        actual_positives = targets.sum()

        precision = true_positives / (predicted_positives + 1e-10)
        recall = true_positives / (actual_positives + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        auc_score = roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy())

        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc_score),
            "true_positives": float(true_positives),
            "true_negatives": float(true_negatives),
            "false_positives": float(false_positives),
            "false_negatives": float(false_negatives),
            "predicted_positives": float(predicted_positives),
            "actual_positives": float(actual_positives),
        }

        return results


@callbacks
def learning_curve(trainer):
    learning_curve = LearningCurve()

    for split_label in trainer.split_names:
        epochs = trainer.epochs
        loss_values = trainer.data_splits_metrics[split_label]["loss"]

        # Add main model curve
        learning_curve.add_curve(
            epochs, loss_values, split_label, split_label, is_baseline=False
        )

    # Generate and save plots
    learning_curve.plot()
    learning_curve.save(os.path.join(trainer.epoch_dir, "learning_curve.png"))
    learning_curve.save(os.path.join(trainer.run_dir, "learning_curve_current.png"))
    learning_curve.close()


@callbacks
def roc_curve(trainer):
    roc_curve_plot = ROCCurve()

    for split_label in trainer.split_names:
        pred = trainer.current_preds
        target = trainer.current_targets

        # Add main model curve
        roc_curve_plot.add_curve(
            pred, target, split_label, split_label, is_baseline=False
        )

    roc_curve_plot.plot()
    roc_curve_plot.save(os.path.join(trainer.epoch_dir, "roc_curve.png"))
    roc_curve_plot.save(os.path.join(trainer.run_dir, "roc_curve_current.png"))
    roc_curve_plot.close()


@callbacks
def pca_plots(trainer):
    pca_dir = os.path.join(trainer.run_dir, "pca")

    for data_split in trainer.split_names:
        split_dir = os.path.join(pca_dir, data_split)
        os.makedirs(split_dir, exist_ok=True)

        trainer.model.eval()

        data = trainer.data_splits[data_split]

        # If user does not specify node types, use all available ones
        if selected_node_types is None:
            selected_node_types = data.node_types
        else:
            # Validate that selected_node_types is a subset
            for nt in selected_node_types:
                if nt not in data.node_types:
                    raise ValueError(f"Node type '{nt}' not found in data.node_types!")

        # 1. Extract embeddings from the trained model
        with torch.no_grad():
            z_dict = trainer.model.encode(
                data.x_dict,
                data.edge_index_dict,
                node_ids={
                    "materials": data["materials"].node_ids,
                    "elements": data["elements"].node_id,
                },
            )

        # 2. Move embeddings to CPU and convert to numpy
        z_dict = {k: v.cpu() for k, v in z_dict.items()}

        # 3. Combine embeddings only for the selected node types
        z_all = torch.cat(
            [z_dict[node_type] for node_type in selected_node_types], dim=0
        ).numpy()
        plot_pca(
            z_all,
            split_dir,
            n_components=4,
        )
